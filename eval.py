import os
import sys
import json
import torch
import argparse
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_from_disk, concatenate_datasets, Audio
import torchaudio

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from onebit_asr.conformer import ConformerASR
from onebit_asr.metrics import (
    compute_wer,
    ctc_beam_search_batch,
    ids_to_text,
)
from src.data.dataset import LibriSpeechDataset, CollateFunction

class CustomLibriSpeechDataset(LibriSpeechDataset):
    def __init__(self, splits, *args, **kwargs):
        self.target_splits = splits
        # Pass 'test' to super, but we will override the dataset loading
        super().__init__(split="test", *args, **kwargs)

    def _load_dataset(self):
        data_dir = "data"
        datasets = []
        for split_name in self.target_splits:
            dataset_path = os.path.join(data_dir, f"{split_name}_subset")
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path not found: {dataset_path}")
                continue
            
            print(f"Loading {split_name} from {dataset_path}...")
            try:
                ds = load_from_disk(dataset_path)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {split_name}: {e}")

        if not datasets:
             raise FileNotFoundError(f"No datasets found for splits {self.target_splits}")
        return concatenate_datasets(datasets)

def _maybe_load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """If `checkpoint_path` is inside a run dir, try to load config.json there."""
    try:
        run_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        cfg_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(cfg_path):
            print(f"Found model config.json")
            with open(cfg_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def evaluate_split(model, split_name, args, tokenizer, cmvn_stats, device):
    print(f"\nEvaluating on {split_name}...")
    
    dataset = CustomLibriSpeechDataset(
        splits=[split_name],
        tokenizer_path=args.tokenizer_path,
        cmvn_stats=cmvn_stats,
        apply_spec_augment=False
    )
    
    collate_fn = CollateFunction(pad_value=0.0, label_pad_value=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    total_words = 0
    totals = {
        'wer_fp32': 0,
        'wer_2bit': 0,
        'wer_1bit': 0,
    }
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            feats = batch['fbank'].to(device)
            feat_lens = batch['fbank_lengths'].to(device)
            # We don't need tokens for the model input, but we need them for reference
            # Note: The dataset returns 'labels' which are token IDs.
            # We need to decode them back to text for WER calculation.
            
            # Forward pass
            # The model expects a dict
            batch_input = {
                'feats': feats,
                'feat_lens': feat_lens,
                'tokens': batch['labels'].to(device), # Not used for inference but passed to forward
                'token_lens': batch['label_lengths'].to(device)
            }
            
            # Use precision=32 for evaluation (or whatever the checkpoint used, but 32 is safe)
            # Or maybe 2 if it was trained with 2-bit?
            # The training script uses precision=2 for teacher and 1 for student.
            # Let's use 32 (full precision) or 2.
            # The ConformerEncoder handles precision.
            # If we want the best performance, maybe 32?
            # But the quantized layers might expect specific bitwidths.
            # Let's try 32 first.
            
            # Run three precisions: 32, 2, 1
            enc_fp, mask_fp, ctc_fp = model(batch_input, precision=32)
            enc_t2, mask_t2, ctc_t2 = model(batch_input, precision=2)
            enc_s1, mask_s1, ctc_s1 = model(batch_input, precision=1)

            valid_fp = mask_fp.sum(dim=1).long()
            valid_t2 = mask_t2.sum(dim=1).long()
            valid_s1 = mask_s1.sum(dim=1).long()

            hyp_fp = ctc_beam_search_batch(ctc_fp, valid_fp, beam_size=args.beam_size, blank_id=3)
            hyp_t2 = ctc_beam_search_batch(ctc_t2, valid_t2, beam_size=args.beam_size, blank_id=3)
            hyp_s1 = ctc_beam_search_batch(ctc_s1, valid_s1, beam_size=args.beam_size, blank_id=3)
            
            # Convert IDs to text
            hyp_texts_fp = []
            for ids in hyp_fp:
                # Filter out special tokens if any (0,1,2 are pad, bos, eos)
                # The tokenizer might handle them, but usually we just decode.
                # The dataset offsets tokens by 4. We need to subtract 4?
                # Wait, LibriSpeechDataModule does offset.
                # But LibriSpeechDataset (which we are using directly) does NOT seem to do offset in __getitem__.
                # Let's check LibriSpeechDataset.__getitem__ again.
                # It calls self.tokenizer.encode(text, out_type=int).
                # It does NOT add offset.
                # However, LibriSpeechDataModule adds offset of 4.
                # The MODEL was trained with the offset (vocab size = spm + 4).
                # So the model outputs IDs in the range [0, V+4].
                # 0=pad, 1=bos, 2=eos, 3=blank.
                # Real tokens start at 4.
                # So we need to subtract 4 before decoding with spm.
                
                valid_ids = [i - 4 for i in ids if i >= 4]
                text = tokenizer.decode(valid_ids)
                hyp_texts_fp.append(text)

            hyp_texts_t2 = []
            for ids in hyp_t2:
                valid_ids = [i - 4 for i in ids if i >= 4]
                text = tokenizer.decode(valid_ids)
                hyp_texts_t2.append(text)

            hyp_texts_s1 = []
            for ids in hyp_s1:
                valid_ids = [i - 4 for i in ids if i >= 4]
                text = tokenizer.decode(valid_ids)
                hyp_texts_s1.append(text)
                
            # Get reference text
            ref_texts = []
            labels = batch['labels'].cpu().numpy()
            for i in range(len(labels)):
                # These are from the dataset, so they are raw spm IDs (no offset).
                # Wait, let's verify.
                # LibriSpeechDataset.__getitem__: labels = self.tokenizer.encode(text, out_type=int)
                # So they are raw spm IDs.
                # But the model was trained with offset labels.
                # So when we compare, we should compare text.
                
                # Filter padding (0 is not pad in spm usually, but collate pads with 0)
                # In dataset.py: label_pad_value=0.
                # So 0 is padding.
                lbl = labels[i]
                valid_lbl = [int(l) for l in lbl if l != 0] # 0 is pad in collate
                text = tokenizer.decode(valid_lbl)
                ref_texts.append(text)
            
            if total_words < 100: # Print first few examples
                print(f"\nRef: {ref_texts[0]}")
                print(f"Hyp(fp32): {hyp_texts_fp[0]}")
                print(f"Hyp(2bit): {hyp_texts_t2[0]}")
                print(f"Hyp(1bit): {hyp_texts_s1[0]}")

            d_fp, w = compute_wer(ref_texts, hyp_texts_fp)
            d_t2, _ = compute_wer(ref_texts, hyp_texts_t2)
            d_s1, _ = compute_wer(ref_texts, hyp_texts_s1)
            totals['wer_fp32'] += d_fp
            totals['wer_2bit'] += d_t2
            totals['wer_1bit'] += d_s1
            total_words += w
            
    wer_fp = totals['wer_fp32'] / total_words if total_words > 0 else 0.0
    wer_t2 = totals['wer_2bit'] / total_words if total_words > 0 else 0.0
    wer_s1 = totals['wer_1bit'] / total_words if total_words > 0 else 0.0
    print(f"WER on {split_name} (fp32): {wer_fp:.4f} ({totals['wer_fp32']}/{total_words})")
    print(f"WER on {split_name} (2-bit): {wer_t2:.4f} ({totals['wer_2bit']}/{total_words})")
    print(f"WER on {split_name} (1-bit): {wer_s1:.4f} ({totals['wer_1bit']}/{total_words})")
    return {
        'fp32': wer_fp,
        '2bit': wer_t2,
        '1bit': wer_s1,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tokenizer_path', type=str, default='src/data/tokenizer.model')
    parser.add_argument('--cmvn_stats_path', type=str, default='src/data/cmvn_stats.pt')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # Load optional config.json from run directory to override defaults
    cfg_json = _maybe_load_config_from_checkpoint(args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint.get('args', {})

    # Prefer config file overrides if present
    if isinstance(cfg_json, dict) and cfg_json:
        # tokenizer/cmvn
        args.tokenizer_path = cfg_json.get('tokenizer_path', args.tokenizer_path)
        args.cmvn_stats_path = cfg_json.get('cmvn_stats_path', args.cmvn_stats_path)
        # model hyperparams
        for k in [
            'input_dim','enc_d_model','enc_layers','enc_heads','enc_d_ff','enc_conv_kernel','dropout',
            'dec_layers','dec_heads','dec_d_ff'
        ]:
            if k in cfg_json:
                train_args[k] = cfg_json[k]
    
    # Load tokenizer
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer {args.tokenizer_path} not found.")
        return
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)
    
    # Load CMVN stats
    if not os.path.exists(args.cmvn_stats_path):
        print(f"Error: CMVN stats {args.cmvn_stats_path} not found.")
        return
    cmvn_stats = torch.load(args.cmvn_stats_path)
    
    # Determine vocab size
    # In train.py: vocab = dm.vocab_size()
    # dm.vocab_size() = sp.get_piece_size() + 4
    vocab_size = sp.get_piece_size() + 4
    
    # Initialize model
    print("Initializing model...")
    device = torch.device(args.device)
    model = ConformerASR(
        input_dim=train_args.get('input_dim', 80),
        vocab_size=vocab_size,
        enc_d_model=train_args.get('enc_d_model', 256),
        enc_layers=train_args.get('enc_layers', 12),
        enc_heads=train_args.get('enc_heads', 4),
        enc_d_ff=train_args.get('enc_d_ff', 1024),
        enc_conv_kernel=train_args.get('enc_conv_kernel', 31),
        enc_dropout=train_args.get('dropout', 0.1),
        dec_layers=train_args.get('dec_layers', 2),
        dec_heads=train_args.get('dec_heads', 4),
        dec_d_ff=train_args.get('dec_d_ff', 1024),
        dec_dropout=train_args.get('dropout', 0.1),
        pad_id=0 # pad_id is 0
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model'])
    print("Model loaded.")
    
    # Evaluate
    wer_clean = evaluate_split(model, "test.clean", args, sp, cmvn_stats, device)
    wer_other = evaluate_split(model, "test.other", args, sp, cmvn_stats, device)
    
    print("\nSummary:")
    print(f"Test Clean WERs: fp32={wer_clean['fp32']:.4f}, 2bit={wer_clean['2bit']:.4f}, 1bit={wer_clean['1bit']:.4f}")
    print(f"Test Other WERs: fp32={wer_other['fp32']:.4f}, 2bit={wer_other['2bit']:.4f}, 1bit={wer_other['1bit']:.4f}")

if __name__ == '__main__':
    main()
