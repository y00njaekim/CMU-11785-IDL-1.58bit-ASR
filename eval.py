import os
import sys
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
from src.data.dataset import LibriSpeechDataset, CollateFunction

def levenshtein_distance(ref, hyp):
    """
    Compute Levenshtein distance between two sequences of words.
    """
    m = len(ref)
    n = len(hyp)
    
    # Create a matrix to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Deletion
                                   dp[i][j - 1],    # Insertion
                                   dp[i - 1][j - 1]) # Substitution
    return dp[m][n]

def compute_wer(refs, hyps):
    """
    Compute WER for a batch of references and hypotheses.
    refs: list of reference strings
    hyps: list of hypothesis strings
    """
    total_dist = 0
    total_words = 0
    
    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()
        dist = levenshtein_distance(r_words, h_words)
        total_dist += dist
        total_words += len(r_words)
        
    return total_dist, total_words

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

def decode_greedy(logits, vocab_size, blank_id=3):
    """
    Greedy decoding of CTC logits.
    logits: [B, T, V]
    Returns: list of token ID lists
    """
    preds = torch.argmax(logits, dim=-1) # [B, T]
    decoded = []
    for i in range(preds.size(0)):
        pred = preds[i]
        # Collapse repeats and remove blanks
        t = pred[0].item()
        tokens = []
        if t != blank_id:
            tokens.append(t)
        
        for j in range(1, len(pred)):
            t_curr = pred[j].item()
            if t_curr != blank_id and t_curr != pred[j-1].item():
                tokens.append(t_curr)
        decoded.append(tokens)
    return decoded

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
    
    total_dist = 0
    total_words = 0
    
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
            
            enc_out, enc_mask, logits_ctc = model(batch_input, precision=32)
            
            # Decode
            # blank_id is 3 based on dataloader_stub.py and dataset.py
            decoded_ids = decode_greedy(logits_ctc, model.ctc_head.out_features, blank_id=3)
            
            # Convert IDs to text
            hyp_texts = []
            for ids in decoded_ids:
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
                hyp_texts.append(text)
                
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
                print(f"Hyp: {hyp_texts[0]}")
                print(f"Pred IDs: {decoded_ids[0]}")
            
            d, w = compute_wer(ref_texts, hyp_texts)
            total_dist += d
            total_words += w
            
    wer = total_dist / total_words if total_words > 0 else 0.0
    print(f"WER on {split_name}: {wer:.4f} ({total_dist}/{total_words})")
    return wer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tokenizer_path', type=str, default='src/data/tokenizer.model')
    parser.add_argument('--cmvn_stats_path', type=str, default='src/data/cmvn_stats.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    train_args = checkpoint['args']
    
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
    print(f"Test Clean WER: {wer_clean:.4f}")
    print(f"Test Other WER: {wer_other:.4f}")

if __name__ == '__main__':
    main()
