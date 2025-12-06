import os
import sys
import json
import torch
import argparse
import math
import time
from datetime import datetime
from pathlib import Path
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
    compute_cer,
    ctc_beam_search_batch,
    ids_to_text,
    compute_ctc_loss,
    analyze_blank_tokens,
    compute_ctc_entropy,
    compute_peak_statistics,
)
from onebit_asr.losses import ctc_loss_from_logits
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

def evaluate_split(model, split_name, args, tokenizer, cmvn_stats, device, blank_id=3):
    """
    Evaluate model on a specific dataset split.
    
    Args:
        model: The ASR model to evaluate
        split_name: Name of the dataset split (e.g., "test.clean")
        args: Command line arguments
        tokenizer: SentencePiece tokenizer
        cmvn_stats: CMVN statistics
        device: Device to run evaluation on
        blank_id: CTC blank token ID (default: 3)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    print(f"\nEvaluating on {split_name}...")
    
    dataset = CustomLibriSpeechDataset(
        splits=[split_name],
        tokenizer_path=args.tokenizer_path,
        cmvn_stats=cmvn_stats,
        apply_spec_augment=False
    )
    
    if len(dataset) == 0:
        print(f"Warning: Dataset {split_name} is empty!")
        return None
    
    collate_fn = CollateFunction(pad_value=0.0, label_pad_value=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    total_words = 0
    total_chars = 0
    total_samples = 0
    perfect_sentences = {'fp32': 0, '2bit': 0, '1bit': 0}
    totals = {
        'wer_fp32': 0,
        'wer_2bit': 0,
        'wer_1bit': 0,
        'cer_fp32': 0,
        'cer_2bit': 0,
        'cer_1bit': 0,
    }
    
    # CTC-specific metrics
    ctc_losses = {'fp32': [], '2bit': [], '1bit': []}
    blank_stats = {'fp32': [], '2bit': [], '1bit': []}
    entropy_stats = {'fp32': [], '2bit': [], '1bit': []}
    peak_stats = {'fp32': [], '2bit': [], '1bit': []}
    
    # Performance metrics
    inference_times = {'fp32': [], '2bit': [], '1bit': []}
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader):
            feats = batch['fbank'].to(device)
            feat_lens = batch['fbank_lengths'].to(device)
            
            # Forward pass
            # The model expects a dict
            batch_input = {
                'feats': feats,
                'feat_lens': feat_lens,
                'tokens': batch['labels'].to(device), # Not used for inference but passed to forward
                'token_lens': batch['label_lengths'].to(device)
            }
            
            # Run three precisions: 32, 2, 1 (with timing)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t0 = time.time()
            enc_fp, mask_fp, ctc_fp = model(batch_input, precision=32)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times['fp32'].append(time.time() - t0)
            
            t0 = time.time()
            enc_t2, mask_t2, ctc_t2 = model(batch_input, precision=2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times['2bit'].append(time.time() - t0)
            
            t0 = time.time()
            enc_s1, mask_s1, ctc_s1 = model(batch_input, precision=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times['1bit'].append(time.time() - t0)

            valid_fp = mask_fp.sum(dim=1).long()
            valid_t2 = mask_t2.sum(dim=1).long()
            valid_s1 = mask_s1.sum(dim=1).long()
            
            # Compute CTC losses (for evaluation, not training)
            tokens = batch['labels'].to(device)
            token_lens = batch['label_lengths'].to(device)
            try:
                loss_fp = ctc_loss_from_logits(ctc_fp, valid_fp, tokens, token_lens, blank_id)
                loss_t2 = ctc_loss_from_logits(ctc_t2, valid_t2, tokens, token_lens, blank_id)
                loss_s1 = ctc_loss_from_logits(ctc_s1, valid_s1, tokens, token_lens, blank_id)
                ctc_losses['fp32'].append(loss_fp.item())
                ctc_losses['2bit'].append(loss_t2.item())
                ctc_losses['1bit'].append(loss_s1.item())
            except Exception as e:
                # CTC loss might fail for some edge cases (e.g., empty sequences)
                if len(ctc_losses['fp32']) == 0:  # Only log first error to avoid spam
                    print(f"Warning: CTC loss computation failed for some batches: {e}")
            
            # Analyze blank tokens
            blank_fp = analyze_blank_tokens(ctc_fp, valid_fp, blank_id)
            blank_t2 = analyze_blank_tokens(ctc_t2, valid_t2, blank_id)
            blank_s1 = analyze_blank_tokens(ctc_s1, valid_s1, blank_id)
            blank_stats['fp32'].append(blank_fp)
            blank_stats['2bit'].append(blank_t2)
            blank_stats['1bit'].append(blank_s1)
            
            # Compute entropy
            entropy_fp = compute_ctc_entropy(ctc_fp, valid_fp)
            entropy_t2 = compute_ctc_entropy(ctc_t2, valid_t2)
            entropy_s1 = compute_ctc_entropy(ctc_s1, valid_s1)
            entropy_stats['fp32'].append(entropy_fp)
            entropy_stats['2bit'].append(entropy_t2)
            entropy_stats['1bit'].append(entropy_s1)
            
            # Compute peak statistics
            peak_fp = compute_peak_statistics(ctc_fp, valid_fp, blank_id)
            peak_t2 = compute_peak_statistics(ctc_t2, valid_t2, blank_id)
            peak_s1 = compute_peak_statistics(ctc_s1, valid_s1, blank_id)
            peak_stats['fp32'].append(peak_fp)
            peak_stats['2bit'].append(peak_t2)
            peak_stats['1bit'].append(peak_s1)

            hyp_fp = ctc_beam_search_batch(ctc_fp, valid_fp, beam_size=args.beam_size, blank_id=3)
            hyp_t2 = ctc_beam_search_batch(ctc_t2, valid_t2, beam_size=args.beam_size, blank_id=3)
            hyp_s1 = ctc_beam_search_batch(ctc_s1, valid_s1, beam_size=args.beam_size, blank_id=3)
            
            # Convert IDs to text
            hyp_texts_fp = []
            for ids in hyp_fp:
                # Filter out special tokens if any (0,1,2 are pad, bos, eos)      
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
                lbl = labels[i]
                valid_lbl = [int(l) for l in lbl if l != 0] # 0 is pad in collate
                text = tokenizer.decode(valid_lbl)
                ref_texts.append(text)
            
            if total_words < 100: # Print first few examples
                print(f"\nRef: {ref_texts[0]}")
                print(f"Hyp(fp32): {hyp_texts_fp[0]}")
                print(f"Hyp(2bit): {hyp_texts_t2[0]}")
                print(f"Hyp(1bit): {hyp_texts_s1[0]}")

            # Word Error Rate
            d_fp, w = compute_wer(ref_texts, hyp_texts_fp)
            d_t2, _ = compute_wer(ref_texts, hyp_texts_t2)
            d_s1, _ = compute_wer(ref_texts, hyp_texts_s1)
            totals['wer_fp32'] += d_fp
            totals['wer_2bit'] += d_t2
            totals['wer_1bit'] += d_s1
            total_words += w
            
            # Character Error Rate
            c_fp, c = compute_cer(ref_texts, hyp_texts_fp)
            c_t2, _ = compute_cer(ref_texts, hyp_texts_t2)
            c_s1, _ = compute_cer(ref_texts, hyp_texts_s1)
            totals['cer_fp32'] += c_fp
            totals['cer_2bit'] += c_t2
            totals['cer_1bit'] += c_s1
            total_chars += c
            
            # Sentence-level accuracy (perfect transcriptions)
            batch_size = len(ref_texts)
            total_samples += batch_size
            for r, h_fp, h_t2, h_s1 in zip(ref_texts, hyp_texts_fp, hyp_texts_t2, hyp_texts_s1):
                if r.strip().lower() == h_fp.strip().lower():
                    perfect_sentences['fp32'] += 1
                if r.strip().lower() == h_t2.strip().lower():
                    perfect_sentences['2bit'] += 1
                if r.strip().lower() == h_s1.strip().lower():
                    perfect_sentences['1bit'] += 1
            
    total_eval_time = time.time() - start_time
    
    # Calculate metrics
    wer_fp = totals['wer_fp32'] / total_words if total_words > 0 else 0.0
    wer_t2 = totals['wer_2bit'] / total_words if total_words > 0 else 0.0
    wer_s1 = totals['wer_1bit'] / total_words if total_words > 0 else 0.0
    
    cer_fp = totals['cer_fp32'] / total_chars if total_chars > 0 else 0.0
    cer_t2 = totals['cer_2bit'] / total_chars if total_chars > 0 else 0.0
    cer_s1 = totals['cer_1bit'] / total_chars if total_chars > 0 else 0.0
    
    sent_acc_fp = perfect_sentences['fp32'] / total_samples if total_samples > 0 else 0.0
    sent_acc_t2 = perfect_sentences['2bit'] / total_samples if total_samples > 0 else 0.0
    sent_acc_s1 = perfect_sentences['1bit'] / total_samples if total_samples > 0 else 0.0
    
    # Validate results
    assert 0 <= wer_fp <= 1.0, f"Invalid WER: {wer_fp}"
    assert 0 <= cer_fp <= 1.0, f"Invalid CER: {cer_fp}"
    assert 0 <= sent_acc_fp <= 1.0, f"Invalid sentence accuracy: {sent_acc_fp}"
    
    # Print results
    print(f"\n=== Evaluation Results for {split_name} ===")
    print(f"WER (fp32): {wer_fp:.4f} ({totals['wer_fp32']}/{total_words})")
    print(f"WER (2-bit): {wer_t2:.4f} ({totals['wer_2bit']}/{total_words})")
    print(f"WER (1-bit): {wer_s1:.4f} ({totals['wer_1bit']}/{total_words})")
    print(f"\nCER (fp32): {cer_fp:.4f} ({totals['cer_fp32']}/{total_chars})")
    print(f"CER (2-bit): {cer_t2:.4f} ({totals['cer_2bit']}/{total_chars})")
    print(f"CER (1-bit): {cer_s1:.4f} ({totals['cer_1bit']}/{total_chars})")
    print(f"\nSentence Accuracy (fp32): {sent_acc_fp:.4f} ({perfect_sentences['fp32']}/{total_samples})")
    print(f"Sentence Accuracy (2-bit): {sent_acc_t2:.4f} ({perfect_sentences['2bit']}/{total_samples})")
    print(f"Sentence Accuracy (1-bit): {sent_acc_s1:.4f} ({perfect_sentences['1bit']}/{total_samples})")
    
    # CTC-specific metrics summary
    print(f"\n=== CTC-Specific Metrics ===")
    
    # CTC Loss
    avg_ctc_fp = avg_ctc_t2 = avg_ctc_s1 = None
    if ctc_losses['fp32']:
        avg_ctc_fp = sum(ctc_losses['fp32']) / len(ctc_losses['fp32'])
        avg_ctc_t2 = sum(ctc_losses['2bit']) / len(ctc_losses['2bit'])
        avg_ctc_s1 = sum(ctc_losses['1bit']) / len(ctc_losses['1bit'])
        print(f"CTC Loss (fp32): {avg_ctc_fp:.4f}")
        print(f"CTC Loss (2-bit): {avg_ctc_t2:.4f}")
        print(f"CTC Loss (1-bit): {avg_ctc_s1:.4f}")
    
    # Blank token statistics
    mean_blank_fp = mean_blank_t2 = mean_blank_s1 = None
    if blank_stats['fp32']:
        mean_blank_fp = sum(s['mean_blank_prob'] for s in blank_stats['fp32']) / len(blank_stats['fp32'])
        mean_blank_t2 = sum(s['mean_blank_prob'] for s in blank_stats['2bit']) / len(blank_stats['2bit'])
        mean_blank_s1 = sum(s['mean_blank_prob'] for s in blank_stats['1bit']) / len(blank_stats['1bit'])
        print(f"\nMean Blank Token Probability:")
        print(f"  fp32: {mean_blank_fp:.4f}")
        print(f"  2-bit: {mean_blank_t2:.4f}")
        print(f"  1-bit: {mean_blank_s1:.4f}")
    
    # Entropy statistics
    mean_ent_fp = mean_ent_t2 = mean_ent_s1 = None
    if entropy_stats['fp32']:
        mean_ent_fp = sum(s['mean_entropy'] for s in entropy_stats['fp32']) / len(entropy_stats['fp32'])
        mean_ent_t2 = sum(s['mean_entropy'] for s in entropy_stats['2bit']) / len(entropy_stats['2bit'])
        mean_ent_s1 = sum(s['mean_entropy'] for s in entropy_stats['1bit']) / len(entropy_stats['1bit'])
        print(f"\nMean CTC Entropy (uncertainty):")
        print(f"  fp32: {mean_ent_fp:.4f}")
        print(f"  2-bit: {mean_ent_t2:.4f}")
        print(f"  1-bit: {mean_ent_s1:.4f}")
    
    # Peak statistics
    mean_peaks_fp = mean_peaks_t2 = mean_peaks_s1 = None
    if peak_stats['fp32']:
        mean_peaks_fp = sum(s['mean_peak_count'] for s in peak_stats['fp32']) / len(peak_stats['fp32'])
        mean_peaks_t2 = sum(s['mean_peak_count'] for s in peak_stats['2bit']) / len(peak_stats['2bit'])
        mean_peaks_s1 = sum(s['mean_peak_count'] for s in peak_stats['1bit']) / len(peak_stats['1bit'])
        print(f"\nMean Peak Count (non-blank predictions per utterance):")
        print(f"  fp32: {mean_peaks_fp:.2f}")
        print(f"  2-bit: {mean_peaks_t2:.2f}")
        print(f"  1-bit: {mean_peaks_s1:.2f}")
    
    # Inference time statistics
    if inference_times['fp32']:
        avg_time_fp = sum(inference_times['fp32']) / len(inference_times['fp32'])
        avg_time_t2 = sum(inference_times['2bit']) / len(inference_times['2bit'])
        avg_time_s1 = sum(inference_times['1bit']) / len(inference_times['1bit'])
        print(f"\nAverage Inference Time per Batch:")
        print(f"  fp32: {avg_time_fp*1000:.2f} ms")
        print(f"  2-bit: {avg_time_t2*1000:.2f} ms")
        print(f"  1-bit: {avg_time_s1*1000:.2f} ms")
        print(f"  Total evaluation time: {total_eval_time:.2f} seconds")
        print(f"  Speedup (2-bit vs fp32): {avg_time_fp/avg_time_t2:.2f}x")
        print(f"  Speedup (1-bit vs fp32): {avg_time_fp/avg_time_s1:.2f}x")
    
    return {
        'fp32': {
            'wer': wer_fp, 'cer': cer_fp, 'sentence_acc': sent_acc_fp,
            'ctc_loss': avg_ctc_fp,
            'blank_prob': mean_blank_fp,
            'entropy': mean_ent_fp,
            'inference_time': sum(inference_times['fp32']) / len(inference_times['fp32']) if inference_times['fp32'] else None,
        },
        '2bit': {
            'wer': wer_t2, 'cer': cer_t2, 'sentence_acc': sent_acc_t2,
            'ctc_loss': avg_ctc_t2,
            'blank_prob': mean_blank_t2,
            'entropy': mean_ent_t2,
            'inference_time': sum(inference_times['2bit']) / len(inference_times['2bit']) if inference_times['2bit'] else None,
        },
        '1bit': {
            'wer': wer_s1, 'cer': cer_s1, 'sentence_acc': sent_acc_s1,
            'ctc_loss': avg_ctc_s1,
            'blank_prob': mean_blank_s1,
            'entropy': mean_ent_s1,
            'inference_time': sum(inference_times['1bit']) / len(inference_times['1bit']) if inference_times['1bit'] else None,
        },
        'metadata': {
            'split_name': split_name,
            'total_samples': total_samples,
            'total_words': total_words,
            'total_chars': total_chars,
            'total_eval_time': total_eval_time,
        }
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
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON file')
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
    
    # Get blank_id from checkpoint or use default
    special_ids = checkpoint.get('special_ids', {})
    blank_id = special_ids.get('blank_id', 3)
    
    # Evaluate
    results = {}
    wer_clean = evaluate_split(model, "test.clean", args, sp, cmvn_stats, device, blank_id=blank_id)
    if wer_clean:
        results['test.clean'] = wer_clean
    
    wer_other = evaluate_split(model, "test.other", args, sp, cmvn_stats, device, blank_id=blank_id)
    if wer_other:
        results['test.other'] = wer_other
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if wer_clean:
        print("\nTest Clean:")
        print(f"  WER:  fp32={wer_clean['fp32']['wer']:.4f}, 2bit={wer_clean['2bit']['wer']:.4f}, 1bit={wer_clean['1bit']['wer']:.4f}")
        print(f"  CER:  fp32={wer_clean['fp32']['cer']:.4f}, 2bit={wer_clean['2bit']['cer']:.4f}, 1bit={wer_clean['1bit']['cer']:.4f}")
        print(f"  Sent: fp32={wer_clean['fp32']['sentence_acc']:.4f}, 2bit={wer_clean['2bit']['sentence_acc']:.4f}, 1bit={wer_clean['1bit']['sentence_acc']:.4f}")
    if wer_other:
        print("\nTest Other:")
        print(f"  WER:  fp32={wer_other['fp32']['wer']:.4f}, 2bit={wer_other['2bit']['wer']:.4f}, 1bit={wer_other['1bit']['wer']:.4f}")
        print(f"  CER:  fp32={wer_other['fp32']['cer']:.4f}, 2bit={wer_other['2bit']['cer']:.4f}, 1bit={wer_other['1bit']['cer']:.4f}")
        print(f"  Sent: fp32={wer_other['fp32']['sentence_acc']:.4f}, 2bit={wer_other['2bit']['sentence_acc']:.4f}, 1bit={wer_other['1bit']['sentence_acc']:.4f}")
    
    # Save results if requested
    if args.save_results and results:
        output_dir = Path(args.output_dir) if args.output_dir else Path('evaluation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_results_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = {}
        for split_name, split_results in results.items():
            json_results[split_name] = {}
            for precision in ['fp32', '2bit', '1bit']:
                json_results[split_name][precision] = {
                    k: (float(v) if isinstance(v, (torch.Tensor, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else None)
                    for k, v in split_results[precision].items()
                }
            json_results[split_name]['metadata'] = split_results.get('metadata', {})
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
