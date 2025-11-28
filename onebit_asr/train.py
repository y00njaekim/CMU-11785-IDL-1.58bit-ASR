
# --------------------------------------------------------------
# File: train.py
# --------------------------------------------------------------
import os
import sys
import math
import argparse
import json
from typing import Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
import socket, time
import wandb
from dotenv import load_dotenv

from onebit_asr.conformer import ConformerASR
from onebit_asr.losses import make_att_targets, att_ce_loss, ctc_loss_from_logits, kl_logits
from onebit_asr.metrics import compute_wer, ctc_beam_search_batch, ids_to_text
import sentencepiece as spm

# Progress bar (fallback to no-op if tqdm is unavailable)
from tqdm.auto import tqdm as _tqdm

def _progress(x, **kwargs):
    if _tqdm is None:
        return x
    return _tqdm(x, **kwargs)


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        # Ensure each param group has an 'initial_lr' to scale from
        for pg in self.optimizer.param_groups:
            if 'initial_lr' not in pg:
                # Use the current lr of the param group as its initial_lr
                pg['initial_lr'] = pg.get('lr', 1e-3)
    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            scale = self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            scale = self.min_lr_ratio + 0.5*(1-self.min_lr_ratio)*(1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale


def sample_sp_mask(n_layers: int, low_p=0.2, high_p=0.9) -> List[int]:
    # log-linear schedule from 0.2 .. 0.9 across layers 1..n
    probs = torch.logspace(math.log10(low_p), math.log10(high_p), steps=n_layers)
    return [int(torch.rand(()) < p) for p in probs]


def run_epoch(model: ConformerASR, dm, optimizer, sched, device, args, train: bool,
              lambda1: float, lambda2: float, gamma_ctc: float,
              spm_processor: Optional[Any] = None) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    model.train(train)
    dl = dm.train_dataloader() if train else dm.valid_dataloader()

    total = 0.0
    count = 0
    total_dist_student = 0
    total_dist_teacher = 0
    total_dist_fp32 = 0
    total_words = 0

    pbar = _progress(dl, total=len(dl), desc=("Train" if train else "Valid"))
    for batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        special = dm.special_ids()
        bos_id, eos_id, pad_id, blank_id = special['bos_id'], special['eos_id'], special['pad_id'], special['blank_id']

        with torch.set_grad_enabled(train):
            # ---------- Teacher: 2‑bit ----------
            enc2, mask2, ctc2 = model(batch, precision=2)
            t_inp, t_out, t_pad = make_att_targets(batch['tokens'], bos_id, eos_id, pad_id)
            logits2 = model.decode_logits(enc2, mask2, t_inp, t_pad)
            Latt2 = att_ce_loss(logits2, t_out, pad_id, label_smoothing=0.1)
            ctc_lens2 = mask2.sum(dim=1).long()
            Lctc2 = ctc_loss_from_logits(ctc2, ctc_lens2, batch['tokens'], batch['token_lens'], blank_id)
            Lint2 = (1-gamma_ctc)*Latt2 + gamma_ctc*Lctc2

            # ---------- Student: 1‑bit ----------
            enc1, mask1, ctc1 = model(batch, precision=1)
            logits1 = model.decode_logits(enc1, mask1, t_inp, t_pad)
            Latt1 = att_ce_loss(logits1, t_out, pad_id, label_smoothing=0.1)
            ctc_lens1 = mask1.sum(dim=1).long()
            Lctc1 = ctc_loss_from_logits(ctc1, ctc_lens1, batch['tokens'], batch['token_lens'], blank_id)
            Lint1 = (1-gamma_ctc)*Latt1 + gamma_ctc*Lctc1
            # KL from 2->1
            Lkl1 = kl_logits(logits1, logits2.detach(), t_pad)

            # Clear intermediate activations to free memory (keep logits2 for KL loss)
            if train:
                del enc2, mask2, ctc2, enc1, mask1, ctc1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---------- Stochastic Precision sub‑model ----------
            sp_mask = sample_sp_mask(n_layers=args.enc_layers)
            encs, masks, ctcs = model(batch, precision=2, sp_mask=sp_mask)
            logitss = model.decode_logits(encs, masks, t_inp, t_pad)
            Latt_s = att_ce_loss(logitss, t_out, pad_id, label_smoothing=0.1)
            ctc_lens_s = masks.sum(dim=1).long()
            Lctc_s = ctc_loss_from_logits(ctcs, ctc_lens_s, batch['tokens'], batch['token_lens'], blank_id)
            Lint_s = (1-gamma_ctc)*Latt_s + gamma_ctc*Lctc_s
            Lkl_s = kl_logits(logitss, logits2.detach(), t_pad)

            loss = Lint2 + lambda1*(Lint1 + Lint_s) + lambda2*(Lkl1 + Lkl_s)
            #loss = Lint2 + lambda1*(Lint1) + lambda2*(Lkl1)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if sched is not None:
                sched.step()
        else:
            # Compute WER for student (1-bit), teacher (2-bit), and fp32 using existing logits
                        # ---------- Full precision: 32‑bit ----------
            encf, maskf, ctcf = model(batch, precision=32)
            if spm_processor is not None:
                # Student decoding
                valid_t_s = mask1.sum(dim=1).long()
                hyp_ids_s = ctc_beam_search_batch(ctc1, valid_t_s, beam_size=args.beam_size, blank_id=blank_id)
                hyps_s = [ids_to_text(h, spm_processor, token_offset=4) for h in hyp_ids_s]
                # Teacher decoding
                valid_t_t = mask2.sum(dim=1).long()
                hyp_ids_t = ctc_beam_search_batch(ctc2, valid_t_t, beam_size=args.beam_size, blank_id=blank_id)
                hyps_t = [ids_to_text(h, spm_processor, token_offset=4) for h in hyp_ids_t]
                # FP32 decoding
                valid_t_f = maskf.sum(dim=1).long()
                hyp_ids_f = ctc_beam_search_batch(ctcf, valid_t_f, beam_size=args.beam_size, blank_id=blank_id)
                hyps_f = [ids_to_text(h, spm_processor, token_offset=4) for h in hyp_ids_f]
                # References
                labels = batch['tokens'].cpu().tolist()
                refs = []
                for lbl in labels:
                    valid_lbl = [int(x - 4) for x in lbl if x != 0]
                    refs.append(spm_processor.decode(valid_lbl))
                d_s, w = compute_wer(refs, hyps_s)
                d_t, _ = compute_wer(refs, hyps_t)
                d_f, _ = compute_wer(refs, hyps_f)
                total_dist_student += d_s
                total_dist_teacher += d_t
                total_dist_fp32 += d_f
                total_words += w

        total += loss.item()
        count += 1
        # Show current (last) loss (and WER on eval) on the progress bar
        if train:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        else:
            wer_s = (total_dist_student / total_words) if total_words > 0 else 0.0
            wer_t = (total_dist_teacher / total_words) if total_words > 0 else 0.0
            wer_f = (total_dist_fp32 / total_words) if total_words > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", wer_s=f"{wer_s:.4f}", wer_t=f"{wer_t:.4f}", wer_fp32=f"{wer_f:.4f}")
    avg_loss = total / max(1, count)
    if train:
        return avg_loss, None, None, None
    else:
        wer_s = (total_dist_student / total_words) if total_words > 0 else 0.0
        wer_t = (total_dist_teacher / total_words) if total_words > 0 else 0.0
        wer_f = (total_dist_fp32 / total_words) if total_words > 0 else 0.0
        return avg_loss, wer_s, wer_t, wer_f


def main():
    # ===== DEBUGPY SETUP FOR REMOTE DEBUGGING =====
    # Uncomment the lines below to enable remote debugging
    # Then connect VSCode debugger to port 5678
    # SSH port forwarding: ssh -L 5678:localhost:5678 catalyst-0-9
    
    # import debugpy
    # print("🐛 Waiting for debugger to attach on port 5678...")
    # debugpy.listen(("0.0.0.0", 5678))
    # debugpy.wait_for_client()
    # print("✓ Debugger attached! Starting execution...")
    # debugpy.breakpoint()  # This will pause execution here
    # =============================================
    
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default="nothing")
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4) # Reduces lr
    p.add_argument('--warmup_steps', type=int, default=4000)
    p.add_argument('--input_dim', type=int, default=80)  # fbank dims
    p.add_argument('--enc_d_model', type=int, default=256)
    p.add_argument('--enc_layers', type=int, default=12)
    p.add_argument('--enc_heads', type=int, default=4)
    p.add_argument('--enc_d_ff', type=int, default=1024)
    p.add_argument('--enc_conv_kernel', type=int, default=31)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--dec_layers', type=int, default=2)
    p.add_argument('--dec_heads', type=int, default=4)
    p.add_argument('--dec_d_ff', type=int, default=1024)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--beam_size', type=int, default=10)
    # paper constants
    p.add_argument('--gamma_ctc', type=float, default=0.2)   # Eq. (1)
    p.add_argument('--lambda1', type=float, default=0.5)     # weight for 1-bit & SP losses
    p.add_argument('--lambda2', type=float, default=1.0)     # weight for KL terms
    p.add_argument('--train_data_fraction', type=float, default=0.5, help="Fraction of training data to use (0.0-1.0, default: 0.1 for 10%%)")
    p.add_argument('--resume', action='store_true', help="resume wandb run")
    p.add_argument('--use_checkpoint', action='store_true', default=True, help="Use gradient checkpointing to save memory")
    p.add_argument('--no_checkpoint', dest='use_checkpoint', action='store_false', help="Disable gradient checkpointing")
    args = p.parse_args()

    #----WandB Initialization----
    # Load environment variables from .env file
    load_dotenv()
    
    # Login to wandb with API key from .env file or environment variable
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Warning: WANDB_API_KEY not found in .env file or environment. WandB may not work properly.")
        print("Create a .env file with: WANDB_API_KEY='your-api-key'")
        print("Or set it with: export WANDB_API_KEY='your-api-key'")
    
    # Create descriptive run name based on training data fraction
    data_pct = int(args.train_data_fraction * 100)
    run_id = f"{socket.gethostname()}-{int(time.time())}"
    run_name = f"run-{data_pct}pct-{run_id}" if args.train_data_fraction < 1.0 else f"run-{run_id}"
    
    wandb.init(
        project="ASR-1bit",
        name=run_name,
        group=f"baseline-conformer-{data_pct}pct" if args.train_data_fraction < 1.0 else "baseline-conformer",
        config=vars(args),
        tags=["baseline", "1bit", "cosine", "adamw", f"{data_pct}pct"] if args.train_data_fraction < 1.0 else ["baseline", "1bit", "cosine", "adamw"],
        resume="allow" if args.resume else None,
        # notes=f"baseline training for 1bit with {data_pct}% of training data",
    )


    # --- You: implement this class in your project per the interface in README.
    from onebit_asr.dataloader_stub import LibriSpeechDataModule  # replace with your actual module
    dm = LibriSpeechDataModule(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Create subset of training data if train_data_fraction < 1.0
    if args.train_data_fraction < 1.0:
        train_dl = dm.train_dataloader()
        # Get the base dataloader to access the dataset
        base_dataloader = train_dl._base
        base_dataset = base_dataloader.dataset
        total_samples = len(base_dataset)
        subset_size = int(total_samples * args.train_data_fraction)
        
        # Create random indices for subset (use fixed seed for reproducibility, or random for variety)
        generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        indices = torch.randperm(total_samples, generator=generator)[:subset_size].tolist()
        subset_dataset = Subset(base_dataset, indices)
        
        # Create new dataloader with subset, using same collate function
        from torch.utils.data import DataLoader
        subset_dl = DataLoader(
            subset_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=base_dataloader.collate_fn,
            pin_memory=base_dataloader.pin_memory if hasattr(base_dataloader, 'pin_memory') else True,
        )
        
        # Wrap in _MappedLoader to maintain interface
        from onebit_asr.dataloader_stub import _MappedLoader
        token_offset = train_dl._token_offset
        dm._train_dl = _MappedLoader(subset_dl, token_offset=token_offset)
        
        print(f"Using {subset_size}/{total_samples} training samples ({args.train_data_fraction*100:.1f}%)")

    vocab = dm.vocab_size()
    special = dm.special_ids()

    device = torch.device(args.device)
    model = ConformerASR(
        input_dim=args.input_dim,
        vocab_size=vocab,
        enc_d_model=args.enc_d_model,
        enc_layers=args.enc_layers,
        enc_heads=args.enc_heads,
        enc_d_ff=args.enc_d_ff,
        enc_conv_kernel=args.enc_conv_kernel,
        enc_dropout=args.dropout,
        dec_layers=args.dec_layers,
        dec_heads=args.dec_heads,
        dec_d_ff=args.dec_d_ff,
        dec_dropout=args.dropout,
        pad_id=special['pad_id'],
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-2)

    # Rough step estimate for scheduler
    steps_per_epoch = max(1, math.ceil( len(dm.train_dataloader()) ))
    total_steps = args.epochs * steps_per_epoch
    sched = WarmupCosine(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    # Checkpoint directory per run
    run_name = (wandb.run.name or f"run-{run_id}") if wandb.run is not None else f"run-{run_id}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config for later evaluation
    config_to_save = dict(vars(args))
    config_to_save.update({
        'run_name': run_name,
        'run_dir': run_dir,
        'vocab_size': vocab,
        'special_ids': special,
    })
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Expose run_dir in wandb
    wandb.config.update({'run_dir': run_dir}, allow_val_change=True)
    best_val = float('inf')

    # Prepare tokenizer for WER decoding
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join('src', 'data', 'tokenizer.model'))  # type: ignore[attr-defined]

    for epoch in range(1, args.epochs+1):
        tr, _, _, _ = run_epoch(model, dm, optimizer, sched, device, args, train=True,
                       lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc)
        va, val_wer_student, val_wer_teacher, val_wer_fp32 = run_epoch(model, dm, optimizer, sched=None, device=device, args=args, train=False,
                                lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc,
                                spm_processor=sp)
        print(f"Epoch {epoch}: train_loss={tr:.4f}  valid_loss={va:.4f}")
        print(f"           valid_wer_student={val_wer_student:.4f}  valid_wer_teacher={val_wer_teacher:.4f}  valid_wer_fp32={val_wer_fp32:.4f}")
        
        wandb.log({
            'epoch': epoch,
            'train_loss': tr,
            'valid_loss': va,
            'valid_wer_1bit': val_wer_student,
            'valid_wer_2bit': val_wer_teacher,
            'valid_wer_32bit': val_wer_fp32,
        })
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'val_loss': va,
        }
        torch.save(ckpt, os.path.join(run_dir, f'ckpt_last.pt'))
        if va < best_val:
            best_val = va
            torch.save(ckpt, os.path.join(run_dir, 'best.pt'))
            print(f"  ✓ New best model saved with valid_loss={best_val:.4f}")


if __name__ == '__main__':
    main()
