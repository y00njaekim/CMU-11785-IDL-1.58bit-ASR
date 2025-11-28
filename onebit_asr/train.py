
# --------------------------------------------------------------
# File: train.py
# --------------------------------------------------------------
import os
import sys
import math
import argparse
from typing import Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import socket, time
import wandb

from onebit_asr.conformer import ConformerASR
from onebit_asr.losses import make_att_targets, att_ce_loss, ctc_loss_from_logits, kl_logits

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
              lambda1: float, lambda2: float, gamma_ctc: float):
    model.train(train)
    dl = dm.train_dataloader() if train else dm.valid_dataloader()

    total = 0.0
    count = 0

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

        total += loss.item()
        count += 1
        # Show current (last) loss on the progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / max(1, count)


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
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=4)
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
    # paper constants
    p.add_argument('--gamma_ctc', type=float, default=0.2)   # Eq. (1)
    p.add_argument('--lambda1', type=float, default=0.5)     # weight for 1-bit & SP losses
    p.add_argument('--lambda2', type=float, default=1.0)     # weight for KL terms
    p.add_argument('--resume', action='store_true', help="resume wandb run")
    args = p.parse_args()

    #----WandB Initialization----
    api_key_file = "wandb_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"Error: WandB API key file '{api_key_file}' not found.")
        sys.exit(1)
    
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()
    wandb.login(key=api_key)

    run_id = f"{socket.gethostname()}-{int(time.time())}"
    wandb.init(
        project="ASR-1bit",
        name=f"run-{run_id}",
        group="baseline-conformer", # runs in the same experiment family are grouped for aligned comparison
        config=vars(args),
        tags=["baseline", "1bit", "cosine", "adamw"],   # used in WandB for filtering and comparing different experiments
        resume="allow" if args.resume else None,
        # notes="baseline training for 1bit",
    )


    # --- You: implement this class in your project per the interface in README.
    from onebit_asr.dataloader_stub import LibriSpeechDataModule  # replace with your actual module
    dm = LibriSpeechDataModule(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

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
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-2)

    # Rough step estimate for scheduler
    steps_per_epoch = max(1, math.ceil( len(dm.train_dataloader()) ))
    total_steps = args.epochs * steps_per_epoch
    sched = WarmupCosine(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        tr = run_epoch(model, dm, optimizer, sched, device, args, train=True,
                       lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc)
        va = run_epoch(model, dm, optimizer, sched=None, device=device, args=args, train=False,
                       lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc)
        print(f"Epoch {epoch}: train_loss={tr:.4f}  valid_loss={va:.4f}")
        
        wandb.log({
            'epoch': epoch,
            'train_loss': tr,
            'valid_loss': va,
        })
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'val_loss': va,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_last.pt'))
        if va < best_val:
            best_val = va
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ New best model saved with valid_loss={best_val:.4f}")


if __name__ == '__main__':
    main()
