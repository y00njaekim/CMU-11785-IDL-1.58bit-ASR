
# --------------------------------------------------------------
# File: train.py
# --------------------------------------------------------------
import os
# Force SentencePiece to use a single thread per process.
# This fixes SegFaults (Exit -11) when running tokenizers in DDP.
os.environ["SPM_NUM_THREADS"] = "1"
import sys
import math
import argparse
import json
import signal
from typing import Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset, DistributedSampler
import torch.distributed as dist
import socket, time
import wandb
from dotenv import load_dotenv

from onebit_asr.conformer_fp import ConformerASR
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


def run_epoch(model: Union[ConformerASR, nn.parallel.DistributedDataParallel], dm, optimizer, sched, device, args, train: bool,
              lambda1: float, lambda2: float, gamma_ctc: float,
              spm_processor: Optional[Any] = None, rank: int = 0) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    model.train(train)
    dl = dm.train_dataloader() if train else dm.valid_dataloader()
    
    if train and dist.is_initialized():
        sampler = dl._base.sampler if hasattr(dl, '_base') and hasattr(dl._base, 'sampler') else None
        if sampler is not None:
            sampler.set_epoch(getattr(run_epoch, '_epoch', 0))

    total = 0.0
    count = 0
    total_dist_fp32 = 0
    total_words = 0

    pbar = _progress(dl, total=len(dl), desc=("Train" if train else "Valid"), disable=(rank != 0))
    for batch in pbar:
        if _shutdown_flag:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        special = dm.special_ids()
        bos_id, eos_id, pad_id, blank_id = special['bos_id'], special['eos_id'], special['pad_id'], special['blank_id']

        with torch.set_grad_enabled(train):
            t_inp, t_out, t_pad = make_att_targets(batch['tokens'], bos_id, eos_id, pad_id)
            
            # ---------- FP32 training ----------
            # We pass targets into the forward pass so decoding happens inside DDP context
            enc, mask, ctc, logits = model(
                batch, 
                precision=32, 
                tgt_inp=t_inp, 
                tgt_pad_mask=t_pad
            )
            
            # Ensure logits is not None (should always be provided during training)
            if logits is None:
                raise RuntimeError("Decoder logits are None - check model forward implementation")
            
            Latt = att_ce_loss(logits, t_out, pad_id, label_smoothing=0.1)
            ctc_lens = mask.sum(dim=1).long()
            Lctc = ctc_loss_from_logits(ctc, ctc_lens, batch['tokens'], batch['token_lens'], blank_id)
            loss = (1-gamma_ctc)*Latt + gamma_ctc*Lctc
            
            if train:
                # Memory cleanup to avoid OOM
                del enc, mask, ctc, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if sched is not None:
                sched.step()
        else:
            # ---------- Validation / Metrics ----------
            if spm_processor is not None:
                # FP32 decoding (Beam Search)
                # Note: enc, mask, ctc, logits are available from the forward pass above
                valid_t = mask.sum(dim=1).long()
                hyp_ids = ctc_beam_search_batch(ctc, valid_t, beam_size=args.beam_size, blank_id=blank_id)
                # Note: token_offset=4 usually handles special tokens in your vocab
                hyps = [ids_to_text(h, spm_processor, token_offset=4) for h in hyp_ids]
                
                # References
                labels = batch['tokens'].cpu().tolist()
                refs = []
                for lbl in labels:
                    # Filter out padding (0) and adjust for offset if needed
                    valid_lbl = [int(x - 4) for x in lbl if x != 0]
                    refs.append(spm_processor.decode(valid_lbl))
                
                d, w = compute_wer(refs, hyps)
                total_dist_fp32 += d
                total_words += w

        total += loss.item()
        count += 1
        
        # Update Progress Bar
        if train:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        else:
            wer_f = (total_dist_fp32 / total_words) if total_words > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", wer=f"{wer_f:.4f}")
            
    avg_loss = total / max(1, count)
    
    # Calculate Final WER for the epoch
    if not train:
        wer_fp32 = (total_dist_fp32 / total_words) if total_words > 0 else 0.0
    else:
        wer_fp32 = None

    # Return matching the expected signature: (loss, student_wer, teacher_wer, fp32_wer)
    # We return 0.0 or None for student/teacher to keep main() compatible
    if train:
        return avg_loss, None, None, None
    else:
        return avg_loss, 0.0, 0.0, wer_fp32
        


_shutdown_flag = False

def signal_handler(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True
    if dist.is_initialized():
        cleanup()

def setup():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Set CUDA device BEFORE initializing process group to avoid NCCL warnings
        # This is critical - the device must be set before NCCL initialization
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group - device is already set above
        dist.init_process_group(
            backend='nccl', 
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return 0, 1, device

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main(args=None):
    if args is None:
        p = argparse.ArgumentParser()
        p.add_argument('--data_dir', type=str, default="nothing")
        p.add_argument('--save_dir', type=str, default='./checkpoints')
        p.add_argument('--epochs', type=int, default=10)
        p.add_argument('--batch_size', type=int, default=2)
        p.add_argument('--num_workers', type=int, default=1, help="Number of DataLoader workers (reduce if you see warnings about too many workers)")
        p.add_argument('--lr', type=float, default=5e-4)
        p.add_argument('--warmup_steps', type=int, default=4000)
        p.add_argument('--input_dim', type=int, default=80)
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
        p.add_argument('--ddp', action='store_true', help='Use Distributed Data Parallel (use torchrun instead)')
        p.add_argument('--gamma_ctc', type=float, default=0.2)
        p.add_argument('--lambda1', type=float, default=0.5)
        p.add_argument('--lambda2', type=float, default=1.0)
        p.add_argument('--train_data_fraction', type=float, default=0.5, help="Fraction of training data to use (0.0-1.0, default: 0.5 for 50%%)")
        p.add_argument('--valid_data_fraction', type=float, default=1.0, help="Fraction of validation data to use (0.0-1.0, default: 1.0 for 100%%)")
        p.add_argument('--resume', action='store_true', help="resume wandb run")
        p.add_argument('--use_checkpoint', action='store_true', default=True, help="Use gradient checkpointing to save memory")
        p.add_argument('--no_checkpoint', dest='use_checkpoint', action='store_false', help="Disable gradient checkpointing")
        args = p.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rank, world_size, device = setup()
    if device is None:
        device = torch.device(args.device)
    
    use_ddp = dist.is_initialized() and world_size > 1
    
    if args.ddp and not use_ddp:
        print("Warning: --ddp flag set but not running with torchrun. Use: torchrun --nproc_per_node=<num_gpus> -m onebit_asr.train ...")

    if rank == 0:
        load_dotenv()
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("Warning: WANDB_API_KEY not found in .env file or environment. WandB may not work properly.")
            print("Create a .env file with: WANDB_API_KEY='your-api-key'")
            print("Or set it with: export WANDB_API_KEY='your-api-key'")
        
        train_pct = int(args.train_data_fraction * 100)
        valid_pct = int(args.valid_data_fraction * 100)
        run_id = f"{socket.gethostname()}-{int(time.time())}"
        run_name = f"1.58bit-ASR-full-precision-{run_id}"
        
        wandb.init(
            project="1.58 bit ASR training",
            name=run_name,
            group=f"full-precision-train{train_pct}pct-valid{valid_pct}pct",
            config=vars(args),
            tags=["full-precision", "fp32", "cosine", "adamw", f"train{train_pct}pct", f"valid{valid_pct}pct"],
            resume="allow" if args.resume else None,
        )


    from onebit_asr.dataloader_stub import LibriSpeechDataModule
    
    if rank != 0:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
    
    # For DDP, use 0 workers to avoid multiprocessing conflicts
    # DDP already uses multiple processes, so additional workers can cause segfaults
    dataloader_num_workers = 0 if use_ddp else args.num_workers
    
    dm = LibriSpeechDataModule(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=dataloader_num_workers,
        train_fraction=args.train_data_fraction
    )
    
    if rank != 0:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    train_dl = dm.train_dataloader()
    valid_dl = dm.valid_dataloader()
    
    # Note: Dataset slicing is now handled in get_dataloaders() to avoid loading full dataset
    # DDP sampler handling - LengthAwareBatchSampler uses batch_sampler which conflicts with DDP's sampler
    # So we need to replace it with a regular DataLoader + DistributedSampler for DDP
    if use_ddp:
        if hasattr(train_dl, '_base'):
            try:
                # Get the underlying dataset (might be a Subset)
                base_dataset = train_dl._base.dataset
                # DistributedSampler works with Subset, but we need to be careful
                # Set num_workers to 0 for DDP to avoid multiprocessing conflicts that can cause segfaults
                # DDP already uses multiple processes, so additional workers can cause issues
                ddp_num_workers = 0
                train_sampler = DistributedSampler(
                    base_dataset, 
                    num_replicas=world_size, 
                    rank=rank, 
                    shuffle=True,
                    drop_last=False
                )
                from torch.utils.data import DataLoader
                train_dl_new = DataLoader(
                    base_dataset,
                    batch_size=args.batch_size,
                    sampler=train_sampler,  # Use sampler instead of batch_sampler for DDP
                    num_workers=ddp_num_workers,
                    collate_fn=train_dl._base.collate_fn,
                    pin_memory=False,  # Disable pin_memory to avoid SIGSEGV with SentencePiece/C++ extensions
                )
                from onebit_asr.dataloader_stub import _MappedLoader
                train_dl = _MappedLoader(train_dl_new, token_offset=train_dl._token_offset)
                # Update the datamodule so subsequent calls return the DDP-compatible loader
                dm._train_dl = train_dl
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to setup DDP sampler, falling back to original: {e}")
                # Fall back to original loader
    
    if args.valid_data_fraction < 1.0:
        base_dataloader = valid_dl._base
        base_dataset = base_dataloader.dataset
        total_samples = len(base_dataset)
        subset_size = int(total_samples * args.valid_data_fraction)
        
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_samples, generator=generator)[:subset_size].tolist()
        subset_dataset = Subset(base_dataset, indices)
        
        from torch.utils.data import DataLoader
        if use_ddp:
            subset_sampler = DistributedSampler(subset_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            subset_dl = DataLoader(
                subset_dataset,
                batch_size=args.batch_size,
                sampler=subset_sampler,
                num_workers=0,  # Use 0 workers for DDP to avoid multiprocessing conflicts
                collate_fn=base_dataloader.collate_fn,
                pin_memory=False,  # Disable pin_memory to avoid SIGSEGV with SentencePiece/C++ extensions
            )
        else:
            subset_dl = DataLoader(
                subset_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=base_dataloader.collate_fn,
                pin_memory=base_dataloader.pin_memory if hasattr(base_dataloader, 'pin_memory') else True,
            )
        
        from onebit_asr.dataloader_stub import _MappedLoader
        token_offset = valid_dl._token_offset
        valid_dl = _MappedLoader(subset_dl, token_offset=token_offset)
        
        if rank == 0:
            print(f"Using {subset_size}/{total_samples} validation samples ({args.valid_data_fraction*100:.1f}%)")
    elif use_ddp:
        if hasattr(valid_dl, '_base'):
            valid_sampler = DistributedSampler(valid_dl._base.dataset, num_replicas=world_size, rank=rank, shuffle=False)
            from torch.utils.data import DataLoader
            valid_dl_new = DataLoader(
                valid_dl._base.dataset,
                batch_size=args.batch_size,
                sampler=valid_sampler,
                num_workers=0,  # Use 0 workers for DDP to avoid multiprocessing conflicts
                collate_fn=valid_dl._base.collate_fn,
                pin_memory=False,  # Disable pin_memory to avoid SIGSEGV with SentencePiece/C++ extensions
            )
            from onebit_asr.dataloader_stub import _MappedLoader
            valid_dl = _MappedLoader(valid_dl_new, token_offset=valid_dl._token_offset)
    
    dm._train_dl = train_dl
    dm._valid_dl = valid_dl

    vocab = dm.vocab_size()
    special = dm.special_ids()

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
    
    # Wrap model in DDP - DDP constructor handles synchronization internally
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-2)

    # Get dataset length safely - use the train_dl we already have
    try:
        if hasattr(train_dl, "_base") and hasattr(train_dl._base, "dataset"):
            base_ds = train_dl._base.dataset
            dataset_len = len(base_ds)
        else:
            dataset_len = len(train_dl)
    except Exception as e:
        if rank == 0:
            print(f"Warning: Could not get dataset length: {e}")
        dataset_len = len(train_dl) if hasattr(train_dl, "__len__") else 1000  # fallback
    
    if rank == 0:
        print(f"Final train dataset size used = {dataset_len} samples")
        print(f"Batch size per-process = {args.batch_size}; world_size = {world_size}")
    
    steps_per_epoch = max(1, math.ceil(dataset_len / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    sched = WarmupCosine(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    if rank == 0:
        run_id = f"{socket.gethostname()}-{int(time.time())}"
        train_pct = int(args.train_data_fraction * 100)
        valid_pct = int(args.valid_data_fraction * 100)
        run_name = (wandb.run.name or f"full-precision-train{train_pct}pct-valid{valid_pct}pct-{run_id}") if wandb.run is not None else f"full-precision-train{train_pct}pct-valid{valid_pct}pct-{run_id}"
        run_dir = os.path.join(args.save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        config_to_save = dict(vars(args))
        config_to_save.update({
            'run_name': run_name,
            'run_dir': run_dir,
            'vocab_size': vocab,
            'special_ids': special,
        })
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        if wandb.run is not None:
            wandb.config.update({'run_dir': run_dir}, allow_val_change=True)
    else:
        run_dir = os.path.join(args.save_dir, 'temp')
        os.makedirs(run_dir, exist_ok=True)
    
    best_val = float('inf')

    sp = None
    if rank == 0:
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join('src', 'data', 'tokenizer.model'))
    
    if use_ddp:
        dist.barrier()  # Synchronize after SentencePiece loading

    try:
        for epoch in range(1, args.epochs+1):
            if _shutdown_flag:
                if rank == 0:
                    print("\nShutdown signal received, saving checkpoint and exiting...")
                break
            
            if use_ddp:
                run_epoch._epoch = epoch
            
            tr, _, _, _ = run_epoch(model, dm, optimizer, sched, device, args, train=True,
                           lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc, rank=rank)
            
            if _shutdown_flag:
                if rank == 0:
                    print("\nShutdown signal received, saving checkpoint and exiting...")
                break
            
            if use_ddp:
                dist.barrier()  # Synchronize after training epoch
            
            va, val_wer_student, val_wer_teacher, val_wer_fp32 = run_epoch(model, dm, optimizer, sched=None, device=device, args=args, train=False,
                                    lambda1=args.lambda1, lambda2=args.lambda2, gamma_ctc=args.gamma_ctc,
                                    spm_processor=sp, rank=rank)
            
            if rank == 0:
                print(f"Epoch {epoch}: train_loss={tr:.4f}  valid_loss={va:.4f}")
                print(f"           valid_wer_student={val_wer_student:.4f}  valid_wer_teacher={val_wer_teacher:.4f}  valid_wer_fp32={val_wer_fp32:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': tr,
                        'valid_loss': va,
                        'valid_wer_1bit': val_wer_student,
                        'valid_wer_2bit': val_wer_teacher,
                        'valid_wer_32bit': val_wer_fp32,
                    })
                
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                ckpt = {
                    'epoch': epoch,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                    'val_loss': va,
                }
                torch.save(ckpt, os.path.join(run_dir, f'ckpt_last.pt'))
                if va < best_val:
                    best_val = va
                    torch.save(ckpt, os.path.join(run_dir, 'best.pt'))
                    print(f"  ✓ New best model saved with valid_loss={best_val:.4f}")
    except KeyboardInterrupt:
        if rank == 0:
            print("\nKeyboard interrupt received, saving checkpoint and exiting...")
    finally:
        # Synchronize all processes before cleanup
        if use_ddp:
            try:
                dist.barrier()  # Synchronize before cleanup
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Barrier failed during cleanup: {e}")
        
        if rank == 0 and not _shutdown_flag:
            try:
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                ckpt = {
                    'epoch': epoch if 'epoch' in locals() else 0,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                    'val_loss': va if 'va' in locals() else best_val,
                }
                torch.save(ckpt, os.path.join(run_dir, f'ckpt_last.pt'))
                print("Final checkpoint saved.")
            except Exception as e:
                print(f"Error saving final checkpoint: {e}")
        
        # Cleanup with error handling
        try:
            cleanup()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Cleanup error (non-fatal): {e}")


if __name__ == '__main__':
    main()
