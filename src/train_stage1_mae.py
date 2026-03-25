# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE_DA3 MAE decoder training script.

Trains a GeneralDecoder_Variable on multi-level DA3 features with
reconstruction, LPIPS, and GAN losses. The DA3 encoder is frozen;
only the MAE decoder is trained.

Normalization flow:
  - cut3r_data returns ImageNet-normalized images: (img - mean) / std
  - RAE_DA3.encode(mode='all') takes ImageNet-normalized 5D input
    and returns raw backbone features: Dict[level -> (B*V, N, 1536)]
  - Decoder predicts pixel patches in ImageNet-normalized space
  - Denormalize: pred * std + mean -> [0,1] for loss computation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Fix xformers compatibility with older PyTorch
import torch.backends.cuda
if not hasattr(torch.backends.cuda, 'is_flash_attention_available'):
    torch.backends.cuda.is_flash_attention_available = lambda: False

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoConfig
from math import sqrt
import wandb

from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE_DA3
from stage1.decoders import GeneralDecoder_Variable
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from cut3r_data import get_data_loader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 MAE decoder on RAE_DA3 features.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--data-path", type=Path, required=True, help="Base data directory.")
    parser.add_argument("--results-dir", type=str, default="results/stage1-mae", help="Output directory.")
    parser.add_argument("--image-size", type=int, default=504, help="Image resolution.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()]
            + ([logging.FileHandler(f"{logging_dir}/log.txt")] if logging_dir else []),
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_vis_multiview(gt, recon):
    """gt, recon: (V, C, H, W) → numpy (H*V, W*2, C)"""
    rows = []
    for i in range(min(gt.shape[0], 2)):
        gt_i = gt[i].permute(1, 2, 0)
        rc_i = recon[i].permute(1, 2, 0)
        rows.append(torch.cat([gt_i, rc_i], dim=1))
    grid = torch.cat(rows, dim=0)
    return grid.cpu().numpy()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(rae, decoder, val_loader, device, lpips_fn=None):
    rae.eval()
    decoder.eval()
    val_rec = val_lpips = 0.0
    n_batches = 0
    vis_images = []

    encoder_mean = rae.encoder_mean  # (1, 3, 1, 1)
    encoder_std = rae.encoder_std

    val_loader.dataset.set_epoch(0)
    for i, image_dict in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        # image_dict is a list of dicts (one per view)
        images_enc = torch.stack([d["img"] for d in image_dict], dim=1).to(device, non_blocking=True)
        b, f, c, h, w = images_enc.shape

        # Ground truth in [0,1]
        images_01 = images_enc * encoder_std[None] + encoder_mean[None]
        images_01_flat = images_01.view(b * f, c, h, w)
        real_normed = images_01_flat * 2.0 - 1.0  # [-1, 1] for LPIPS

        # Encode all levels
        all_feats = rae.encode(images_enc, mode="all")  # Dict[level -> (B*V, N+1, 1536)]
        # Strip CLS (first token) from each level, then concat
        z = torch.cat([all_feats[k][:, 1:, :] for k in sorted(all_feats.keys())], dim=-1)

        # Decode
        output = decoder(z, input_size=(h, w), drop_cls_token=False).logits
        x_rec = decoder.unpatchify(output, (h, w))  # (B*V, 3, H, W) in ImageNet-norm space
        # Denormalize to [0,1]
        x_rec = x_rec * encoder_std.squeeze(0) + encoder_mean.squeeze(0)

        rec_loss = F.l1_loss(x_rec, images_01_flat)

        if lpips_fn is not None:
            recon_normed = x_rec * 2.0 - 1.0
            lpips_val = lpips_fn(real_normed, recon_normed).mean()
        else:
            lpips_val = torch.tensor(0.0, device=device)

        val_rec += rec_loss.item()
        val_lpips += lpips_val.item()
        n_batches += 1

        # Visualise first few
        images_5d = images_01.view(b, f, c, h, w)
        recon_5d = x_rec.view(b, f, c, h, w).clamp(0, 1)
        for b_ in range(b):
            vis_images.append(
                wandb.Image(make_vis_multiview(images_5d[b_], recon_5d[b_]), caption=f"Sample {b_ + i * b}")
            )

    return {
        "val/recon": val_rec / max(n_batches, 1),
        "val/lpips": val_lpips / max(n_batches, 1),
        "val/images": vis_images,
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    d_fn = {"hinge": hinge_d_loss, "vanilla": vanilla_d_loss}[disc_kind]
    g_fn = {"vanilla": vanilla_g_loss}[gen_kind]
    return d_fn, g_fn


def prepare_dataloader(dataset, batch_size, workers, rank, world_size, test=False):
    return get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test,
        fixed_length=True,
        world_size=world_size,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(path, step, epoch, decoder, ema_decoder, optimizer, scheduler,
                    disc, disc_optimizer, disc_scheduler):
    state = {
        "step": step,
        "epoch": epoch,
        "decoder": decoder.module.state_dict() if isinstance(decoder, DDP) else decoder.state_dict(),
        "ema_decoder": ema_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.module.state_dict() if isinstance(disc, DDP) else disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, decoder_ddp, ema_decoder, optimizer, scheduler,
                    disc_ddp, disc_optimizer, disc_scheduler):
    ckpt = torch.load(path, map_location="cpu")
    decoder_ddp.module.load_state_dict(ckpt["decoder"])
    ema_decoder.load_state_dict(ckpt["ema_decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "disc" in ckpt:
        disc_mod = disc_ddp.module if isinstance(disc_ddp, DDP) else disc_ddp
        disc_mod.load_state_dict(ckpt["disc"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        if disc_scheduler is not None and ckpt.get("disc_scheduler") is not None:
            disc_scheduler.load_state_dict(ckpt["disc_scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    # ---- Load config ----
    full_cfg = OmegaConf.load(args.config)
    rae_config = full_cfg.get("stage_1")
    training_cfg = OmegaConf.to_container(full_cfg.get("training", {}), resolve=True) or {}
    training_cfg = dict(training_cfg)
    gan_cfg = OmegaConf.to_container(full_cfg.get("gan", {}), resolve=True) or {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section.")
    disc_cfg = gan_cfg.get("disc", {})
    loss_cfg = gan_cfg.get("loss", {})

    # ---- Loss weights ----
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    recon_weight = float(loss_cfg.get("recon_weight", 1.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    # ---- Training hypers ----
    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val and float(clip_grad_val) > 0 else None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 5000))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 100))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- Decoder config ----
    decoder_cfg = OmegaConf.to_container(full_cfg.get("decoder", {}), resolve=True) or {}
    decoder_config_path = decoder_cfg.get("config_path", "configs/decoder/ViTXL")
    decoder_patch_size = int(decoder_cfg.get("patch_size", 14))
    dropout_prob = float(decoder_cfg.get("dropout", 0.0))
    # hidden_size = 1536 * 4 for 4 concatenated DA3 levels
    decoder_hidden_size = int(decoder_cfg.get("hidden_size", 1536 * 4))

    # ---- Experiment directory ----
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        experiment_name = f"{experiment_index:03d}-RAE_DA3_MAE{precision_suffix}"
        experiment_dir = os.path.join(args.results_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        if args.wandb:
            entity = os.environ.get("ENTITY", "gld")
            project = os.environ.get("PROJECT", "RAE_stage1")
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = checkpoint_dir = None
        logger = create_logger(None)

    # ---- Build RAE_DA3 (encoder only, frozen) ----
    rae = instantiate_from_config(rae_config).to(device)
    rae.encoder.eval()
    rae.encoder.requires_grad_(False)
    # Freeze everything in RAE — we only train the standalone decoder
    for p in rae.parameters():
        p.requires_grad_(False)
    rae.eval()

    # ---- Build standalone MAE decoder (trainable) ----
    dec_config = AutoConfig.from_pretrained(decoder_config_path)
    dec_config.hidden_size = decoder_hidden_size
    dec_config.patch_size = decoder_patch_size
    decoder = GeneralDecoder_Variable(dec_config, base_image_size=(504, 504)).to(device)
    decoder.train()

    ema_decoder = deepcopy(decoder).to(device).eval()
    ema_decoder.requires_grad_(False)

    ddp_decoder = DDP(decoder, device_ids=[device.index], broadcast_buffers=False)
    optimizer, optim_msg = build_optimizer([p for p in decoder.parameters() if p.requires_grad], training_cfg)

    # ---- Discriminator ----
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    ddp_disc.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)
    disc_scheduler: LambdaLR | None = None

    # ---- LPIPS ----
    lpips = LPIPS().to(device)
    lpips.eval()

    # ---- Mixed precision ----
    scaler: GradScaler | None
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)

    # ---- Data ----
    loader = prepare_dataloader(full_cfg.train_dataset, batch_size, num_workers, rank, world_size)
    val_loader = prepare_dataloader(full_cfg.test_dataset, batch_size, num_workers, rank, world_size, test=True)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches.")

    # ---- Schedulers ----
    scheduler: LambdaLR | None = None
    if training_cfg.get("scheduler"):
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler"):
        disc_scheduler, _ = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, global_step = load_checkpoint(
            ckpt_path, ddp_decoder, ema_decoder, optimizer, scheduler,
            ddp_disc, disc_optimizer, disc_scheduler,
        )
        logger.info(f"Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")

    # ---- Normalization buffers ----
    encoder_mean = rae.encoder_mean  # (1, 3, 1, 1) on device
    encoder_std = rae.encoder_std

    if rank == 0:
        n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        n_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"MAE decoder trainable params: {n_dec / 1e6:.2f}M")
        logger.info(f"Discriminator trainable params: {n_disc / 1e6:.2f}M")
        logger.info(f"GAN: disc_loss={disc_loss_type}, gen_loss={gen_loss_type}")
        logger.info(f"Weights: recon={recon_weight}, lpips={perceptual_weight}, disc={disc_weight}")
        logger.info(f"GAN starts epoch {gan_start_epoch}, disc updates epoch {disc_update_epoch}, LPIPS epoch {lpips_start_epoch}")
        logger.info(f"Training {num_epochs} epochs, bs={batch_size}/GPU, {steps_per_epoch} steps/epoch, world_size={world_size}")

    last_layer = decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch

    # =====================================================================
    # Training loop
    # =====================================================================
    for epoch in range(start_epoch, num_epochs):
        ddp_decoder.train()
        loader.dataset.set_epoch(epoch)
        loader.batch_sampler.set_epoch(epoch)

        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0

        pbar = tqdm(loader, total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs}") if rank == 0 else loader

        for step, image_dict in enumerate(pbar):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0

            # ---- Prepare data ----
            # image_dict: list of dicts, one per view
            images_enc = torch.cat([d["img"].unsqueeze(1) for d in image_dict], dim=1)
            images_enc = images_enc.to(device, non_blocking=True)  # (B, V, C, H, W) ImageNet-normed
            b, f, c, h, w = images_enc.shape

            # Ground truth [0,1]
            images_01 = (images_enc * encoder_std[None] + encoder_mean[None]).view(b * f, c, h, w).contiguous()
            real_normed = (images_01 * 2.0 - 1.0).contiguous()  # [-1, 1]

            # ---- Forward: encode (frozen) → decode (trainable) ----
            optimizer.zero_grad(set_to_none=True)
            ddp_disc.eval()

            with autocast(**autocast_kwargs):
                with torch.no_grad():
                    all_feats = rae.encode(images_enc, mode="all")
                # Strip CLS token (first token) from each level, then concat
                # encode(mode='all') returns (B*V, N+1, 1536) with CLS at index 0
                z = torch.cat([all_feats[k][:, 1:, :] for k in sorted(all_feats.keys())], dim=-1)

                # Decoder forward
                logits = ddp_decoder(z, input_size=(h, w), drop_cls_token=False).logits
                # Get the underlying module for unpatchify (DDP doesn't expose it)
                recon = ddp_decoder.module.unpatchify(logits, (h, w))  # (B*V, 3, H, W) ImageNet-norm

                # Denormalize to [0,1]
                recon_01 = recon * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                recon_pm1 = recon_01 * 2.0 - 1.0  # [-1, 1]

                rec_loss = F.l1_loss(recon_01, images_01)

                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_pm1)
                else:
                    lpips_loss = rec_loss.new_zeros(())

                recon_total = recon_weight * rec_loss + perceptual_weight * lpips_loss

                if use_gan:
                    i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(recon_pm1, output_size=(224, 224))
                    recon_crop = TF.crop(recon_pm1, i_crop, j_crop, h_crop, w_crop)
                    fake_aug = disc_aug.aug(recon_crop)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)

            # ---- Backward ----
            if use_gan:
                adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, max_d_weight)
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total

            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_decoder.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_decoder.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            del recon, recon_01, recon_pm1
            update_ema(ema_decoder, ddp_decoder.module, ema_decay)

            # ---- Discriminator update ----
            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                ddp_decoder.eval()
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        with torch.no_grad():
                            logits_d = ddp_decoder(z, input_size=(h, w), drop_cls_token=False).logits
                            recon_d = ddp_decoder.module.unpatchify(logits_d, (h, w))
                            recon_d_01 = recon_d * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                            recon_d_pm1 = recon_d_01 * 2.0 - 1.0
                            # Discretize
                            fake_det = recon_d_pm1.clamp(-1.0, 1.0)
                            fake_det = torch.round((fake_det + 1.0) * 127.5) / 127.5 - 1.0

                        i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(real_normed, output_size=(224, 224))
                        fake_crop = TF.crop(fake_det, i_crop, j_crop, h_crop, w_crop)
                        real_crop = TF.crop(real_normed, i_crop, j_crop, h_crop, w_crop)
                        fake_input = disc_aug.aug(fake_crop)
                        real_input = disc_aug.aug(real_crop)
                        logits_fake_d, logits_real_d = ddp_disc(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real_d, logits_fake_d)
                        accuracy = (logits_real_d > logits_fake_d).float().mean()

                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()

                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real_d.detach().mean(),
                        "logits_fake": logits_fake_d.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    if disc_scheduler is not None:
                        disc_scheduler.step()

                ddp_disc.eval()
                ddp_decoder.train()

            # ---- Logging ----
            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "gan/weight": adaptive_weight.detach().item() if isinstance(adaptive_weight, torch.Tensor) else adaptive_weight,
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update({
                        "loss/disc": disc_metrics["disc_loss"].item(),
                        "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                        "disc/logits_real": disc_metrics["logits_real"].item(),
                        "disc/logits_fake": disc_metrics["logits_fake"].item(),
                        "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                    })
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)

            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0 and rank == 0:
                save_checkpoint(
                    f"{checkpoint_dir}/{global_step:07d}.pt",
                    global_step, epoch,
                    ddp_decoder, ema_decoder, optimizer, scheduler,
                    ddp_disc, disc_optimizer, disc_scheduler,
                )

            global_step += 1

            # ---- Periodic validation ----
            if rank == 0 and global_step % 5000 == 0:
                logger.info(f"Running validation at step {global_step}...")
                torch.cuda.empty_cache()
                val_stats = run_validation(rae, ema_decoder, val_loader, device, lpips_fn=lpips if use_lpips else None)
                log_copy = {k: v for k, v in val_stats.items() if k != "val/images"}
                logger.info(f"[Val @ step {global_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in log_copy.items()))
                if args.wandb:
                    wandb_utils.log(log_copy, step=global_step)
                    wandb_utils.log({"val/reconstructions": val_stats["val/images"]}, step=global_step)

        # ---- Epoch summary ----
        if rank == 0 and num_batches > 0:
            epoch_stats = {
                f"epoch/{k}": (epoch_metrics[k] / num_batches).item()
                for k in ["recon", "lpips", "gan", "total"]
            }
            logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    cleanup_distributed()


if __name__ == "__main__":
    main()
