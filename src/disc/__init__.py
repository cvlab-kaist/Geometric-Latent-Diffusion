import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS


def build_discriminator(
    config: dict,
    device: torch.device,
) -> tuple[DinoDiscriminator, DiffAug]:
    """Instantiate Dino-based discriminator and its augmentation policy."""
    arch_cfg = config.get("arch", {})
    ckpt_path = arch_cfg.get("dino_ckpt_path")
    if not ckpt_path:
        raise ValueError("DINO discriminator requires 'dino_ckpt_path' in gan.disc.arch.")
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=ckpt_path,
        ks=int(arch_cfg.get("ks", 3)),
        key_depths=tuple(arch_cfg.get("key_depths", (2, 5, 8, 11))),
        norm_type=arch_cfg.get("norm_type", "bn"),
        using_spec_norm=bool(arch_cfg.get("using_spec_norm", True)),
        norm_eps=float(arch_cfg.get("norm_eps", 1e-6)),
        recipe=arch_cfg.get("recipe", "S_8"),
    ).to(device)

    aug_cfg = config.get("augment", {})
    augment = DiffAug(prob=float(aug_cfg.get("prob", 1.0)), cutout=float(aug_cfg.get("cutout", 0.0)))
    return disc, augment


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "build_discriminator",
]
