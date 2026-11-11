"""Inference helpers for NAFNet (matches CropImagesBlock + PrepareTensorsBlock preprocessing)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.nafnet.model import NAFNet
from src.nafnet.grid_fix_model import BilinearNAFNet
from src.nafnet.rggb_model_wrapper import RGGBModelWrapper

__all__ = [
    "NAFNet",
    "BilinearNAFNet",
    "nafnet_jdd_default",
    "bilinear_nafnet_jdd_default",
    "load_checkpoint",
    "load_rggb_checkpoint",
    "predict_tensor",
    "output_to_numpy_rgb",
]

def bilinear_nafnet_jdd_default(**overrides) -> BilinearNAFNet:
    kwargs = dict(
        img_channel=1,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        output_channels=3,
    )
    kwargs.update(overrides)
    return BilinearNAFNet(**kwargs)

def nafnet_jdd_default(**overrides) -> NAFNet:
    """Same architecture as ``notebooks/nafnet_train.ipynb`` unless overridden."""
    kwargs = dict(
        img_channel=1,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        output_channels=3,
    )
    kwargs.update(overrides)
    return NAFNet(**kwargs)


def load_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> None:

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)


def load_rggb_checkpoint(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    defaults: Optional[dict] = None,
) -> tuple[RGGBModelWrapper, str]:
    """Rebuild ``RGGBModelWrapper`` around a ``nafnet_train_widget`` checkpoint.

    The checkpoint stores only ``model_state_dict``, not the architecture it was
    trained with, so both the factory and the wrapper's dimensions are read back
    off the tensors:

    * ``NAFNet`` upsamples with ``1x1 conv + PixelShuffle(2)``, so its first
      ``ups`` module is a convolution and the state dict carries
      ``model.ups.<i>.0.weight``. ``BilinearNAFNet`` upsamples bilinearly, so the
      same weight lands on ``model.ups.<i>.1.weight``. The two are told apart from
      the keys alone.
    * ``intro.weight`` is ``(width, in_channels, 3, 3)`` and ``ending.weight`` is
      ``(out_channels_4x, width, 3, 3)``, which pins the three numbers
      ``RGGBModelWrapper`` needs. The inner ``img_channel`` / ``output_channels``
      come from ``model.intro.weight`` and ``model.ending.weight``.

    ``enc_blk_nums`` / ``dec_blk_nums`` / ``middle_blk_num`` are not recoverable
    from the shapes and default to ``[1, 1, 1]`` / ``1`` - the panel defaults.
    A run that used something else must pass them via *defaults*; a wrong guess
    still fails loudly on ``load_state_dict(strict=True)``.

    Returns the loaded model in ``eval`` mode together with the factory name.
    """
    defaults = dict(defaults or {})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]

    if "model.ups.0.0.weight" in state:
        factory = nafnet_jdd_default
    elif "model.ups.0.1.weight" in state:
        factory = bilinear_nafnet_jdd_default
    else:
        raise ValueError(
            f"Cannot tell the architecture from {path}: neither "
            f"'model.ups.0.0.weight' (NAFNet) nor 'model.ups.0.1.weight' "
            f"(BilinearNAFNet) is present."
        )

    width, in_channels, _, _ = state["intro.weight"].shape
    out_channels_4x = state["ending.weight"].shape[0]

    model = RGGBModelWrapper(
        factory(
            img_channel=int(state["model.intro.weight"].shape[1]),
            output_channels=int(state["model.ending.weight"].shape[0]),
            **defaults,
        ),
        in_channels=int(in_channels),
        out_channels_4x=int(out_channels_4x),
        width=int(width),
    )
    model.load_state_dict(state, strict=strict)
    if map_location is not None:
        model = model.to(map_location)
    return model.eval(), factory.__name__


@torch.inference_mode()
def predict_tensor(
    model: torch.nn.Module,
    input_bchw: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Run forward pass; returns ``[B, 3, H, W]`` on CPU float32."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = input_bchw.to(device, dtype=torch.float32)
    out = model(x)
    return out.detach().float().cpu()


def output_to_numpy_rgb(out_bchw: torch.Tensor, batch_index: int = 0) -> np.ndarray:
    """``[B, 3, H, W]`` → RGB float array ``[H, W, 3]`` in ``[0, 1]``."""
    x = out_bchw[batch_index].clamp(0, 1).permute(1, 2, 0).numpy()
    return x.astype(np.float32)
