from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative.networks.nets import PatchDiscriminator


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator3D(nn.Module):
    """Memory-conscious 3D ResNet generator for patch-based CycleGAN."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        channels = int(base_channels)
        layers: list[nn.Module] = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(in_channels, channels, kernel_size=7, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(2):
            next_channels = channels * 2
            layers.extend(
                [
                    nn.Conv3d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(next_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        for _ in range(int(num_res_blocks)):
            layers.append(ResidualBlock3D(channels))

        for _ in range(2):
            next_channels = channels // 2
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.ReflectionPad3d(1),
                    nn.Conv3d(channels, next_channels, kernel_size=3, bias=False),
                    nn.InstanceNorm3d(next_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        layers.extend(
            [
                nn.ReflectionPad3d(3),
                nn.Conv3d(channels, out_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def make_generator(config: Mapping[str, object]) -> ResnetGenerator3D:
    return ResnetGenerator3D(
        in_channels=1,
        out_channels=1,
        base_channels=int(config["generator_channels"]),
        num_res_blocks=int(config["generator_res_blocks"]),
    )


def make_discriminator(config: Mapping[str, object]) -> PatchDiscriminator:
    return PatchDiscriminator(
        spatial_dims=3,
        num_channels=int(config["discriminator_channels"]),
        in_channels=1,
        out_channels=1,
        num_layers_d=int(config["discriminator_layers"]),
        norm="INSTANCE",
        bias=False,
        last_conv_kernel_size=1,
    )


def get_discriminator_logits(output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_requires_grad(models: nn.Module | list[nn.Module], requires_grad: bool) -> None:
    if not isinstance(models, list):
        models = [models]
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad)
