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


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.shortcut: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.block = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


class DownsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock3D(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UpsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.fuse = ResidualConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.up(x), skip], dim=1))


class ResUNetGenerator3D(nn.Module):
    """3D residual U-Net generator adapted from the reference 2D CBCT-to-CT CycleGAN project."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ngf = int(base_channels)

        self.enc1 = ResidualConvBlock3D(in_channels, ngf)
        self.enc2 = DownsampleBlock3D(ngf, ngf * 2)
        self.enc3 = DownsampleBlock3D(ngf * 2, ngf * 4)
        self.enc4 = DownsampleBlock3D(ngf * 4, ngf * 8)
        self.enc5 = DownsampleBlock3D(ngf * 8, ngf * 8)

        self.bottleneck = nn.Sequential(
            ResidualConvBlock3D(ngf * 8, ngf * 8),
            ResidualConvBlock3D(ngf * 8, ngf * 8),
        )

        self.dec4 = UpsampleBlock3D(ngf * 8, ngf * 8, ngf * 8)
        self.dec3 = UpsampleBlock3D(ngf * 8, ngf * 4, ngf * 4)
        self.dec2 = UpsampleBlock3D(ngf * 4, ngf * 2, ngf * 2)
        self.dec1 = UpsampleBlock3D(ngf * 2, ngf, ngf)

        self.out = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(ngf, out_channels, kernel_size=3),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        z = self.bottleneck(e5)
        d4 = self.dec4(z, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        residual = self.out(d1)
        if self.in_channels == self.out_channels:
            return torch.tanh(x + residual)
        return torch.tanh(residual)


class SpectralNormDiscriminator3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16, use_dropout: bool = False) -> None:
        super().__init__()
        ndf = int(base_channels)
        layers: list[nn.Module] = [
            nn.utils.spectral_norm(nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm3d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(0.5))
        layers.append(nn.utils.spectral_norm(nn.Conv3d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)))
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


def make_generator(config: Mapping[str, object]) -> nn.Module:
    generator_type = str(config.get("generator_type", "resnet")).lower()
    if generator_type == "resunet":
        return ResUNetGenerator3D(
            in_channels=1,
            out_channels=1,
            base_channels=int(config["generator_channels"]),
        )
    return ResnetGenerator3D(
        in_channels=1,
        out_channels=1,
        base_channels=int(config["generator_channels"]),
        num_res_blocks=int(config["generator_res_blocks"]),
    )


def make_discriminator(config: Mapping[str, object]) -> nn.Module:
    discriminator_type = str(config.get("discriminator_type", "patchgan")).lower()
    if discriminator_type == "spectral" or bool(config.get("discriminator_spectral_norm", False)):
        return SpectralNormDiscriminator3D(
            in_channels=1,
            base_channels=int(config["discriminator_channels"]),
            use_dropout=bool(config.get("discriminator_dropout", False)),
        )
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
