from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generative.networks.schedulers import DDPMScheduler  # noqa: E402
from utils import (  # noqa: E402
    CachedSlicePairDataset,
    NiftiSlicePairDataset,
    build_pair_list,
    create_model,
    load_checkpoint,
    print_config_summary,
    read_config,
    save_checkpoint,
    seed_worker,
    set_seed,
)


def _move_scheduler_tensors_to_device(scheduler: DDPMScheduler, device: torch.device) -> None:
    for name in ("betas", "alphas", "alphas_cumprod", "one", "timesteps"):
        value = getattr(scheduler, name, None)
        if torch.is_tensor(value):
            setattr(scheduler, name, value.to(device))


class VisdomReporter:
    def __init__(self, config: dict, device: torch.device) -> None:
        self.enabled = bool(config.get("visdom_enabled", False))
        self.device = device
        self.loss_initialized = False
        self.viz = None

        if not self.enabled:
            return

        try:
            from visdom import Visdom

            self.viz = Visdom(
                server=str(config.get("visdom_server", "http://localhost")),
                port=int(config.get("visdom_port", 8097)),
                env=str(config.get("visdom_env", "ddpm_pelvis")),
                use_incoming_socket=bool(config.get("visdom_use_incoming_socket", False)),
                raise_exceptions=False,
            )
            if not self.viz.check_connection(timeout_seconds=2):
                print("Visdom is enabled, but the server is not reachable. Start it with: python -m visdom.server")
                self.enabled = False
        except Exception as exc:
            print(f"Visdom is enabled, but initialization failed: {exc}")
            self.enabled = False

    def plot_losses(self, epoch: int, train_loss: float, val_loss: float) -> None:
        if not self.enabled or self.viz is None:
            return

        update = "append" if self.loss_initialized else None
        self.viz.line(
            X=torch.tensor([[epoch, epoch]], dtype=torch.float32),
            Y=torch.tensor([[train_loss, val_loss]], dtype=torch.float32),
            win="loss",
            update=update,
            opts={
                "title": "DDPM Loss",
                "xlabel": "epoch",
                "ylabel": "MSE",
                "legend": ["train", "val"],
            },
        )
        self.loss_initialized = True

    def _orient_for_display(self, images: torch.Tensor, config: dict) -> torch.Tensor:
        rotate_k = int(config.get("visdom_rotate_k", 1)) % 4
        if rotate_k:
            images = torch.rot90(images, k=rotate_k, dims=(-2, -1))
        if bool(config.get("visdom_flip_lr", False)):
            images = torch.flip(images, dims=(-1,))
        if bool(config.get("visdom_flip_ud", False)):
            images = torch.flip(images, dims=(-2,))
        return images

    def _to_visdom_range(self, images: torch.Tensor, config: dict) -> torch.Tensor:
        images = self._orient_for_display(images, config)
        return ((images.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)

    @torch.no_grad()
    def show_input_preview(self, val_loader: DataLoader, config: dict) -> None:
        if not self.enabled or self.viz is None:
            return

        try:
            batch = next(iter(val_loader))
            num_images = max(1, int(config.get("visdom_num_images", 4)))
            cbct = batch["cbct"][:num_images]
            target_ct = batch["ct"][: cbct.shape[0]]
            panel = torch.stack((cbct, target_ct), dim=1).flatten(0, 1)
            panel = self._to_visdom_range(panel, config)
            self.viz.images(
                panel,
                nrow=2,
                win="inputs",
                opts={"title": "Validation preview: CBCT / real CT"},
            )
            print(f"Visdom input preview updated: groups={cbct.shape[0]}")
        except Exception as exc:
            print(f"Visdom input preview failed: {exc}")

    @torch.no_grad()
    def show_samples(
        self,
        model: torch.nn.Module,
        scheduler: DDPMScheduler,
        val_loader: DataLoader,
        config: dict,
        use_amp: bool,
        epoch: int,
    ) -> None:
        if not self.enabled or self.viz is None:
            return

        was_training = model.training
        try:
            batch = next(iter(val_loader))
            num_images = max(1, int(config.get("visdom_num_images", 4)))
            inference_batch_size = max(1, int(config.get("visdom_inference_batch_size", 1)))
            cbct_cpu = batch["cbct"][:num_images]
            target_ct_cpu = batch["ct"][: cbct_cpu.shape[0]]
            inference_batch_size = min(inference_batch_size, cbct_cpu.shape[0])
            preview_timestep = int(config.get("visdom_preview_timestep", int(config["num_train_timesteps"]) // 2))
            preview_timestep = max(0, min(preview_timestep, int(config["num_train_timesteps"]) - 1))

            model.eval()
            panel_chunks = []
            for start in range(0, cbct_cpu.shape[0], inference_batch_size):
                end = min(start + inference_batch_size, cbct_cpu.shape[0])
                cbct = cbct_cpu[start:end].to(self.device, non_blocking=True)
                target_ct = target_ct_cpu[start:end].to(self.device, non_blocking=True)
                timesteps = torch.full((target_ct.shape[0],), preview_timestep, device=self.device, dtype=torch.long)
                noise = torch.randn_like(target_ct)
                noisy_ct = scheduler.add_noise(original_samples=target_ct, noise=noise, timesteps=timesteps)
                model_input = torch.cat((cbct, noisy_ct), dim=1)

                with autocast(device_type=self.device.type, enabled=use_amp):
                    predicted_noise = model(x=model_input, timesteps=timesteps)

                alphas_cumprod = scheduler.alphas_cumprod.to(device=self.device, dtype=target_ct.dtype)
                alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                predicted_ct = (noisy_ct - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
                panel_chunks.append(torch.stack((cbct, target_ct, predicted_ct.clamp(-1, 1)), dim=1).detach().cpu())

                del cbct, target_ct, timesteps, noise, noisy_ct, model_input, predicted_noise, predicted_ct
            if was_training:
                model.train()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            panel = torch.cat(panel_chunks, dim=0).flatten(0, 1)
            panel = self._to_visdom_range(panel, config)
            self.viz.images(
                panel,
                nrow=3,
                win="samples",
                opts={"title": f"Epoch {epoch}: CBCT / real CT / denoised prediction"},
            )
            print(
                f"Visdom samples updated: epoch={epoch}, groups={cbct_cpu.shape[0]}, "
                f"inference_batch_size={inference_batch_size}"
            )
        except Exception as exc:
            if was_training:
                model.train()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Visdom sample visualization failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D conditional DDPM baseline for CBCT-to-CT synthesis.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Resume from latest, best, or a checkpoint path. Without a value, uses output_dir/latest.pt.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable config auto_resume and start from epoch 1.")
    parser.add_argument("--device", type=str, default=None, help="Override device, for example cuda or cpu.")
    return parser.parse_args()


def _resolve_checkpoint_alias(value: str | Path, output_dir: Path) -> Path:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "latest":
        return output_dir / "latest.pt"
    if lowered == "best":
        return output_dir / "best.pt"

    path = Path(text)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_resume_checkpoint(args: argparse.Namespace, config: dict, output_dir: Path) -> Path | None:
    if args.no_resume:
        return None

    if args.resume is not None:
        return _resolve_checkpoint_alias(args.resume, output_dir)

    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        return _resolve_checkpoint_alias(resume_checkpoint, output_dir)

    if bool(config.get("auto_resume", False)):
        latest_path = output_dir / "latest.pt"
        if latest_path.exists():
            return latest_path
        print(f"auto_resume is enabled, but no checkpoint was found at: {latest_path}")

    return None


def resolve_step_count(loader: DataLoader, configured_steps: int | None) -> int:
    loader_steps = len(loader)
    if configured_steps is None:
        return loader_steps

    configured_steps = int(configured_steps)
    if configured_steps <= 0:
        return loader_steps

    return min(configured_steps, loader_steps)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
    steps_per_epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        range(steps_per_epoch),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:04d}/{total_epochs:04d} train",
        unit="batch",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=1.0,
        maxinterval=5.0,
        miniters=10,
        leave=True,
        disable=not show_progress,
    )
    loader_iter = iter(loader)
    for _ in progress:
        batch = next(loader_iter)
        cbct = batch["cbct"].to(device=device, non_blocking=True)
        ct = batch["ct"].to(device=device, non_blocking=True)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (ct.shape[0],), device=device).long()
        noise = torch.randn_like(ct)
        noisy_ct = scheduler.add_noise(original_samples=ct, noise=noise, timesteps=timesteps)
        model_input = torch.cat((cbct, noisy_ct), dim=1)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            prediction = model(x=model_input, timesteps=timesteps)
            loss = F.mse_loss(prediction.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_batches += 1
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}", avg=f"{total_loss / total_batches:.4f}", refresh=False
        )

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
    steps_per_epoch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        range(steps_per_epoch),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:04d}/{total_epochs:04d} val",
        unit="batch",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=1.0,
        maxinterval=5.0,
        miniters=10,
        leave=True,
        disable=not show_progress,
    )
    loader_iter = iter(loader)
    for _ in progress:
        batch = next(loader_iter)
        cbct = batch["cbct"].to(device=device, non_blocking=True)
        ct = batch["ct"].to(device=device, non_blocking=True)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (ct.shape[0],), device=device).long()
        noise = torch.randn_like(ct)
        noisy_ct = scheduler.add_noise(original_samples=ct, noise=noise, timesteps=timesteps)
        model_input = torch.cat((cbct, noisy_ct), dim=1)

        with autocast(device_type=device.type, enabled=use_amp):
            prediction = model(x=model_input, timesteps=timesteps)
            loss = F.mse_loss(prediction.float(), noise.float())

        total_loss += float(loss.detach().item())
        total_batches += 1
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}", avg=f"{total_loss / total_batches:.4f}", refresh=False
        )

    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["seed"]))

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print_config_summary(
        config,
        [
            "data_root",
            "data_layout",
            "preprocessed_dir",
            "output_dir",
            "use_preprocessed",
            "auto_resume",
            "resume_checkpoint",
            "device",
            "spatial_size",
            "spacing",
            "hu_min",
            "hu_max",
            "batch_size",
            "num_workers",
            "epochs",
            "train_steps_per_epoch",
            "val_steps_per_epoch",
            "lr",
            "num_train_timesteps",
            "num_inference_steps",
            "model_channels",
            "attention_levels",
            "num_res_blocks",
            "num_head_channels",
            "norm_num_groups",
            "amp",
            "visdom_enabled",
            "visdom_env",
            "visdom_num_images",
            "visdom_inference_batch_size",
            "visdom_num_workers",
            "visdom_preview_on_start",
            "visdom_use_incoming_socket",
            "visdom_preview_timestep",
            "visdom_rotate_k",
            "visdom_flip_lr",
            "visdom_flip_ud",
        ],
        title="Training configuration",
    )

    if bool(config.get("use_preprocessed", False)):
        train_ds = CachedSlicePairDataset(config["preprocessed_dir"], "train")
        val_ds = CachedSlicePairDataset(config["preprocessed_dir"], "val")
        train_case_count = len(train_ds.case_ids)
        val_case_count = len(val_ds.case_ids)
        print(f"Using preprocessed slices from: {config['preprocessed_dir']}")
    else:
        train_pairs = build_pair_list(config, "train")
        val_pairs = build_pair_list(config, "val")
        train_ds = NiftiSlicePairDataset(train_pairs, config, training=True)
        val_ds = NiftiSlicePairDataset(val_pairs, config, training=False)
        train_case_count = len(train_pairs)
        val_case_count = len(val_pairs)

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    vis_generator = torch.Generator()
    vis_generator.manual_seed(int(config["seed"]) + 1)
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    vis_loader_kwargs = dict(loader_kwargs)
    vis_loader_kwargs["batch_size"] = max(1, int(config.get("visdom_num_images", 4)))
    vis_loader_kwargs["num_workers"] = max(0, int(config.get("visdom_num_workers", 0)))
    vis_loader_kwargs["persistent_workers"] = int(vis_loader_kwargs["num_workers"]) > 0
    vis_loader = DataLoader(val_ds, shuffle=True, generator=vis_generator, drop_last=False, **vis_loader_kwargs)
    train_steps_per_epoch = resolve_step_count(train_loader, config.get("train_steps_per_epoch"))
    val_steps_per_epoch = resolve_step_count(val_loader, config.get("val_steps_per_epoch"))

    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = DDPMScheduler(num_train_timesteps=int(config["num_train_timesteps"]))
    scaler = GradScaler(device.type, enabled=use_amp)
    visualizer = VisdomReporter(config, device)
    if bool(config.get("visdom_preview_on_start", True)):
        visualizer.show_input_preview(vis_loader, config)

    start_epoch = 1
    best_val_loss = math.inf
    resume_path = resolve_resume_checkpoint(args, config, output_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        print(f"Resumed epoch: {start_epoch - 1} | Next epoch: {start_epoch} | Best val loss: {best_val_loss:.6f}")
    else:
        print("Resume: disabled or no checkpoint found; starting from epoch 1.")

    print(f"Device: {device}")
    print(f"Train cases: {train_case_count} | Validation cases: {val_case_count}")
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}")
    print(f"Train batches available: {len(train_loader)} | Train steps/epoch: {train_steps_per_epoch}")
    print(f"Validation batches available: {len(val_loader)} | Validation steps/eval: {val_steps_per_epoch}")
    show_batch_progress = bool(config.get("show_batch_progress", True))
    if not show_batch_progress:
        print("Batch tqdm progress is disabled. Set show_batch_progress=true to enable per-epoch train/val progress bars.")

    total_start = time.time()
    total_epochs = int(config["epochs"])
    if start_epoch > total_epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}, "
            f"but config epochs={total_epochs}. Increase epochs to continue training."
        )
        return

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            scheduler,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            total_epochs,
            show_batch_progress,
            train_steps_per_epoch,
        )

        should_validate = epoch == 1 or epoch % int(config["val_interval"]) == 0
        val_loss = best_val_loss
        if should_validate:
            val_loss = validate(
                model, val_loader, scheduler, device, use_amp, epoch, total_epochs, show_batch_progress, val_steps_per_epoch
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_loss, config, scaler=scaler)
            visualizer.plot_losses(epoch, train_loss, val_loss)
            if epoch % int(config["visdom_interval"]) == 0:
                visualizer.show_samples(model, scheduler, vis_loader, config, use_amp, epoch)

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, best_val_loss, config, scaler=scaler)
        if epoch % int(config["save_interval"]) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, best_val_loss, config, scaler=scaler
            )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:04d}/{int(config['epochs']):04d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val_loss={best_val_loss:.6f} time={elapsed:.1f}s"
        )

    print(f"Training completed in {(time.time() - total_start) / 60.0:.1f} min")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
