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

        try:
            batch = next(iter(val_loader))
            cbct = batch["cbct"][: int(config["visdom_num_images"])].to(self.device)
            target_ct = batch["ct"][: cbct.shape[0]].to(self.device)
            sample = torch.randn_like(target_ct)

            vis_scheduler = DDPMScheduler(num_train_timesteps=int(config["num_train_timesteps"]))
            steps = min(int(config["visdom_num_inference_steps"]), int(config["num_train_timesteps"]))
            vis_scheduler.set_timesteps(num_inference_steps=steps, device=self.device)
            _move_scheduler_tensors_to_device(vis_scheduler, self.device)

            was_training = model.training
            model.eval()
            for timestep in vis_scheduler.timesteps:
                t_int = int(timestep.item())
                timesteps = torch.full((sample.shape[0],), t_int, device=self.device, dtype=torch.long)
                model_input = torch.cat((cbct, sample), dim=1)
                with autocast(device_type=self.device.type, enabled=use_amp):
                    model_output = model(x=model_input, timesteps=timesteps)
                sample, _ = vis_scheduler.step(model_output, t_int, sample)
            if was_training:
                model.train()

            panel = torch.cat((cbct, target_ct, sample), dim=0)
            panel = ((panel.detach().float().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
            self.viz.images(
                panel,
                nrow=cbct.shape[0],
                win="samples",
                opts={"title": f"Epoch {epoch}: CBCT / real CT / generated CT"},
            )
        except Exception as exc:
            print(f"Visdom sample visualization failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D conditional DDPM baseline for CBCT-to-CT synthesis.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Override device, for example cuda or cpu.")
    return parser.parse_args()


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
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
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
    for batch in progress:
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
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
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
    for batch in progress:
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
            "device",
            "spatial_size",
            "spacing",
            "hu_min",
            "hu_max",
            "batch_size",
            "num_workers",
            "epochs",
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
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = DDPMScheduler(num_train_timesteps=int(config["num_train_timesteps"]))
    scaler = GradScaler(device.type, enabled=use_amp)
    visualizer = VisdomReporter(config, device)

    start_epoch = 1
    best_val_loss = math.inf
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    print(f"Device: {device}")
    print(f"Train cases: {train_case_count} | Validation cases: {val_case_count}")
    print(f"Train samples/epoch: {len(train_ds)} | Validation samples: {len(val_ds)}")
    show_batch_progress = bool(config.get("show_batch_progress", True))
    if not show_batch_progress:
        print("Batch tqdm progress is disabled. Set show_batch_progress=true to enable per-epoch train/val progress bars.")

    total_start = time.time()
    total_epochs = int(config["epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, scheduler, optimizer, scaler, device, use_amp, epoch, total_epochs, show_batch_progress
        )

        should_validate = epoch == 1 or epoch % int(config["val_interval"]) == 0
        val_loss = best_val_loss
        if should_validate:
            val_loss = validate(model, val_loader, scheduler, device, use_amp, epoch, total_epochs, show_batch_progress)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_loss, config)
            visualizer.plot_losses(epoch, train_loss, val_loss)
            if epoch % int(config["visdom_interval"]) == 0:
                visualizer.show_samples(model, scheduler, val_loader, config, use_amp, epoch)

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, best_val_loss, config)
        if epoch % int(config["save_interval"]) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, best_val_loss, config)

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
