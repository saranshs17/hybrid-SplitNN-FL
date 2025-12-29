import argparse
import yaml
import time
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..nodes.iot_client import IoTClient
from ..nodes.fog_node import FogNode
from ..nodes.cloud_server import CloudServer
from ..core.protocol import SplitFederatedProtocol

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("psfl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def build_random_dataloader(batch_size, num_batches=8, channels=3, h=32, w=32, num_classes=10):
    total = batch_size * num_batches
    x = torch.randn(total, channels, h, w)
    y = torch.randint(0, num_classes, (total,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def make_results_dir(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

def save_log(results_dir: Path, lines: List[str]):
    (results_dir / "log.txt").write_text("\n".join(lines))

def run(config):
    device = torch.device(config.get("device", "cpu"))
    batch_size = config.get("batch_size", 16)
    dataset = config.get("dataset", "random")
    epochs = config.get("epochs", 1)
    num_clients = config.get("num_clients", 2)
    num_fog = config.get("num_fog", 1)

    results_dir = Path("results")
    make_results_dir(results_dir)

    clients = []
    for i in range(num_clients):
        if dataset == "cifar10":
            try:
                from torchvision import datasets, transforms
                transform = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ])
                ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
                subset_size = min(512, len(ds))
                ds = torch.utils.data.Subset(ds, list(range(subset_size)))
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
            except Exception as e:
                logger.warning("torchvision/CIFAR not available or failed: %s — falling back to random", e)
                dl = build_random_dataloader(batch_size)
        else:
            dl = build_random_dataloader(batch_size)
        clients.append(IoTClient(f"client_{i}", dl, device=device))

    fogs = [FogNode(i, device=device) for i in range(num_fog)]
    cloud = CloudServer(device=device)
    proto = SplitFederatedProtocol(clients, fogs, cloud)

    all_batch_losses = []
    all_bytes_per_batch = []
    lines = []
    start_all = time.time()

    for epoch in range(epochs):
        logger.info("Epoch %d/%d starting", epoch+1, epochs)
        epoch_start = time.time()

        for batch_idx in range(1):
            packets = []
            for client in clients:
                smashed, labels = client.forward_pass()
                packets.append((client, smashed, labels))

            processed = []
            for (client, smashed, labels) in packets:
                fog_out = fogs[0].forward_pass(smashed)
                processed.append((client, fog_out, labels, smashed))

            batch_losses = []
            batch_bytes = []
            for (client, fog_out, labels, smashed) in processed:
                loss, logits = cloud.infer_and_loss(fog_out.detach(), labels)  
                cloud.step(loss)
                batch_losses.append(float(loss.detach().cpu().item()))
                bytes_transferred = smashed.numel() * 4
                batch_bytes.append(int(bytes_transferred))

            avg_loss = float(np.mean(batch_losses))
            avg_bytes = int(np.mean(batch_bytes))
            all_batch_losses.append(avg_loss)
            all_bytes_per_batch.append(avg_bytes)

            line = f"Epoch {epoch+1} batch {batch_idx+1} — loss {avg_loss:.6f} — approx bytes {avg_bytes/1024:.2f} KB"
            logger.info(line)
            lines.append(line)

        epoch_elapsed = time.time() - epoch_start
        logger.info("Epoch %d complete in %.3fs", epoch+1, epoch_elapsed)
        lines.append(f"Epoch {epoch+1} runtime: {epoch_elapsed:.3f}s")

    total_elapsed = time.time() - start_all
    lines.append(f"Total runtime: {total_elapsed:.3f}s")
    save_log(results_dir, lines)

    x = list(range(1, len(all_batch_losses)+1))
    if len(x) == 0:
        logger.warning("No batches recorded — nothing to plot")
        return

    loss_path = results_dir / "loss_curve.png"
    bytes_path = results_dir / "bytes_per_batch.png"

    plt.figure(figsize=(6, 3.5))
    plt.plot(x, all_batch_losses, marker='o', linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Prototype: Loss per Step ")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(x, [b/1024.0 for b in all_bytes_per_batch], marker='o', linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("KB per batch (approx)")
    plt.title("Approx Bytes Transferred per Batch (smashed activations)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(bytes_path, dpi=200)
    plt.close()

    logger.info("Saved results: %s, %s, %s", results_dir / "log.txt", loss_path, bytes_path)
    logger.info("Run finished — total time %.3fs", total_elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} not found")
    cfg = yaml.safe_load(open(cfg_path))
    run(cfg)
