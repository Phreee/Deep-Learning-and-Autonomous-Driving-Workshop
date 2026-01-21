import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class DrivingDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], float(row["steering"])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, steering = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([steering], dtype=torch.float32)


class SimpleSteeringCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


def resize_and_normalize(img, size=(66, 200)):
    img = img.resize((size[1], size[0]))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "tutorial_runs", "behavioral_cloning_data", "driving_log_local.csv")
    out_dir = os.path.join(base_dir, "tutorial_runs", "output")
    os.makedirs(out_dir, exist_ok=True)

    dataset = DrivingDataset(data_csv, transform=resize_and_normalize)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    model = SimpleSteeringCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
    log_rows = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, steering in train_loader:
            imgs = imgs.to(device)
            steering = steering.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, steering)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, steering in val_loader:
                imgs = imgs.to(device)
                steering = steering.to(device)
                preds = model(imgs)
                loss = criterion(preds, steering)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        log_rows.append([epoch, train_loss, val_loss])
        print(f"epoch {epoch} train_loss {train_loss:.6f} val_loss {val_loss:.6f}")

    elapsed = time.time() - start
    model_path = os.path.join(out_dir, "behavioral_cloning_model.pth")
    torch.save(model.state_dict(), model_path)

    log_path = os.path.join(out_dir, "behavioral_cloning_train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(log_rows)

    metrics = {
        "samples": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "train_loss_last": round(log_rows[-1][1], 6),
        "val_loss_last": round(log_rows[-1][2], 6),
        "elapsed_sec": round(elapsed, 2),
    }
    metrics_path = os.path.join(out_dir, "behavioral_cloning_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    epochs_arr = [r[0] for r in log_rows]
    train_losses = [r[1] for r in log_rows]
    val_losses = [r[2] for r in log_rows]
    plt.figure(figsize=(4, 3))
    plt.plot(epochs_arr, train_losses, label="train_loss")
    plt.plot(epochs_arr, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("Behavioral Cloning Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "behavioral_cloning_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("saved_model", model_path)
    print("saved_log", log_path)
    print("saved_metrics", metrics_path)
    print("saved_plot", plot_path)


if __name__ == "__main__":
    main()
