import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from models.dual_stream import DualStreamDeepfakeDetector
from utils.dataset import DeepfakeDataset


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khsl et al.).
    Pulls authentic face representations together into a compact cluster,
    pushing synthetic variations away.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # Remove self-similarity from diagonals
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask

        # Numerical stability via log-sum-exp
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_pos_pairs

        return mean_log_prob_pos.mean()


def train_epoch(model, loader, bce_criterion, supcon_criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        use_amp = (device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Extract spatial and frequency embeddings before the dense head
            f_spatial = model.spatial_backbone(images)
            fft_map = model.fft_extractor(images)
            f_freq = model.freq_encoder(fft_map)
            f_fused = model.fusion(f_spatial, f_freq)
            combined = torch.cat([f_fused, f_freq], dim=1)

            logits = model.classifier(combined).squeeze(-1)

            loss_bce = bce_criterion(logits, labels)
            loss_con = supcon_criterion(combined, labels)
            loss = loss_bce + (0.15 * loss_con)  # Hybrid objective

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        running_loss += loss.item() * images.size(0)
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    epoch_acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    return epoch_loss, epoch_acc, epoch_auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        running_loss += loss.item() * images.size(0)
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.5

    val_acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    return val_loss, val_acc, val_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training target hardware: {device}")

    train_data = DeepfakeDataset(root_dir="data", split="train")
    val_data = DeepfakeDataset(root_dir="data", split="val")

    print(f"[*] Dataset samples detected: {len(train_data)} train | {len(val_data)} val")

    if len(train_data) == 0 or len(val_data) == 0:
        print("[!] Dataset folders empty. Populate data/ before running full training.")
        return

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)

    model = DualStreamDeepfakeDetector(pretrained=True).to(device)
    bce_criterion = nn.BCEWithLogitsLoss()
    supcon_criterion = SupConLoss(temperature=0.07)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    epochs = 10

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc = train_epoch(model, train_loader, bce_criterion, supcon_criterion, optimizer, scaler, device)
        v_loss, v_acc, v_auc = validate(model, val_loader, bce_criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.2f}% | AUC: {tr_auc:.4f} || "
              f"Val Loss: {v_loss:.4f} | Acc: {v_acc*100:.2f}% | AUC: {v_auc:.4f}")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(model.state_dict(), "checkpoints/best_dual_stream_model.pth")
            print(f"    --> Saved best checkpoint: checkpoints/best_dual_stream_model.pth (AUC: {best_val_auc:.4f})")


if __name__ == "__main__":
    main()