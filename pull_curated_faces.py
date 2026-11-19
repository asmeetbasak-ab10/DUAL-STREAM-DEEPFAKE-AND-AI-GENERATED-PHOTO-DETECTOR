import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATA_DIR = "data"

# Ensure directories exist without deleting root folders
for split in ["train", "val"]:
    for cat in ["real", "fake"]:
        os.makedirs(os.path.join(DATA_DIR, split, cat), exist_ok=True)

print("[*] Streaming verified 140K Real vs Fake Human Faces from Hugging Face...")

dataset = load_dataset("TheKernel01/140k-Real-and-Fake-Faces", split="train", streaming=True)

limits = {
    "train_real": 150,
    "train_fake": 150,
    "val_real": 30,
    "val_fake": 30
}
counts = {k: 0 for k in limits}

for item in tqdm(dataset, desc="Streaming Faces"):
    try:
        img = item["image"].convert("RGB")
        label = int(item["label"])  # 0: Real (FFHQ), 1: Fake (StyleGAN)

        if label == 0:
            if counts["train_real"] < limits["train_real"]:
                img.save(os.path.join(DATA_DIR, "train", "real", f"real_{counts['train_real']:04d}.jpg"))
                counts["train_real"] += 1
            elif counts["val_real"] < limits["val_real"]:
                img.save(os.path.join(DATA_DIR, "val", "real", f"real_{counts['val_real']:04d}.jpg"))
                counts["val_real"] += 1
        elif label == 1:
            if counts["train_fake"] < limits["train_fake"]:
                img.save(os.path.join(DATA_DIR, "train", "fake", f"fake_{counts['train_fake']:04d}.jpg"))
                counts["train_fake"] += 1
            elif counts["val_fake"] < limits["val_fake"]:
                img.save(os.path.join(DATA_DIR, "val", "fake", f"fake_{counts['val_fake']:04d}.jpg"))
                counts["val_fake"] += 1

        if all(counts[k] >= limits[k] for k in limits):
            break
    except Exception:
        continue

print("\n[+] Dataset saved:")
print(f"    Train: {counts['train_real']} Real | {counts['train_fake']} Fake")
print(f"    Val:   {counts['val_real']} Real | {counts['val_fake']} Fake")