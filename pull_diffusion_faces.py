import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATA_DIR = "data"

for split in ["train", "val"]:
    os.makedirs(os.path.join(DATA_DIR, split, "fake"), exist_ok=True)

print("[*] Streaming multi-generator samples (Midjourney, Stable Diffusion, ADM) from Tiny-GenImage...")

# Parquet-native multi-diffusion benchmark
dataset = load_dataset(
    "TheKernel01/Tiny-GenImage",
    split="train",
    streaming=True
)

train_quota = 120
val_quota = 30
saved_train = 0
saved_val = 0

for item in tqdm(dataset, desc="Streaming Diffusion/MJ Batches"):
    try:
        # label: 0 = Real, 1 = Fake (Midjourney, SD, ADM, etc.)
        if int(item["label"]) == 1:
            img = item["image"].convert("RGB")
            
            if saved_train < train_quota:
                fname = f"gen_fake_train_{saved_train:04d}.jpg"
                img.save(os.path.join(DATA_DIR, "train", "fake", fname))
                saved_train += 1
            elif saved_val < val_quota:
                fname = f"gen_fake_val_{saved_val:04d}.jpg"
                img.save(os.path.join(DATA_DIR, "val", "fake", fname))
                saved_val += 1
                
        if saved_train >= train_quota and saved_val >= val_quota:
            break
    except Exception:
        continue

print(f"\n[+] Successfully added {saved_train} modern AI (Diffusion/MJ) samples to data/train/fake")
print(f"[+] Successfully added {saved_val} modern AI (Diffusion/MJ) samples to data/val/fake")