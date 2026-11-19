import os
import torch
import numpy as np
from PIL import Image, ImageOps
from models.dual_stream import DualStreamDeepfakeDetector
from utils.dataset import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/best_dual_stream_model.pth"

print(f"[*] Testing device: {device}")
print(f"[*] Checkpoint exists: {os.path.exists(checkpoint_path)}")

# 1. Initialize and load model
model = DualStreamDeepfakeDetector(pretrained=False)
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print("[*] Successfully loaded checkpoint.")
else:
    print("[!] ERROR: Checkpoint NOT found. You must run train.py first.")
    exit(1)

model.to(device)
model.eval()

# 2. Check sample validation images from your data directory
val_fake_dir = "data/val/fake"
val_real_dir = "data/val/real"

def evaluate_image(img_path):
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    raw_np = np.array(pil_img.resize((224, 224)))
    
    transform = get_transforms(split="val")
    tensor_img = transform(image=raw_np)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = model(tensor_img).squeeze(-1)
        prob = torch.sigmoid(logit).item()
    return prob

print("\n--- Testing 3 Real Validation Images ---")
for f in os.listdir(val_real_dir)[:3]:
    p = os.path.join(val_real_dir, f)
    prob = evaluate_image(p)
    print(f"Real sample [{f}]: Fake Probability = {prob:.4f} -> {'FAKE' if prob >= 0.5 else 'REAL'}")

print("\n--- Testing 3 Fake Validation Images ---")
for f in os.listdir(val_fake_dir)[:3]:
    p = os.path.join(val_fake_dir, f)
    prob = evaluate_image(p)
    print(f"Fake sample [{f}]: Fake Probability = {prob:.4f} -> {'FAKE' if prob >= 0.5 else 'REAL'}")