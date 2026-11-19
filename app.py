import io
import os
import base64
import time
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import matplotlib

# Use non-interactive backend for server-side plot rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Register HEIC/HEIF/AVIF support if pillow-heif is installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Direct import of your exact model class and transforms[cite: 3, 8]
from models.dual_stream import DualStreamDeepfakeDetector
from utils.dataset import get_transforms
from utils.face_cropper import extract_face_roi

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
CHECKPOINT_PATH = os.path.join("checkpoints", "best_dual_stream_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTION_THRESHOLD = 0.45

app = FastAPI(
    title="Dual-Stream Deepfake & AI-Generated Photo Detector API",
    description="Spatial (EfficientNet-B4) + Frequency (2D-FFT) Cross-Attention Forensic API",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

val_transform = get_transforms(split="val")

# ==========================================
# 2. LOAD MODEL & CHECKPOINT WEIGHTS
# ==========================================
print(f"[INFO] Initializing DualStreamDeepfakeDetector on {DEVICE}...")
model = DualStreamDeepfakeDetector(pretrained=False).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"[INFO] Successfully loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"[ERROR] Checkpoint NOT found at {CHECKPOINT_PATH}! Run train.py first.")

model.eval()


# ==========================================
# 3. TRI-VIEW FORENSIC REPORT GENERATOR
# ==========================================
def generate_tri_view_base64(
    face_pil: Image.Image,
    input_tensor: torch.Tensor,
    model_instance: torch.nn.Module,
    verdict: str,
    confidence: float,
) -> str:
    probe_resized = face_pil.resize((224, 224))
    probe_np = np.array(probe_resized)

    # 2D-FFT Spectrum
    gray_np = np.array(probe_resized.convert("L"), dtype=np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(gray_np))
    magnitude_spectrum = np.log1p(np.abs(f_shift))

    # Grad-CAM on last Conv2d layer of spatial_backbone
    activations = []
    gradients = []

    target_layer = None
    for m in model_instance.spatial_backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model_instance.zero_grad()
    logit = model_instance(input_tensor).squeeze()
    logit.backward()

    fh.remove()
    bh.remove()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BILINEAR)
    ) / 255.0

    # Render Tri-View Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
    fig.patch.set_facecolor("#0A0F1C")

    axes[0].imshow(probe_np)
    axes[0].set_title("1. Face ROI Probe (224x224)", color="#00F0FF", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(magnitude_spectrum, cmap="inferno")
    axes[1].set_title("2. 2D-FFT Log Spectrum", color="#00F0FF", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(probe_np)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("3. Grad-CAM Spatial Heatmap", color="#00F0FF", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    banner_color = "#FF3366" if verdict == "AI_GENERATED" else "#00E676"
    fig.suptitle(
        f"FORENSIC VERDICT: {verdict}  |  CONFIDENCE: {confidence:.2f}%",
        color=banner_color,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ==========================================
# 4. API ENDPOINTS & FRONTEND UI
# ==========================================
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_cyber_ui():
    return FileResponse(os.path.join("static", "index.html"))


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "Dual-Stream Deepfake & AI-Generated Photo Detector",
        "device": str(DEVICE),
        "checkpoint_loaded": os.path.exists(CHECKPOINT_PATH),
    }


@app.post("/api/v1/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    try:
        image_bytes = await file.read()
        raw_img = Image.open(io.BytesIO(image_bytes))
        format_detected = raw_img.format or "UNKNOWN"
        pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image file: {str(e)}")

    # 1. Extract portrait ROI
    face_crop, face_detected = extract_face_roi(pil_img, margin=0.25)

    # 2. Apply exact Albumentations val transform from utils/dataset.py[cite: 3]
    raw_np = np.array(face_crop.resize((224, 224)))
    input_tensor = val_transform(image=raw_np)["image"].unsqueeze(0).to(DEVICE)

    # 3. Forward pass
    with torch.no_grad():
        logit = model(input_tensor).squeeze(-1)
        fake_probability = torch.sigmoid(logit).item()

    is_deepfake = fake_probability >= DETECTION_THRESHOLD
    verdict = "AI_GENERATED" if is_deepfake else "AUTHENTIC"
    confidence = (fake_probability if is_deepfake else (1.0 - fake_probability)) * 100.0

    # 4. Generate Tri-View Visual Forensic Report
    try:
        forensic_b64 = generate_tri_view_base64(
            face_pil=face_crop,
            input_tensor=input_tensor.clone().detach().requires_grad_(True),
            model_instance=model,
            verdict=verdict,
            confidence=confidence,
        )
    except Exception as grad_err:
        print(f"[WARN] Tri-View generation fallback: {grad_err}")
        forensic_b64 = None

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "status": "success",
        "filename": file.filename,
        "format_detected": format_detected,
        "face_detected": face_detected,
        "verdict": verdict,
        "is_deepfake": is_deepfake,
        "confidence_percentage": round(confidence, 2),
        "raw_fake_probability": round(fake_probability, 4),
        "inference_time_ms": elapsed_ms,
        "forensic_report_base64": forensic_b64,
    }