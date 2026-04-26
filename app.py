"""
TerraCrop · SEUT · TM2Space 2026
streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import streamlit.components.v1 as components
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (app.py sits next to both checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
SEG_CHECKPOINT = Path(__file__).parent / "best_model.pt"
AE_CHECKPOINT = Path(__file__).parent / "ae_best_9096.pt"

SEG_MIOU = "0.54"
NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128

st.set_page_config(page_title="TerraCrop", page_icon="🛰", layout="wide")

# ═══════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f5f5f5; color: #111827; }
#MainMenu, footer, header { visibility: hidden; }
section.main > div[data-testid="stVerticalBlock"],
.block-container {
    width: min(100%, 1240px) !important;
    max-width: 1240px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 20px 24px 32px !important;
}
@media (max-width: 760px) {
    section.main > div[data-testid="stVerticalBlock"],
    .block-container { padding: 14px 12px 24px !important; }
}
div[data-testid="column"],
div[data-testid="column"] > div,
div[data-testid="column"] [data-testid="stVerticalBlock"] {
    min-width: 0 !important; max-width: 100% !important;
}
div[data-testid="stHorizontalBlock"] { gap: 10px !important; max-width: 100% !important; }
div[data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] > div,
div[data-testid="stImage"],
div[data-testid="stImage"] img,
svg { max-width: 100% !important; }
div[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px 14px;
}
div[data-testid="metric-container"] label {
    font-size: 10px !important; text-transform: uppercase; letter-spacing: .07em;
    color: #9ca3af !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 19px !important; font-weight: 500 !important; color: #111827 !important;
    font-family: 'JetBrains Mono', monospace !important; letter-spacing: -.02em; margin-top: 6px;
}
div[data-testid="stMetricDelta"] { display: none; }
div[data-testid="stFileUploader"] {
    background: #ffffff !important; border: 1px solid #e5e7eb !important;
    border-radius: 10px !important; padding: 12px !important; max-width: 100% !important;
}
div[data-testid="stFileUploader"] label { font-size: 11px !important; color: #6b7280 !important; font-weight: 500 !important; }
div[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #d1d5db !important; background: #fafafa !important;
    border-radius: 8px !important; padding: 16px 12px !important;
}
div[data-testid="stFileUploaderDropzone"] button {
    background: #111827 !important; color: #ffffff !important; border: none !important;
    border-radius: 8px !important; font-size: 12px !important; font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button {
    background: #111827 !important; color: #ffffff !important; border: none !important;
    border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 11px 16px !important; width: 100% !important; font-family: 'Inter', sans-serif !important;
    cursor: pointer !important; margin-top: 0; transition: background 0.15s;
}
.stButton > button:hover { background: #1f2937 !important; border: none !important; color: #ffffff !important; }
.stButton > button:focus, .stButton > button:active {
    border: none !important; box-shadow: none !important; color: #ffffff !important; background: #1f2937 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 3px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #9ca3af; font-size: 12px;
    font-weight: 500; border-radius: 6px; padding: 6px 16px; font-family: 'Inter', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #111827 !important; box-shadow: 0 1px 2px rgba(0,0,0,.04) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 12px !important; }
div[data-testid="stSelectbox"] label { display: none; }
div[data-testid="stSelectbox"] > div > div {
    border: 1px solid #e5e7eb !important; border-radius: 8px !important;
    background: #ffffff !important; font-size: 13px !important;
    font-family: 'Inter', sans-serif !important; min-height: 38px !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #d1d5db; }
.tc-label {
    font-size: 10px; font-weight: 500; letter-spacing: .08em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 10px; margin-top: 0;
}
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }
[data-testid="stHorizontalBlock"] { gap: 10px !important; }
[data-testid="column"] [data-testid="stVerticalBlock"] { gap: 0.55rem !important; }
[data-testid="stImage"] { padding: 0; margin: 0; }
[data-testid="stImage"] img { border-radius: 6px; image-rendering: pixelated; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════


class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_grid = PATCH_GRID
        self.output_size = LABEL_SIZE
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, NUM_CLASSES, kernel_size=1),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        g = self.patch_grid
        x = tokens.permute(0, 2, 1).reshape(B, D, g, g)
        x = self.decoder(x)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class TerraMindSegmenter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.seg_head = SegmentationHead()

    def forward(self, images):
        from src.terramind import preprocess

        x = preprocess(images)
        tokens = self.backbone({"S2L2A": x})
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        return self.seg_head(tokens)


class MaskEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 8 * 8, latent_dim),
        )

    def forward(self, mask):
        x = F.one_hot(mask, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        return self.net(x)


class MaskDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256 * 8 * 8), nn.Dropout(0.3))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, NUM_CLASSES, 2, stride=2),
        )

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 256, 8, 8))


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — only runs once per session)
# ═══════════════════════════════════════════════════════════════════


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── TerraMind segmenter ───────────────────────────────────────
    from src.terramind import get_model

    backbone = get_model(variant="small")
    seg_model = TerraMindSegmenter(backbone).to(device)
    seg_ckpt = torch.load(SEG_CHECKPOINT, map_location=device)
    seg_model.load_state_dict(seg_ckpt.get("model_state_dict", seg_ckpt))
    seg_model.eval()

    # ── Autoencoder ───────────────────────────────────────────────
    ae_ckpt = torch.load(AE_CHECKPOINT, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
    latent_dim = (
        ae_state["encoder.net.14.weight"].shape[0]
        if "encoder.net.14.weight" in ae_state
        else 384
    )

    encoder = MaskEncoder(latent_dim).to(device)
    decoder = MaskDecoder(latent_dim).to(device)
    encoder.load_state_dict(
        {
            k.replace("encoder.", ""): v
            for k, v in ae_state.items()
            if k.startswith("encoder.")
        }
    )
    decoder.load_state_dict(
        {
            k.replace("decoder.", ""): v
            for k, v in ae_state.items()
            if k.startswith("decoder.")
        }
    )
    encoder.eval()
    decoder.eval()

    return seg_model, encoder, decoder, device, latent_dim


@st.cache_data(max_entries=1)
def get_train_stats():
    """Compute training-split normalisation stats (cached between uploads)."""
    from src.dataset import filter_patches, split_patches, compute_dataset_stats

    PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
    filtered = filter_patches(
        PASTIS_ROOT, [1, 2, 3], require_all=False, min_pixel_fraction=0.05
    )
    train_meta, _, _ = split_patches(filtered)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)
    return mean, std  # (10,) numpy arrays


# ═══════════════════════════════════════════════════════════════════
# NPY PROCESSING
# ═══════════════════════════════════════════════════════════════════


def load_npy(file_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Parse uploaded .npy bytes → (normalised tensor, raw_s2 for display).

    Returns:
        tensor:  (1, 10, 128, 128) float32 — model input (z-scored)
        raw_s2:  (10, H, W)        float32 — [0,1] for false-colour display
    """
    import io

    arr = np.load(io.BytesIO(file_bytes)).astype(np.float32)

    # Collapse time dimension
    if arr.ndim == 5:  # (B, T, C, H, W)
        arr = arr.mean(axis=1)
    elif arr.ndim == 4:  # (T, C, H, W)
        arr = arr.mean(axis=0)[None]
    elif arr.ndim == 3:  # (C, H, W)
        arr = arr[None]

    if arr.shape[1] != 10:
        raise ValueError(f"Expected 10 S2 bands, got {arr.shape[1]}")

    # Raw [0,1] for display
    raw_s2 = (arr[0] / 10000.0).clip(0, 1)

    # Normalise — same as PASTISCropDataset
    mean, std = get_train_stats()
    norm = arr / 10000.0
    norm = (norm - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-6)
    return torch.from_numpy(norm), raw_s2


def s2_to_rgb(raw_s2: np.ndarray) -> np.ndarray:
    """(10, H, W) [0,1] → (H, W, 3) uint8  NIR-Red-Green false colour."""
    rgb = np.stack([raw_s2[6], raw_s2[2], raw_s2[1]], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

COLORS = {0: "#f9fafb", 1: "#16a34a", 2: "#d97706", 3: "#ea580c"}
CLASSES = {0: "Background", 1: "Meadow", 2: "Wheat", 3: "Corn"}


def to_rgba(m: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    for cid, hc in COLORS.items():
        rgba[m == cid] = [int(hc[1:3], 16), int(hc[3:5], 16), int(hc[5:7], 16), 255]
    return rgba


@torch.no_grad()
def run_pipeline(
    seg_model, encoder, decoder, device, latent_dim, tensor: torch.Tensor
) -> dict:
    """
    Full pipeline:
        tensor (1,10,128,128) → seg logits → seg mask
                              → AE encoder → z (bytes)
                              → AE decoder → recon mask

    Returns a result dict with masks + telemetry.
    """
    tensor = tensor.to(device)

    # Step 1 — segmentation
    t0 = time.time()
    logits = seg_model(tensor)  # (1, 4, 128, 128)
    mask = logits.argmax(dim=1)  # (1, 128, 128) int64
    seg_ms = (time.time() - t0) * 1000

    # Step 2 — compress
    t1 = time.time()
    z = encoder(mask)  # (1, latent_dim)
    compressed_bytes = z.cpu().numpy().tobytes()
    enc_ms = (time.time() - t1) * 1000

    # Step 3 — decompress
    t2 = time.time()
    z_in = (
        torch.tensor(np.frombuffer(compressed_bytes, dtype=np.float32).copy())
        .reshape(1, latent_dim)
        .to(device)
    )
    recon_logits = decoder(z_in)  # (1, 4, 128, 128)
    recon_mask = recon_logits.argmax(dim=1)  # (1, 128, 128)
    dec_ms = (time.time() - t2) * 1000

    seg_np = mask.squeeze(0).cpu().numpy()
    recon_np = recon_mask.squeeze(0).cpu().numpy()

    fg = seg_np > 0
    fg_acc = float((recon_np[fg] == seg_np[fg]).mean()) if fg.sum() > 0 else 1.0
    px_acc = float((recon_np == seg_np).mean())

    raw_bytes = int(mask.numel())  # one int64 per pixel = 16 384 values

    return dict(
        seg_mask=seg_np,
        recon_mask=recon_np,
        raw=raw_bytes,
        comp=len(compressed_bytes),
        ratio=round(raw_bytes / len(compressed_bytes), 1),
        fg_acc=fg_acc,
        px_acc=px_acc,
        seg_ms=seg_ms,
        enc_ms=enc_ms,
        dec_ms=dec_ms,
    )


# ═══════════════════════════════════════════════════════════════════
# SESSION / LOG HELPERS
# ═══════════════════════════════════════════════════════════════════

TAG_C = {
    "sys": ("#f3f4f6", "#6b7280"),
    "infer": ("#dcfce7", "#15803d"),
    "compress": ("#fff7ed", "#c2410c"),
    "transmit": ("#dbeafe", "#1d4ed8"),
    "success": ("#dcfce7", "#15803d"),
    "error": ("#fef2f2", "#b91c1c"),
}

if "logs" not in st.session_state:
    st.session_state.logs = []
if "result" not in st.session_state:
    st.session_state.result = None
if "raw_s2" not in st.session_state:
    st.session_state.raw_s2 = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


def add_log(tag, msg):
    st.session_state.logs.append(
        {"t": time.strftime("%H:%M:%S"), "tag": tag, "msg": msg}
    )


# Boot logs (only once)
if not st.session_state.logs:
    add_log("sys", "TerraCrop v1.0 · SEUT · TM2Space 2026")
    add_log("sys", "TerraMind-S + MaskAutoencoder loaded")
    add_log("sys", "PASTIS-HD · 2,374 patches · 4 crop classes")


# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS (runs once, shows spinner)
# ═══════════════════════════════════════════════════════════════════

with st.spinner("Loading TerraMind + AE models…"):
    seg_model, encoder, decoder, device, latent_dim = load_models()

r = st.session_state.result

# ═══════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    f"""
<div style="height:56px;padding:0 24px;display:flex;align-items:center;
            justify-content:space-between;border-bottom:1px solid #e5e7eb;
            background:#ffffff;position:sticky;top:0;z-index:999;">
  <div style="display:flex;align-items:center;gap:10px;">
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <circle cx="9" cy="9" r="3" fill="#111827"/>
      <circle cx="9" cy="9" r="7" stroke="#111827" stroke-width="1" fill="none"/>
      <circle cx="14" cy="4" r="1.5" fill="#16a34a"/>
    </svg>
    <span style="font-size:14px;font-weight:500;letter-spacing:-.01em;color:#111827;">TerraCrop</span>
    <div style="width:1px;height:14px;background:#e5e7eb;margin:0 6px;"></div>
    <span style="font-size:12px;color:#9ca3af;">On-orbit crop intelligence · SEUT · TM2Space 2026</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:11px;color:#9ca3af;font-family:'JetBrains Mono',monospace;">
      orbit 14,287 · {time.strftime("%H:%M:%S")} UTC
    </span>
    <div style="display:flex;align-items:center;gap:6px;background:#f0fdf4;
                border:1px solid #bbf7d0;border-radius:20px;padding:4px 11px;">
      <div style="width:6px;height:6px;border-radius:50%;background:#16a34a;"></div>
      <span style="font-size:11px;color:#15803d;font-weight:500;">Link nominal</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# METRICS ROW
# ═══════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5, gap="small")
with m1:
    st.metric("Raw mask", f"{r['raw']:,} B" if r else "16,384 B")
with m2:
    st.metric("Compressed", f"{r['comp']} B" if r else "—")
with m3:
    st.metric("Ratio", f"{r['ratio']}×" if r else "—")
with m4:
    st.metric("Fg accuracy", f"{r['fg_acc'] * 100:.2f}%" if r else "—")
with m5:
    st.metric("Seg mIoU", SEG_MIOU)

st.markdown(
    "<div style='height:1px;background:#e5e7eb;margin:20px 0;'></div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# THREE COLUMNS
# ═══════════════════════════════════════════════════════════════════
cL, cM, cR = st.columns([1, 1, 1], gap="medium")

# ── LEFT: UPLOAD ─────────────────────────────────────────────────
with cL:
    st.markdown('<p class="tc-label">Input image</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a PASTIS S2 .npy file",
        type=["npy"],
        label_visibility="collapsed",
    )

    if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
        with st.spinner("Running TerraMind → AE pipeline…"):
            try:
                tensor, raw_s2 = load_npy(uploaded.read())
                res = run_pipeline(
                    seg_model, encoder, decoder, device, latent_dim, tensor
                )

                st.session_state.raw_s2 = raw_s2
                st.session_state.result = res
                st.session_state.uploaded_name = uploaded.name

                add_log("infer", f"{uploaded.name} · segmented 128×128")
                add_log(
                    "compress",
                    f"{res['raw']:,} → {res['comp']} B · {res['enc_ms']:.0f} ms",
                )
                add_log("transmit", f"Payload {res['comp']} B · S-band")
                add_log("success", f"Reconstructed · fg acc {res['fg_acc'] * 100:.2f}%")
                r = res
                st.rerun()
            except Exception as e:
                add_log("error", str(e))
                st.error(f"Error: {e}")

    EMPTY = (
        '<div style="height:240px;border:1.5px dashed #e5e7eb;border-radius:10px;'
        "display:flex;align-items:center;justify-content:center;color:#d1d5db;"
        'font-size:12px;background:#fafafa;">{}</div>'
    )

    if st.session_state.raw_s2 is not None:
        # S2 false-colour composite
        rgb = s2_to_rgb(st.session_state.raw_s2)
        st.markdown(
            '<div style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#f9fafb;">',
            unsafe_allow_html=True,
        )
        st.image(rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style=\"font-size:10px;color:#9ca3af;font-family:'JetBrains Mono',monospace;"
            f'margin:8px 0 0;line-height:1.6;">{st.session_state.uploaded_name}<br/>'
            f"NIR · Red · Green false-colour composite</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(EMPTY.format("Upload a .npy file to begin"), unsafe_allow_html=True)

# ── MIDDLE: SEGMENTATION / RECONSTRUCTION ────────────────────────
with cM:
    st.markdown('<p class="tc-label">Segmentation</p>', unsafe_allow_html=True)

    EMPTY = (
        '<div style="height:240px;border:1.5px dashed #e5e7eb;border-radius:10px;'
        "display:flex;align-items:center;justify-content:center;color:#d1d5db;"
        'font-size:12px;background:#fafafa;">{}</div>'
    )

    ti, to = st.tabs(["Segmentation mask", "Reconstructed mask"])
    with ti:
        if r is not None:
            st.image(
                to_rgba(r["seg_mask"]), use_container_width=True, output_format="PNG"
            )
            st.markdown(
                f"<p style=\"font-size:10px;color:#9ca3af;font-family:'JetBrains Mono',monospace;margin:6px 0 0;\">"
                f"Direct TerraMind output · {r['seg_ms']:.0f} ms</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(EMPTY.format("Upload .npy to segment"), unsafe_allow_html=True)

    with to:
        if r is not None:
            st.image(
                to_rgba(r["recon_mask"]), use_container_width=True, output_format="PNG"
            )
            st.markdown(
                f"""
            <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;">
              <span style="background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;
                           padding:3px 9px;border-radius:5px;font-size:10px;font-weight:500;
                           font-family:'JetBrains Mono',monospace;">{r["comp"]} bytes</span>
              <span style="background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;
                           padding:3px 9px;border-radius:5px;font-size:10px;font-weight:500;
                           font-family:'JetBrains Mono',monospace;">{r["fg_acc"] * 100:.2f}% acc</span>
              <span style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
                           padding:3px 9px;border-radius:5px;font-size:10px;font-weight:500;
                           font-family:'JetBrains Mono',monospace;">{r["ratio"]}× compression</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(EMPTY.format("Awaiting transmission"), unsafe_allow_html=True)

    # Legend
    legend_items = ""
    for cid, name in CLASSES.items():
        c = COLORS[cid]
        border = "#e5e7eb" if cid == 0 else c
        legend_items += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                    background:#ffffff;border:1px solid #e5e7eb;border-radius:7px;">
          <div style="width:10px;height:10px;border-radius:2px;flex-shrink:0;
                      background:{c};border:1px solid {border};"></div>
          <span style="font-size:11px;font-weight:500;color:#374151;">{name}</span>
        </div>"""
    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:14px;">{
            legend_items
        }</div>',
        unsafe_allow_html=True,
    )

# ── RIGHT: TELEMETRY / MODEL / LOG ───────────────────────────────
with cR:

    def tc_label(t):
        st.markdown(f'<p class="tc-label">{t}</p>', unsafe_allow_html=True)

    def tc_table(rows):
        n = len(rows)
        inner = ""
        for i, (l, v) in enumerate(rows):
            border = "border-bottom:1px solid #f3f4f6;" if i < n - 1 else ""
            inner += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;{border}">
              <span style="font-size:11px;color:#6b7280;white-space:nowrap;">{l}</span>
              <span style="font-size:11px;font-weight:500;color:#111827;
                           font-family:'JetBrains Mono',monospace;text-align:right;
                           margin-left:8px;">{v}</span>
            </div>"""
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;'
            f'padding:2px 14px;margin-bottom:14px;">{inner}</div>',
            unsafe_allow_html=True,
        )

    tc_label("Telemetry")
    tc_table(
        [
            ("Raw mask", f"{r['raw']:,} B" if r else "16,384 B"),
            ("Compressed", f"{r['comp']} B" if r else "—"),
            ("Ratio", f"{r['ratio']}×" if r else "—"),
            ("Px accuracy", f"{r['px_acc'] * 100:.2f}%" if r else "—"),
            ("Fg accuracy", f"{r['fg_acc'] * 100:.2f}%" if r else "—"),
            ("Seg", f"{r['seg_ms']:.0f} ms" if r else "—"),
            ("Encode", f"{r['enc_ms']:.0f} ms" if r else "—"),
            ("Decode", f"{r['dec_ms']:.0f} ms" if r else "—"),
        ]
    )

    tc_label("Model")
    tc_table(
        [
            ("Backbone", "TerraMind-S"),
            ("Dataset", "PASTIS-HD"),
            ("Latent dim", f"{latent_dim}d"),
            ("Compressed", f"{latent_dim * 4} B"),
            ("Seg mIoU", SEG_MIOU),
            ("Device", str(device)),
        ]
    )

    tc_label("Log")
    log_rows = ""
    for e in reversed(st.session_state.logs[-10:]):
        bg, fg = TAG_C.get(e["tag"], ("#f3f4f6", "#6b7280"))
        log_rows += f"""
        <div style="display:flex;gap:8px;padding:6px 0;
                    border-bottom:1px solid #f9fafb;align-items:flex-start;">
          <span style="font-size:9px;color:#d1d5db;min-width:46px;flex-shrink:0;
                       font-family:'JetBrains Mono',monospace;padding-top:2px;">{e["t"]}</span>
          <span style="font-size:8px;font-weight:600;padding:2px 6px;border-radius:3px;
                       background:{bg};color:{fg};min-width:54px;text-align:center;
                       text-transform:uppercase;letter-spacing:.05em;flex-shrink:0;">{e["tag"]}</span>
          <span style="font-size:10px;color:#6b7280;line-height:1.5;">{e["msg"]}</span>
        </div>"""
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;'
        f'padding:4px 12px;max-height:180px;overflow-y:auto;">{log_rows}</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════
# PIPELINE STRIP
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='height:1px;background:#e5e7eb;margin:24px 0 16px;'></div>",
    unsafe_allow_html=True,
)
st.markdown('<p class="tc-label">Pipeline</p>', unsafe_allow_html=True)

stages = [
    ("Capture", "Sentinel-2A L1C", "128×128"),
    ("Encode", "TerraMind-S", "ViT"),
    ("Segment", "Conv decoder", f"mIoU {SEG_MIOU}"),
    ("Compress", "Autoencoder", f"{latent_dim * 4} B"),
    ("Downlink", "S-band 64kbps", "~24 ms"),
    ("Reconstruct", "Ground decoder", f"{r['fg_acc'] * 100:.2f}%" if r else "—"),
]
for col, (i, (label, sub, val)) in zip(
    st.columns(len(stages), gap="small"), enumerate(stages)
):
    last = i == len(stages) - 1
    bg = "#f0fdf4" if last else "#ffffff"
    border = "#bbf7d0" if last else "#e5e7eb"
    valcol = "#15803d" if last else "#374151"
    with col:
        st.markdown(
            f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:12px 10px;text-align:center;">
          <div style="font-size:11px;font-weight:500;color:#111827;margin-bottom:3px;">{label}</div>
          <div style="font-size:9px;color:#9ca3af;margin-bottom:6px;">{sub}</div>
          <div style="font-size:10px;font-weight:500;color:{valcol};
                      font-family:'JetBrains Mono',monospace;">{val}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='height:1px;background:#e5e7eb;margin:28px 0 18px;'></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tc-label" style="text-align:center;">Benchmark results</p>',
    unsafe_allow_html=True,
)

benchmark_rows = [
    ("MostFrequent", "Baseline", "—", "0.068", "0.204", "0.000", "0.000", "0.532"),
    ("SegFormer-B3", "Baseline", "44.6", "0.091", "0.171", "0.036", "0.066", "0.351"),
    ("UNet-ResNet18", "Baseline", "14.4", "0.069", "0.132", "0.076", "0.000", "0.303"),
    (
        "UNet-EfficientNetB4",
        "Baseline",
        "20.2",
        "0.086",
        "0.116",
        "0.062",
        "0.081",
        "0.243",
    ),
    (
        "TerraCrop (ours)",
        "TerraMind",
        "23.0",
        "0.544",
        "0.530",
        "0.516",
        "0.586",
        "0.855",
    ),
    (
        "TerraCrop + AE (full pipeline)",
        "Pipeline",
        "36.2",
        "0.490",
        "0.456",
        "0.494",
        "0.522",
        "0.882",
    ),
]
bar_rows = [
    ("MostFrequent", 0.068, "base"),
    ("SegFormer-B3", 0.091, "base"),
    ("UNet-ResNet18", 0.069, "base"),
    ("UNet-EfficientNetB4", 0.086, "base"),
    ("TerraCrop (ours)", 0.544, "tm"),
    ("Pipeline + AE (ours)", 0.490, "pipe"),
]

body = ""
for model_name, category, params, miou, meadow, wheat, corn, acc in benchmark_rows:
    if category == "TerraMind":
        tr_class, tag_class, model_html = (
            "bench-highlight-tm",
            "bench-tag-tm",
            f"<strong>{model_name}</strong>",
        )
    elif category == "Pipeline":
        tr_class, tag_class, model_html = (
            "bench-highlight-pipe",
            "bench-tag-pipe",
            f"<strong>{model_name}</strong>",
        )
    else:
        tr_class, tag_class, model_html = "", "bench-tag-base", model_name
    vc = " class='bench-best'" if category in {"TerraMind", "Pipeline"} else ""
    body += f"<tr class='{tr_class}'><td>{model_html}</td><td><span class='bench-tag {
        tag_class
    }'>{category}</span></td><td>{params}</td><td{vc}>{miou}</td><td{vc}>{
        meadow
    }</td><td{vc}>{wheat}</td><td{vc}>{corn}</td><td{vc}>{acc}</td></tr>"

bars = ""
for label, value, kind in bar_rows:
    pct = value / 0.60 * 100
    bars += f"<div class='bench-bar-row'><div class='bench-bar-label'>{
        label
    }</div><div class='bench-bar-track'><div class='bench-bar-fill bench-bar-{
        kind
    }' style='width:{pct:.1f}%;'></div></div><div class='bench-bar-value'>{
        value:.3f}</div></div>"

components.html(
    f"""
<style>
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,-apple-system,sans-serif;background:transparent;color:#111827}}
.bench-wrap{{max-width:980px;margin:0 auto 8px;background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:18px;box-shadow:0 1px 2px rgba(0,0,0,.025)}}
.bench-title{{font-size:15px;font-weight:600;color:#111827;margin:0 0 4px;text-align:center}}
.bench-subtitle{{font-size:11px;color:#9ca3af;margin:0 0 16px;text-align:center}}
.bench-table-scroll{{overflow-x:auto}}.bench-table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:18px}}
.bench-table th{{padding:8px 10px;text-align:left;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#9ca3af;border-bottom:1px solid #e5e7eb;white-space:nowrap}}
.bench-table td{{padding:8px 10px;border-bottom:1px solid #f3f4f6;color:#374151;white-space:nowrap}}
.bench-table tr:last-child td{{border-bottom:none}}
.bench-tag{{display:inline-block;font-size:10px;font-weight:600;padding:2px 7px;border-radius:5px}}
.bench-tag-base{{background:#f3f4f6;color:#6b7280}}.bench-tag-tm{{background:#dcfce7;color:#15803d}}.bench-tag-pipe{{background:#dbeafe;color:#1d4ed8}}
.bench-highlight-tm{{background:#f0fdf4}}.bench-highlight-pipe{{background:#eff6ff}}.bench-best{{font-weight:600;color:#15803d!important}}
.bench-note{{font-size:11px;color:#6b7280;background:#f9fafb;border:1px solid #f3f4f6;border-radius:8px;padding:9px 11px;margin:0 0 16px}}
.bench-legend{{display:flex;justify-content:center;gap:16px;flex-wrap:wrap;font-size:11px;color:#6b7280;margin-bottom:14px}}
.bench-legend span{{display:flex;align-items:center;gap:6px}}
.bench-dot{{width:10px;height:10px;border-radius:2px;display:inline-block}}
.bench-dot-base{{background:#b4b2a9}}.bench-dot-tm{{background:#3b6d11}}.bench-dot-pipe{{background:#185fa5}}
.bench-bars{{display:grid;gap:9px}}
.bench-bar-row{{display:grid;grid-template-columns:170px minmax(120px,1fr) 44px;align-items:center;gap:10px}}
.bench-bar-label{{font-size:11px;color:#374151;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bench-bar-track{{height:18px;background:#f3f4f6;border-radius:999px;overflow:hidden}}
.bench-bar-fill{{height:100%;border-radius:999px}}
.bench-bar-base{{background:#b4b2a9}}.bench-bar-tm{{background:#3b6d11}}.bench-bar-pipe{{background:#185fa5}}
.bench-bar-value{{font-size:11px;color:#111827;font-family:'JetBrains Mono',monospace;text-align:right}}
</style>
<div class='bench-wrap'>
  <h3 class='bench-title'>TerraCrop Benchmark Results</h3>
  <p class='bench-subtitle'>Consolidated mIoU comparison across baselines, TerraMind, and the compressed full pipeline.</p>
  <div class='bench-table-scroll'><table class='bench-table'>
    <thead><tr><th>Model</th><th>Category</th><th>Params (M)</th><th>mIoU</th><th>Meadow</th><th>Wheat</th><th>Corn</th><th>Accuracy</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>
  <p class='bench-note'>AE compression loss = <strong>0.054 mIoU</strong> (0.544 → 0.490), while the full pipeline remains far above the zero-shot baselines.</p>
  <div class='bench-legend'>
    <span><i class='bench-dot bench-dot-base'></i>Baselines</span>
    <span><i class='bench-dot bench-dot-tm'></i>TerraCrop</span>
    <span><i class='bench-dot bench-dot-pipe'></i>Full pipeline + AE</span>
  </div>
  <div class='bench-bars' aria-label='mIoU bar chart'>{bars}</div>
</div>
""",
    height=610,
    scrolling=False,
)
