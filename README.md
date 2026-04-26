# TerraCrop — On-Orbit Crop Intelligence

Sentinel-2 → crop mask → 1.5 KB downlink  
Built for satellites that can't afford to send home every pixel.

---

## 1. What problem are you solving?

A crop insurance underwriter or agriculture coordinator needs to verify what was planted where (e.g., Meadow, Wheat, Corn) across large regions. Today, this requires waiting for Sentinel-2 imagery to downlink, queue for processing, and be reviewed by analysts — a **3–5 day delay**. This increases fraud risk and delays decisions.

**TerraCrop moves the decision onboard the satellite.**  
Instead of sending full imagery, it sends a compressed crop mask.

**Success = usable crop map delivered in minutes, not days.**

---

## 2. What did you build?

We built an end-to-end pipeline:

**Sentinel-2 → TerraMind segmentation → crop mask → autoencoder compression → downlink**

- Base model: **TerraMind-1.0-small (EO-pretrained encoder)**
- Head: Lightweight conv decoder (3 crops + background)
- Dataset: **PASTIS (Sentinel-2 time-series)**
- Preprocessing: temporal mean-collapse → 10-band → converted to 12-band TerraMind input (224×224)
- Fine-tuning:
  - Frozen encoder, trained segmentation head
  - 150 epochs, batch 32
  - AdamW, lr = 5e-4, weight decay = 1e-5
  - Weighted cross-entropy
- Compression: convolutional autoencoder → **384-d latent**



<img width="908" height="926" alt="Screenshot 2026-04-26 at 8 45 46 AM" src="https://github.com/user-attachments/assets/26b04f0a-b3e3-4c02-89a2-ae6173930b1e" />




**Why TerraMind?**  
Zero-shot baselines (SegFormer, UNet) achieve ~0.09 mIoU.  
EO pretraining boosts us to **0.544 mIoU (~6× gain)**.

---

## 3. How we measured it

We evaluated mIoU over foreground crop classes only: Meadow, Wheat, and Corn. Background is ignored for mIoU but included in the reconstructed mask. Baselines were zero-shot ImageNet segmentation models and a MostFrequent classifier, evaluated on the same filtered PASTIS validation setup. The full-pipeline score measures the reconstructed mask after segmentation plus compression/decompression, so the gap from TerraCrop alone is the measured compression loss.

| Method | Category | Params (M) | mIoU | Meadow IoU | Wheat IoU | Corn IoU | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MostFrequent | Baseline | — | 0.068 | 0.204 | 0.000 | 0.000 | 0.532 |
| SegFormer-B3 | Baseline | 44.6 | 0.091 | 0.171 | 0.036 | 0.066 | 0.351 |
| UNet-ResNet18 | Baseline | 14.4 | 0.069 | 0.132 | 0.076 | 0.000 | 0.303 |
| UNet-EfficientNetB4 | Baseline | 20.2 | 0.086 | 0.116 | 0.062 | 0.081 | 0.243 |
| **TerraCrop** | **TerraMind** | **23.0** | **0.544** | **0.530** | **0.516** | **0.586** | **0.855** |
| **TerraCrop + AE** | **Full pipeline** | **36.2** | **0.490** | **0.456** | **0.494** | **0.522** | **0.882** |



<img width="925" height="404" alt="Screenshot 2026-04-26 at 8 05 57 AM" src="https://github.com/user-attachments/assets/d1a8978e-b65e-4262-8835-f9dfd95d4d28" />



**Result:** TerraCrop improves mIoU from the best listed zero-shot baseline, `0.091`, to `0.544`. The full satellite-to-ground pipeline reaches `0.490` mIoU after compression and reconstruction, giving an AE compression loss of `0.054` mIoU.


## 4. What's the orbital-compute story?

**Model footprint**
- Segmentation: **116 MB**
- Autoencoder: **51 MB**
- Total: **167 MB (~43.8M params)**

**Inference + memory**
- Batch size: 1
- Fits in **<200 MB memory**
- Suitable for **FP32 deployment**

**Bandwidth savings**
- Raw Sentinel-2 tile: ~700 MB
- Output: **1.5 KB**
- Reduction: **~466,000× (~99.9998%)**

**Latency (estimate)**
- ~1.5s/tile on T4 (observed class)
- ~3–5s/tile expected on Jetson Orin Nano (TOPS scaling)

**Verdict:**  
Feasible on Jetson-class satellite payloads.  
Final latency + power not yet measured on hardware.

---

## 5. What doesn't work yet?

- Only **3 crop classes** (PASTIS has 18)
- Temporal signal discarded (mean pooling)
- No real **Jetson hardware benchmarks**

**Next steps (1 week)**
- Train full 18-class taxonomy  
- Add temporal attention (use crop phenology)  
- Benchmark on Jetson (latency + power)  
- Train on float 32, followed by 8 bit Quantization (int/float)
- Improve compression (VQ-VAE / entropy coding)  

---
## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit demo:

```bash
streamlit run app.py
```

Run CLI inference:

```bash
python infer.py
```

The CLI prompts for:

1. a PASTIS Sentinel-2 `.npy` sample,
2. a TerraMind segmentation checkpoint,
3. an autoencoder checkpoint,
4. an output path for the reconstructed mask.

Run benchmarking:

```bash
python src/bench.py
```

Run training:

```bash
python train.py
python src/encode/train.py
```

## Repo contents

```text
submissions/SEUT/
├── README.md
├── infer.py                  # CLI entry point for satellite → ground pipeline
├── train.py                  # TerraMind segmentation fine-tuning
├── output_mask.png           # sample output mask
└── src/
    ├── dataset.py            # PASTIS filtering/loading utilities
    ├── terramind.py          # TerraMind backbone + preprocessing
    ├── utils.py              # validation metrics
    ├── bench.py              # benchmark/evaluation script
    ├── graph.py              # plotting and reporting helpers
    ├── plotting.py           # visualizations
    └── encode/
        ├── model.py          # mask autoencoder
        └── train.py          # autoencoder training
```

## Team: SEUT (Satellite Edge Uplink Technology)

- Dhruv Nandigam — model training / evaluation / integration / everything :)
- Charvi Mahalakshmi Bayana — autoencoder compression pipeline / frontend / demo
- Meet Sangani - ideation / visualization / presentation / writeup

