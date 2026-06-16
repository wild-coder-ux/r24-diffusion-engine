# R24 — Process-Isolated Diffusion Inference Engine

An infrastructure framework for running large Stable Diffusion models on resource-constrained consumer hardware via **subprocess lifecycle isolation** — no weight modification, no quality loss.

Originally validated on SDXL. Extended tonight to **SD3 Medium** and **SD3.5 Large (8B)**.

---

## 🏆 Benchmark Results

| Model | Params | Architecture | Monolith Peak | Monolith Result | R24 Peak | R24 Result | Output |
|---|---|---|---|---|---|---|---|
| SDXL 1.0 | 3.5B | UNet | 7.13 GB | ❌ CRASH | ~6 GB | ✅ 1024px | Published |
| SD3 Medium | 2B | MMDiT | ~5 GB | ❌ CRASH | 1.30 GB | ✅ 512px | New |
| **SD3.5 Large** | **8B** | **MMDiT** | **7.30 GB** | **❌ CRASH** | **4.99 GB** | **✅ 1024px** | **New** |

**Hardware:** NVIDIA RTX 4070 Laptop, 7.62 GB VRAM, Pop!_OS Linux

---

## 🚀 Core Breakthrough

Standard pipelines load all components simultaneously into one CUDA context:

```
Text Encoders + Transformer/UNet + VAE → all in VRAM at once → CRASH
```

R24 maps each component to an independent OS process lifecycle:

```
[Stage 1: Text] → saves to disk → process exits → VRAM wiped
[Stage 2: Transformer] → fresh VRAM slate → saves latents → exits
[Stage 3: VAE] → fresh VRAM slate → final image → exits
```

The OS hard-guarantees VRAM release between stages. No CUDA context survives process termination.

---

## 📊 SD3.5 Large — Key Result

SD3.5 Large is an **8 billion parameter MMDiT** requiring ~16 GB VRAM monolithically.

```
Monolith:  7.30 GB → OutOfMemoryError at 49.6s
R24:       4.99 GB → 1024×1024 image in 138s
GPU:       7.62 GB total
```

The 8B transformer runs in **4-bit NF4 quantization** (~4.45 GB), leaving headroom for denoising activations. T5-XXL text encoder runs on CPU RAM. CLIP encoders run on GPU sequentially and are deleted before Stage 2 loads.

---

## 📂 Repository Structure

### SDXL Pipeline
- `r24_orchestrator_xl.py` — master orchestrator
- `stage1_text_xl.py` — dual CLIP encoder (ViT-L + OpenCLIP ViT-bigG)
- `stage2_unet_xl.py` — 2.6B UNet denoising loop
- `stage3_vae_xl.py` — FP16-fixed VAE decode
- `baseline_benchmark_xl.py` — monolith crash baseline

### SD3 Medium Pipeline
- `run_sd3.py` — orchestrator
- `stage1_sd3_text.py` — CLIP-L + CLIP-G (GPU) + T5-XXL (CPU)
- `stage2_sd3_mmdit.py` — 2B MMDiT, int8 quantization
- `stage3_sd3_vae.py` — VAE decode

### SD3.5 Large Pipeline
- `run_sd35.py` — orchestrator
- `stage1_sd35_text.py` — CLIP-L + CLIP-G (GPU) + T5-XXL (CPU)
- `stage2_sd35_mmdit.py` — 8B MMDiT, **4-bit NF4 quantization**
- `stage3_sd35_vae.py` — VAE decode
- `sd35_monolith.py` — crash baseline

---

## 💻 Requirements

- OS: Linux (tested on Pop!_OS / Ubuntu)
- GPU: NVIDIA with ≥ 8 GB VRAM (for SD3.5 Large)
- RAM: 32 GB recommended (T5-XXL loads on CPU)
- Python: 3.10+
- CUDA: 12.x
- `pip install diffusers transformers bitsandbytes torch torchvision`

---

## 🔬 Research Context

Named after Indian mathematicians whose work on sequences and efficient computation inspired the framework:
**Ramanujan, Aryabhata, Madhava, Pingala, Virahanka (R24)**

The subprocess serialization principle — do one thing completely, release all resources, then proceed — mirrors the Pingala binary enumeration approach: strict sequential state, no overlap, no waste.

Related: [R24 Text Retrieval (Zenodo DOI: 10.5281/zenodo.20600066)](https://zenodo.org/records/20600066)

---

## 📸 Sample Output

Prompt: *"A cinematic shot of a futuristic city"*
Model: SD3.5 Large (8B) via R24 — 1024×1024 — Peak VRAM: 4.99 GB

*(see sd35_output.png)*
