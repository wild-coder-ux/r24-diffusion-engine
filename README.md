# Low-VRAM SDXL Orchestrator

An intelligent, process-isolated infrastructure framework designed to execute Stable Diffusion XL (SDXL) 1024x1024 generation loops locally on resource-constrained hardware (under 2 GB VRAM). 

By decoupling the standard monolithic execution graph into sequential, isolated operating system lifecycles, this pipeline completely bypasses standard CUDA out-of-memory constraints without altering structural model weights or sacrificing visual fidelity.

## 🚀 Key Architectural Breakthrough
Standard deployment models fail on lower-end consumer hardware because they simultaneously crowd the GPU memory space with multiple dense text encoders, a 2.6-billion parameter UNet backbone, and heavy VAE decoders. 

This orchestrator breaks that mold through **Process-Insulated Lifecycle Isolation**:
* **Stage 1 (Text Conditioning):** Spawns an isolated thread to run dual text encoders, caches the text tokens to storage, and terminates completely to free 100% of its memory footprint.
* **Stage 2 (Denoising Matrix):** Boots up on a completely clear memory slate to process the heavy UNet attention loops. It utilizes host system RAM as a secure dynamic fallback safety net to swallow memory overflows safely before writing raw latents to disk and exiting.
* **Stage 3 (VAE Decoding):** Executes a final standalone process to seamlessly parse latents into high-fidelity, uncompressed images using an FP16-fixed VAE decoding block.

---

## 💻 System Configuration & Requirements

* **Operating System:** Linux (Fully tested on Pop!_OS / Ubuntu)
* **GPU (Hardware Boundary):** Any NVIDIA GPU with a minimum of **2.0 GB VRAM**
* **System RAM:** **16 GB** (Crucial as an overflow swap safety net for the UNet phase)
* **Software Prerequisites:** Python 3.10+, PyTorch (CUDA enabled), Diffusers, Transformers

---

## 📂 Core Repository Structure

* `r24_orchestrator_xl.py` - The main sequential pipeline orchestrator.
* `stage1_text_xl.py` - Isolated process handling textual token mapping.
* `stage2_unet_xl.py` - Process-isolated 2.6B parameter core diffusion generation loop.
* `stage3_vae_xl.py` - Isolated FP16-fixed pixel synthesis layer.
* `baseline_benchmark_xl_true.py` - Fair, local-cached monolithic test script used to empirically verify hardware threshold failures.

---

## 🛠️ Quick Start & Usage

To generate an image using the intelligent memory management pipeline, execute the main orchestrator script directly via terminal:

```bash
python3 r24_orchestrator_xl.py "a vibrant beach scene at sunset, cinematic lighting, 8k resolution"
