#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import time
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_true_monolithic_xl_baseline():
    print("==================================================================")
    print("RUNNING THE TRUE, LOCAL-CACHED MONOLITHIC SDXL BASELINE")
    print("==================================================================")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    try:
        print("Loading local FP16-Fixed VAE component...")
        # Force the baseline to use the exact same cached fixed VAE
        local_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )

        print("Assembling complete monolithic SDXL from local cache...")
        # Bypasses new network downloads by loading components directly
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=local_vae,
            torch_dtype=torch.float16,
            local_files_only=True # STRICT RULE: Absolute zero network pulls allowed
        ).to(DEVICE)

        prompt = "a vibrant beach scene at sunset, cinematic lighting, 8k resolution"

        print("\nExecuting 20 inference steps on a single monolithic memory graph...")
        image = pipe(prompt, num_inference_steps=20).images[0]
        image.save("baseline_xl_beach_true.png")

        end_time = time.time()
        print("\n==================================================================")
        print("SUCCESS: True Baseline Completed without Network Overhead")
        print("==================================================================")
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Monolithic Peak VRAM Consumption: {peak_vram:.2f} MB")

    except RuntimeError as cuda_error:
        print("\n==================================================================")
        print("[TRUE CRITICAL FAILURE] THE MONOLITHIC GRAPH HAS CRASHED THE GPU!")
        print("==================================================================")
        print(f"Error Message: {cuda_error}")
        print("==================================================================")

        # Capture the exact hard limit memory ceiling before the system panicked
        if torch.cuda.is_available():
            peak_vram_at_crash = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Peak VRAM Reached Immediately Before Crash: {peak_vram_at_crash:.2f} MB")
        print("==================================================================")
        sys.exit(1)

if __name__ == "__main__":
    run_true_monolithic_xl_baseline()
