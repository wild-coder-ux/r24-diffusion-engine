#!/usr/bin/env python3
import subprocess
import os
import sys
import torch

def run_r24_xl_pipeline(prompt_string):
    print("==================================================================")
    print("R24 XL SUBPROCESS ENGINE: SEQUENTIAL DESKTOP ORCHESTRATION")
    print("==================================================================")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Step 1: SDXL Dual Text Embedding Generation Process
        print("\n[R24 CONTROL] Launching Stage 1: Isolated Text Conditioning...")
        subprocess.run([sys.executable, "stage1_text_xl.py", prompt_string], check=True)

        # Step 2: SDXL UNet 2.6B Parameter Denoising Process
        print("\n[R24 CONTROL] Launching Stage 2: Process-Insulated Attention Denoising...")
        subprocess.run([sys.executable, "stage2_unet_xl.py"], check=True)

        # Step 3: SDXL FP16-Fixed VAE Latent Pixel Decoding Process
        print("\n[R24 CONTROL] Launching Stage 3: Isolated Latent Pixel Decoding...")
        subprocess.run([sys.executable, "stage3_vae_xl.py"], check=True)

        print("\n==================================================================")
        print("SUCCESS: SDXL Pipeline fully completed via isolated lifecycles!")
        print("==================================================================")

    except subprocess.CalledProcessError as e:
        print("\n==================================================================")
        print(f"[ORCHESTRATOR ERROR] Pipeline failed during execution phase.")
        print(f"Command tracking failed on: {e.cmd}")
        print("==================================================================")
        sys.exit(1)

if __name__ == "__main__":
    # Your default sunset beach prompt from your successful experiment run!
    target_prompt = "a vibrant beach scene at sunset, cinematic lighting, 8k resolution"

    # Allows you to pass a custom prompt via terminal if you want to test others
    if len(sys.argv) > 1:
        target_prompt = sys.argv[1]

    run_r24_xl_pipeline(target_prompt)
