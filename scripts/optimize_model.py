#!/usr/bin/env python3
import argparse
import os
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo


def optimize_and_push(input_path, repo_id, push_to_hub=True):
    print(f"Loading checkpoint from {input_path}...")
    # Load with mmap to save RAM if possible, though we need to load into memory to convert types
    checkpoint = torch.load(input_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Optimizing weights to FP16...")
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.is_floating_point():
                tensor = tensor.half()
            new_state_dict[key] = tensor.contiguous()
        else:
            new_state_dict[key] = tensor

    # Free original memory
    del checkpoint
    del state_dict

    output_path = "audiosr.safetensors"
    print(f"Saving to {output_path}...")
    save_file(new_state_dict, output_path)

    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Optimized model size: {file_size_gb:.2f} GB")

    if push_to_hub:
        print(f"Pushing to Hugging Face Hub: {repo_id}...")
        api = HfApi()

        # Ensure repo exists
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Note on repo creation: {e}")

        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo="audiosr.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )
        print("Upload complete!")
    else:
        print("Skipping upload.")


if __name__ == "__main__":
    optimize_and_push(
        "/Users/nicolasoulianov/synth/audiosr_basic/pytorch_model.bin",
        "oulianov/audio-sr",
        True,
    )
