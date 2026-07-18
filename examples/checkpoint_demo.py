#!/usr/bin/env python3

"""
Demo script showing how the incremental saving works in sac_example.py

This script demonstrates:
1. How checkpoints are saved every 100,000 steps
2. How training can be resumed from the last checkpoint
3. Directory structure for checkpoints and models
"""

import os
import json


def show_checkpoint_structure():
    """Show the expected directory structure for checkpoints"""
    print("Expected directory structure after running training:")
    print(".")
    print("├── models/")
    print("│   ├── sac_checkpoint_100000.zip")
    print("│   ├── sac_checkpoint_200000.zip")
    print("│   ├── sac_checkpoint_300000.zip")
    print("│   ├── ...")
    print("│   └── sac_final_<run_id>.zip")
    print("├── checkpoints/")
    print("│   └── latest_checkpoint.json")
    print("└── runs/")
    print("    └── <run_id>/")
    print("        └── tensorboard logs")
    print()


def show_checkpoint_content():
    """Show example content of a checkpoint file"""
    print("Example content of 'checkpoints/latest_checkpoint.json':")
    example_checkpoint = {
        "timesteps_completed": 300000,
        "model_path": "models/sac_checkpoint_300000.zip",
        "run_id": "abc123xyz"
    }
    print(json.dumps(example_checkpoint, indent=2))
    print()


def check_existing_checkpoints():
    """Check if there are any existing checkpoints"""
    checkpoint_path = "checkpoints/latest_checkpoint.json"
    models_dir = "models"

    print("Checking for existing checkpoints...")

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"✓ Found checkpoint: {checkpoint}")

        if os.path.exists(checkpoint["model_path"]):
            print(f"✓ Model file exists: {checkpoint['model_path']}")
        else:
            print(f"✗ Model file missing: {checkpoint['model_path']}")
    else:
        print("✗ No checkpoint file found")

        # Check for individual model files
        if os.path.exists(models_dir):
            import glob
            model_files = glob.glob(os.path.join(models_dir, "sac_checkpoint_*.zip"))
            if model_files:
                print(f"✓ Found {len(model_files)} checkpoint model(s):")
                for model in sorted(model_files):
                    print(f"  - {model}")
            else:
                print("✗ No checkpoint models found")
        else:
            print(f"✗ Models directory '{models_dir}' doesn't exist")


if __name__ == "__main__":
    print("=== SAC Training Checkpoint System Demo ===\n")

    show_checkpoint_structure()
    show_checkpoint_content()
    check_existing_checkpoints()

    print("\nHow to use:")
    print("1. Run 'python sac_example.py' to start training")
    print("2. Training will save checkpoints every 100,000 steps")
    print("3. If interrupted, simply run 'python sac_example.py' again")
    print("4. The script will automatically resume from the last checkpoint")
    print("\nKey features:")
    print("- Automatic checkpoint detection and resuming")
    print("- WandB run continuity (if run_id is preserved)")
    print("- Incremental model saving every 100,000 steps")
    print("- Final model saved at the end of training")
