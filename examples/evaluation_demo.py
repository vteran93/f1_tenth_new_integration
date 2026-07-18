#!/usr/bin/env python3
"""
Example script showing different ways to use the SAC evaluation with video recording.
This script demonstrates the various feature flag combinations.
"""

import os
import subprocess
import sys


def run_evaluation(train=False, record_video=True, evaluate_all_checkpoints=False, description=""):
    """Run the SAC evaluation with specified parameters"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"train={train}, record_video={record_video}, evaluate_all_checkpoints={evaluate_all_checkpoints}")
    print(f"{'='*80}")

    # Create a temporary script with the desired parameters
    script_content = f"""
# Auto-generated evaluation script
import sys
sys.path.append('.')

# Configuration
train = {train}
record_video = {record_video}
evaluate_all_checkpoints = {evaluate_all_checkpoints}

# Import and run the main script logic
exec(open('sac_example.py').read())
"""

    with open('temp_eval.py', 'w') as f:
        f.write(script_content)

    try:
        result = subprocess.run([sys.executable, 'temp_eval.py'],
                                capture_output=False, text=True)
        if result.returncode != 0:
            print(f"Error running evaluation: {result.returncode}")
    finally:
        # Clean up temporary file
        if os.path.exists('temp_eval.py'):
            os.remove('temp_eval.py')


def main():
    """Demonstrate different evaluation modes"""

    # Check if models directory exists
    if not os.path.exists('models'):
        print("Error: 'models' directory not found. Please ensure you have trained models available.")
        return

    print("SAC Evaluation Demo")
    print("This script demonstrates different evaluation modes.")
    print("\nAvailable modes:")
    print("1. Single checkpoint with video")
    print("2. Single checkpoint without video (live display)")
    print("3. All checkpoints with video")
    print("4. All checkpoints without video")
    print("5. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                run_evaluation(
                    train=False,
                    record_video=True,
                    evaluate_all_checkpoints=False,
                    description="Single checkpoint with video recording"
                )
            elif choice == '2':
                run_evaluation(
                    train=False,
                    record_video=False,
                    evaluate_all_checkpoints=False,
                    description="Single checkpoint with live display"
                )
            elif choice == '3':
                run_evaluation(
                    train=False,
                    record_video=True,
                    evaluate_all_checkpoints=True,
                    description="All checkpoints with video recording"
                )
            elif choice == '4':
                run_evaluation(
                    train=False,
                    record_video=False,
                    evaluate_all_checkpoints=True,
                    description="All checkpoints with live display"
                )
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
