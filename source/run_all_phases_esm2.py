"""
Run all training phases with ESM2 embeddings.

Training flow (single split, no cross-validation):
  Phase 1:
    HLA_ESM2.py → trains HLA model, saves encoder_P
    TCR_ESM2.py → loads encoder_P from HLA_ESM2, trains TCR model, saves encoder_P

  Phase 2:
    HLA_ESM2_2.py → loads encoder_P from TCR_ESM2, trains HLA model, saves encoder_P
    TCR_ESM2_2.py → loads encoder_P from HLA_ESM2_2, trains TCR model, saves encoder_P

Usage:
    cd /home/mclab/mjp/unifyimmun/source
    python run_all_phases_esm2.py
"""

import subprocess
import os
import time

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)


PYTHON_PATH = "/home/mjp/miniconda3/envs/unifyimmun/bin/python"

def run_phase(script_name, phase_name):
    """Run a training script and report timing."""
    print("\n" + "=" * 60)
    print(f"Starting {phase_name}...")
    print("=" * 60)
    start_time = time.time()

    result = subprocess.run(
        [PYTHON_PATH, script_name],
        cwd=_current_dir,
        capture_output=False,
    )

    elapsed = time.time() - start_time
    print(f"\n{phase_name} finished in {elapsed/60:.2f} minutes")
    print("-" * 60)

    if result.returncode != 0:
        print(f"Warning: {script_name} returned non-zero exit code: {result.returncode}")
        return False
    return True


def main():
    total_start = time.time()

    print("\n" + "#" * 60)
    print("# ESM2 Training Pipeline - Single Split (No Cross-validation)")
    print("#" * 60)
    print(f"Project root: {_project_root}")
    print(f"Scripts directory: {_current_dir}")

    # Create trained_model directories
    model_dirs = ["HLA_ESM2", "TCR_ESM2", "HLA_ESM2_2", "TCR_ESM2_2"]
    for dir_name in model_dirs:
        dir_path = os.path.join(_project_root, "trained_model", dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # Phase 1
    print("\n### Phase 1 ###")
    success = run_phase("HLA_ESM2.py", "HLA_ESM2 (Phase 1)")
    if not success:
        print("Error in HLA_ESM2.py, stopping pipeline")
        return

    success = run_phase("TCR_ESM2.py", "TCR_ESM2 (Phase 1)")
    if not success:
        print("Error in TCR_ESM2.py, stopping pipeline")
        return

    # Phase 2
    print("\n### Phase 2 ###")
    success = run_phase("HLA_ESM2_2.py", "HLA_ESM2_2 (Phase 2)")
    if not success:
        print("Error in HLA_ESM2_2.py, stopping pipeline")
        return

    success = run_phase("TCR_ESM2_2.py", "TCR_ESM2_2 (Phase 2)")
    if not success:
        print("Error in TCR_ESM2_2.py, stopping pipeline")
        return

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 60)
    print("# All Phases Completed Successfully!")
    print("#" * 60)
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.2f} minutes)")

    # Print summary of saved models
    print("\nSaved models:")
    for dir_name in model_dirs:
        dir_path = os.path.join(_project_root, "trained_model", dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"  {dir_name}: {len(files)} files")
            for f in sorted(files):
                print(f"    - {f}")


if __name__ == "__main__":
    main()