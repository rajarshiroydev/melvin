import re
import yaml
import importlib
import importlib.util
import subprocess
import argparse
import json
import sys
import asyncio
import os
import time
import pandas as pd
from datetime import datetime

from pathlib import Path

# Ensure these files exist in the same directory
from code_generator import generate_training_script_llm, fix_training_script_llm
from modality_detector import collect_dataset_metadata, detect_modality_llm


class MLEAgent:
    def __init__(self, competition_id, runs_base_dir, seed=42):
        self.competition_id = competition_id
        self.seed = seed

        # ------------------------------------------------------------------
        # Unique Run Directory Logic
        # ------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{competition_id}"

        # Resolve the full path to ensure it goes to melvin/runs/
        self.output_dir = Path(runs_base_dir).resolve() / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[INFO] Initialized new run. Logs and outputs will be saved to:\n       -> {self.output_dir}"
        )

        # ------------------------------------------------------------------
        # DYNAMIC MLEBENCH CACHE DIRECTORY LOCATOR  (FIX FOR LIGHTNING AI)
        # ------------------------------------------------------------------
        def _possible_cache_roots():
            home = Path.home()
            return [
                home / ".cache/mle-bench/data",
                home / "Library/Caches/mle-bench/data",
                Path("/home/zeus/.cache/mle-bench/data"),  # Lightning internal user
                Path("/root/.cache/mle-bench/data"),  # Docker / root containers
                Path(
                    "/teamspace/studios/this_studio/.cache/mle-bench/data"
                ),  # Lightning Studio
            ]

        # Find actual dataset directory
        dataset_path = None
        for root in _possible_cache_roots():
            candidate = root / competition_id
            if candidate.exists():
                dataset_path = candidate
                break

        if dataset_path is None:
            print(
                "[WARN] No existing dataset found. Using default HOME-based cache path."
            )
            dataset_path = Path.home() / ".cache/mle-bench/data" / competition_id

        self.cache_dir = dataset_path
        self.prepared_public = self.cache_dir / "prepared/public"
        self.prepared_private = self.cache_dir / "prepared/private"

        print(f"[INFO] Using dataset directory:\n       -> {self.cache_dir}")

        # ------------------------------------------------------------------
        # Path Calculation for New Folder Structure
        # ------------------------------------------------------------------
        agent_dir = Path(__file__).resolve().parent  # .../Hexo/melvin/agents

        # Go up 2 levels (agents -> melvin -> Hexo) to find mle-bench
        self.repo_root = agent_dir.parent.parent / "mle-bench"

        if not self.repo_root.exists():
            print(f"[ERROR] Could not find mle-bench at: {self.repo_root}")
            print("Please ensure the folder structure is correct.")
            sys.exit(1)

        if str(self.repo_root) not in sys.path:
            sys.path.append(str(self.repo_root))

        self.config_path = (
            self.repo_root
            / "mlebench"
            / "competitions"
            / competition_id
            / "config.yaml"
        )

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at: {self.config_path}")

        self.config = yaml.safe_load(open(self.config_path))

    # ----------------------------------------------------------------------
    # HELPER: Hardware Detection
    # ----------------------------------------------------------------------
    def get_hardware_profile(self):
        import psutil
        import shutil

        # CPU
        cpu_cores = os.cpu_count()

        # RAM
        ram_total_gb = round(psutil.virtual_memory().total / 1e9, 2)

        # DISK
        disk_total_gb = round(shutil.disk_usage("/").total / 1e9, 2)
        disk_free_gb = round(shutil.disk_usage("/").free / 1e9, 2)

        # GPU (if available)
        gpu_name = "None"
        gpu_vram_gb = 0
        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except:
            pass

        profile = {
            "cpu_cores": cpu_cores,
            "ram_gb": ram_total_gb,
            "disk_total_gb": disk_total_gb,
            "disk_free_gb": disk_free_gb,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram_gb,
        }

        # Log for transparency
        self.log({"step": "hardware_detection", "profile": profile})

        return profile

    # ----------------------------------------------------------------------
    # HELPER: Dynamic Module Loader
    # ----------------------------------------------------------------------
    def load_module_from_path(self, module_path_str):
        module_name_full, fn_name = module_path_str.split(":")

        if "-" not in module_name_full:
            module = importlib.import_module(module_name_full)
            return getattr(module, fn_name)

        parts = module_name_full.split(".")
        current_path = self.repo_root
        clean_parts = []

        for part in parts:
            if "-" in part:
                hyphen_name = part
                underscore_name = part.replace("-", "_")

                hyphen_dir = current_path / hyphen_name
                underscore_symlink = current_path / underscore_name

                if hyphen_dir.exists() and hyphen_dir.is_dir():
                    if not underscore_symlink.exists():
                        try:
                            underscore_symlink.symlink_to(hyphen_name)
                            importlib.invalidate_caches()
                        except OSError:
                            pass

                current_path = underscore_symlink
                clean_parts.append(underscore_name)
            else:
                current_path = current_path / part
                clean_parts.append(part)

        clean_module_path = ".".join(clean_parts)
        try:
            module = importlib.import_module(clean_module_path)
            return getattr(module, fn_name)
        except ImportError as e:
            raise ImportError(f"Failed to import fixed path '{clean_module_path}': {e}")

    # ----------------------------------------------------------------------
    # STEP 1 — PREPARE DATA
    # ----------------------------------------------------------------------
    def prepare_data(self):
        if self.prepared_public.exists():
            print("[INFO] Prepared data already exists. Skipping prepare step.")
            return

        print("[INFO] Preparing dataset...")
        preparer_path = self.config["preparer"]
        prepare_fn = self.load_module_from_path(preparer_path)

        raw_dir = self.cache_dir / "raw"
        public_dir = self.prepared_public
        private_dir = self.prepared_private

        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)

        prepare_fn(raw_dir, public_dir, private_dir)
        print("[INFO] Dataset prepared successfully.")

    # ----------------------------------------------------------------------
    # STEP 2 — DETECT TASK/MODALITY
    # ----------------------------------------------------------------------
    def full_detect(self):
        metadata = collect_dataset_metadata(self.prepared_public)
        result = asyncio.run(detect_modality_llm(metadata))
        self.log({"step": "modality_detection", "result": result})
        return (
            result["modality"],
            result["task_type"],
            result["target_col"],
            result.get("classes", []),
            metadata,
        )

    # ----------------------------------------------------------------------
    # STEP 3 — GENERATE TRAINING CODE
    # ----------------------------------------------------------------------
    def full_codegen(self, modality, task_type, target_col, classes, metadata, hardware):
        script_path = self.output_dir / "generated_train_script.py"

        asyncio.run(
            generate_training_script_llm(
                modality=modality,
                task_type=task_type,
                target_col=target_col,
                classes=classes,
                metadata=metadata,
                dataset_dir=self.prepared_public,
                output_path=script_path,
                seed=self.seed,
                hardware=hardware,
            )
        )
        print("[INFO] Training script generated.")
        return script_path

    # ----------------------------------------------------------------------
    # HELPER: PACKAGES
    # ----------------------------------------------------------------------
    @property
    def package_mapping(self):
        return {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
            "skimage": "scikit-image",
            "protobuf": "protobuf",
            "fitz": "pymupdf",
            "audiofile": "audiofile",
            "soundfile": "soundfile",
            "librosa": "librosa",
            "torchaudio": "torchaudio",
            "torchvision": "torchvision",
            "pydicom": "pydicom",
            "nibabel": "nibabel",
            "albumentations": "albumentations",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "transformers": "transformers",
            "tokenizers": "tokenizers",
            "datasets": "datasets",
            "evaluate": "evaluate",
            "sentencepiece": "sentencepiece",
            "SentencePiece": "sentencepiece",
            "accelerate": "accelerate",
            "py7zr": "py7zr",
        }

    # ----------------------------------------------------------------------
    # HELPER: ERROR RECOVERY (Dependency)
    # ----------------------------------------------------------------------
    def _handle_execution_error(self, error_msg):
        # 1. NumPy 2.x Fix
        if (
            "NumPy 1.x cannot be run in NumPy 2" in error_msg
            or "Failed to initialize NumPy" in error_msg
        ):
            print("[WARN] Environment Conflict: Downgrading NumPy...")
            self.log({"step": "error_recovery", "action": "downgrade_numpy"})
            try:
                subprocess.run(["uv", "pip", "install", "numpy<2"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False

        # 2. Scipy/Numpy Linkage Fix
        if "numpy.char" in error_msg or "module named 'numpy.char'" in error_msg:
            print("[WARN] SciPy/NumPy Mismatch Detected. Reinstalling SciPy...")
            self.log({"step": "error_recovery", "action": "reinstall_scipy"})
            try:
                subprocess.run(
                    ["uv", "pip", "install", "--force-reinstall", "scipy"], check=True
                )
                return True
            except subprocess.CalledProcessError:
                return False

        # 3. Install Missing Module
        missing_module = None
        match_std = re.search(r"No module named '(.+?)'", error_msg)
        if match_std:
            missing_module = match_std.group(1)

        if missing_module and "numpy" in missing_module:
            return False

        if not missing_module:
            match_req = re.search(r"requires the ([a-zA-Z0-9]+) library", error_msg)
            if match_req:
                missing_module = match_req.group(1)

        if missing_module:
            package_map = {
                "sklearn": "scikit-learn",
                "cv2": "opencv-python",
                "PIL": "Pillow",
                "yaml": "PyYAML",
            }
            package = package_map.get(missing_module, missing_module)

            if "mlebench" in package:
                return False

            print(f"[WARN] Installing missing dependency: '{package}'...")
            self.log(
                {
                    "step": "error_recovery",
                    "missing": missing_module,
                    "installing": package,
                }
            )

            try:
                subprocess.run(["uv", "pip", "install", package], check=True)
                importlib.invalidate_caches()
                return True
            except subprocess.CalledProcessError:
                return False

        return False

    # ----------------------------------------------------------------------
    # STEP 4 — RUN TRAINING SCRIPT (Streaming Output)
    # ----------------------------------------------------------------------
    def run_training_script(self, script_path):
        print("[INFO] Running training script...")
        self.log({"step": "training", "status": "starting", "seed": self.seed})

        max_retries = 10

        for attempt in range(max_retries):
            # We use Popen instead of run to capture output in real-time
            process = subprocess.Popen(
                [sys.executable, "-u", script_path.name],  # -u forces unbuffered output
                cwd=self.output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout so we see errors in order
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output to console AND capture it for the Fixer
            captured_logs = []

            # Read line by line as they happen
            for line in process.stdout:
                print(line, end="")  # Print to your terminal immediately
                captured_logs.append(line)

            # Wait for process to finish and get exit code
            process.wait()

            # Combine logs into one string
            full_log_output = "".join(captured_logs)

            if process.returncode == 0:
                self.log(
                    {"step": "training", "status": "success", "attempt": attempt + 1}
                )
                print("[INFO] Training script executed successfully.")
                return

            # --- FAILURE HANDLING ---
            print(f"\n[WARN] Process crashed with exit code {process.returncode}")

            # Logic to handle recovery
            # We pass the full_log_output because it contains the Traceback
            if self._handle_execution_error(full_log_output):
                print(f"[INFO] Retrying after install (Attempt {attempt + 1})...")
                continue

            print(
                f"[WARN] Code Execution Failed. Attempting AI Repair (Attempt {attempt + 1})..."
            )

            try:
                current_code = script_path.read_text()

                # Pass the captured logs to the fixer
                fixed_code = asyncio.run(
                    fix_training_script_llm(current_code, full_log_output)
                )

                script_path.write_text(fixed_code)
                print(f"[INFO] Applied AI Fix. Retrying...")
                self.log({"step": "error_recovery", "action": "ai_code_fix"})
                continue

            except Exception as fix_err:
                print(f"[ERROR] AI Repair failed: {fix_err}")
                raise RuntimeError(f"Training failed: {full_log_output}")

        raise RuntimeError("Training script failed too many times.")

    # ----------------------------------------------------------------------
    # STEP 5 — GRADE SUBMISSION (Via Wrapper Process)
    # ----------------------------------------------------------------------
    def grade_submission(self):
        print("[INFO] Grading submission...")

        submission_path = self.output_dir / "submission.csv"
        if not submission_path.exists():
            print("[ERROR] submission.csv not found!")
            return None

        grader_module_str = self.config["grader"]["grade_fn"]
        answers_path = self.prepared_private / "test.csv"
        repo_root_str = str(self.repo_root)

        # Calculate path to the static wrapper script
        wrapper_path = Path(__file__).parent / "grader_wrapper.py"

        if not wrapper_path.exists():
            print(f"[ERROR] grader_wrapper.py not found at {wrapper_path}")
            return None

        cmd = [
            sys.executable,
            str(wrapper_path),
            "--repo_root",
            repo_root_str,
            "--grader_module",
            grader_module_str,
            "--submission",
            str(submission_path),
            "--answers",
            str(answers_path),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            score = None
            for line in result.stdout.splitlines():
                if "SCORE_OUTPUT:" in line:
                    score = float(line.split("SCORE_OUTPUT:")[1])
                    print(f"[INFO] Final Score (Seed {self.seed}): {score}")

                    self.log({"step": "grading", "status": "success", "score": score})

                    with open(
                        self.output_dir / f"grading_report_seed_{self.seed}.json", "w"
                    ) as f:
                        json.dump({"score": score, "seed": self.seed}, f, indent=4)
                    return score

            print(
                f"[WARN] Grading finished but no score found. Output:\n{result.stdout}"
            )
            return None

        except subprocess.CalledProcessError as e:
            err = e.stderr
            print(f"[ERROR] Grading process crashed:\n{err}")

            if self._handle_execution_error(err):
                print("[INFO] Attempting to retry grading after environment fix...")
                return self.grade_submission()
            return None

    def log(self, entry: dict):
        log_path = self.output_dir / "reasoning_log.jsonl"
        entry_with_time = {"timestamp": time.time(), **entry}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry_with_time) + "\n")

    def run(self):
        self.hardware = self.get_hardware_profile()
        self.prepare_data()
        modality, task_type, target_col, classes, metadata = self.full_detect()

        script_path = self.full_codegen(
        modality, task_type, target_col, classes, metadata, self.hardware
        )

        self.run_training_script(script_path)
        self.grade_submission()


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    default_runs_dir = current_file.parent.parent / "runs"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", required=True)
    parser.add_argument("-o", "--output", default=str(default_runs_dir))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = MLEAgent(args.competition, args.output, args.seed)
    agent.run()
