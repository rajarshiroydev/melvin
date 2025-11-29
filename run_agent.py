import re
import yaml
import importlib
import importlib.util
import subprocess
import argparse
import json
import sys
import asyncio
import time
import pandas as pd
from datetime import datetime

from pathlib import Path
from code_generator import generate_training_script_llm, fix_training_script_llm
from modality_detector import collect_dataset_metadata, detect_modality_llm


class MLEAgent:
    def __init__(self, competition_id, runs_base_dir, seed=42):
        self.competition_id = competition_id
        self.seed = seed

        # ------------------------------------------------------------------
        # NEW: Unique Run Directory Logic
        # Format: runs/{YYYY-MM-DD_HH-MM-SS}_{competition_id}
        # ------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{competition_id}"

        self.output_dir = Path(runs_base_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[INFO] Initialized new run. Logs and outputs will be saved to:\n       -> {self.output_dir}"
        )

        # MLEbench cache directory
        self.cache_dir = Path.home() / "Library/Caches/mle-bench/data" / competition_id
        self.prepared_public = self.cache_dir / "prepared/public"
        self.prepared_private = self.cache_dir / "prepared/private"

        # Calculate path relative to run_agent.py
        agent_dir = Path(__file__).resolve().parent
        self.repo_root = agent_dir.parent / "mle-bench"

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
    # HELPER: Dynamic Module Loader (Symlink Strategy)
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
    def full_codegen(self, modality, task_type, target_col, classes, metadata):
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

        # 2. Install Missing Module
        missing_module = None
        match_std = re.search(r"No module named '(.+?)'", error_msg)
        if match_std:
            missing_module = match_std.group(1)

        if not missing_module:
            match_req = re.search(r"requires the ([a-zA-Z0-9]+) library", error_msg)
            if match_req:
                missing_module = match_req.group(1)

        if missing_module:
            package = self.package_mapping.get(missing_module, missing_module)
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
    # STEP 4 — RUN TRAINING SCRIPT (Loop with Fixer)
    # ----------------------------------------------------------------------
    def run_training_script(self, script_path):
        print("[INFO] Running training script...")
        self.log({"step": "training", "status": "starting", "seed": self.seed})

        max_retries = 10

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    [sys.executable, script_path.name],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=self.output_dir,
                )
                print(result.stdout)
                self.log(
                    {"step": "training", "status": "success", "attempt": attempt + 1}
                )
                print("[INFO] Training script executed successfully.")
                return

            except subprocess.CalledProcessError as e:
                error_output = e.stderr
                print(e.stdout)
                print(error_output, file=sys.stderr)

                self.log(
                    {
                        "step": "training_fail",
                        "attempt": attempt + 1,
                        "error_snippet": error_output[:500],
                    }
                )

                # A. Try Dependency Fix
                if self._handle_execution_error(error_output):
                    print(f"[INFO] Retrying after install (Attempt {attempt + 1})...")
                    continue

                # B. Try Code Logic Fix (Reflexion)
                print(
                    f"[WARN] Code Execution Failed. Attempting AI Repair (Attempt {attempt + 1})..."
                )

                try:
                    current_code = script_path.read_text()

                    # Call the fixer LLM
                    fixed_code = asyncio.run(
                        fix_training_script_llm(current_code, error_output)
                    )

                    # Overwrite the script
                    script_path.write_text(fixed_code)

                    print(f"[INFO] Applied AI Fix. Retrying...")
                    self.log({"step": "error_recovery", "action": "ai_code_fix"})
                    continue

                except Exception as fix_err:
                    print(f"[ERROR] AI Repair failed: {fix_err}")
                    raise e

        raise RuntimeError("Training script failed too many times.")

    # ----------------------------------------------------------------------
    # STEP 5 — GRADE SUBMISSION
    # ----------------------------------------------------------------------
    def grade_submission(self):
        print("[INFO] Grading submission...")

        submission_path = self.output_dir / "submission.csv"
        if not submission_path.exists():
            print("[ERROR] submission.csv not found!")
            return None

        grader_path = self.config["grader"]["grade_fn"]
        answers_path = self.prepared_private / "test.csv"

        for attempt in range(5):
            try:
                grade_fn = self.load_module_from_path(grader_path)

                # Basic check
                sample_sub = self.prepared_public / "sample_submission.csv"
                sub_cols = list(pd.read_csv(submission_path).columns)
                sam_cols = list(pd.read_csv(sample_sub).columns)

                if sub_cols != sam_cols:
                    print("[WARN] Submission columns do not match sample!")

                sub_df = pd.read_csv(submission_path)
                ans_df = pd.read_csv(answers_path)

                score = grade_fn(sub_df, ans_df)
                print(f"[INFO] Final Score (Seed {self.seed}): {score}")

                self.log({"step": "grading", "status": "success", "score": score})

                with open(
                    self.output_dir / f"grading_report_seed_{self.seed}.json", "w"
                ) as f:
                    json.dump({"score": score, "seed": self.seed}, f, indent=4)

                return score

            except Exception as e:
                err_str = str(e)
                print(f"[WARN] Grading failed: {err_str}")
                if self._handle_execution_error(err_str):
                    continue
                raise e

    def log(self, entry: dict):
        log_path = self.output_dir / "reasoning_log.jsonl"
        entry_with_time = {"timestamp": time.time(), **entry}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry_with_time) + "\n")

    def run(self):
        self.prepare_data()
        modality, task_type, target_col, classes, metadata = self.full_detect()

        script_path = self.full_codegen(
            modality, task_type, target_col, classes, metadata
        )

        self.run_training_script(script_path)
        self.grade_submission()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", required=True)
    # Changed default from "agent_output" to "runs"
    parser.add_argument("-o", "--output", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = MLEAgent(args.competition, args.output, args.seed)
    agent.run()
