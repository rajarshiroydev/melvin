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

from pathlib import Path
from code_generator import generate_training_script_llm
from modality_detector import collect_dataset_metadata, detect_modality_llm


class MLEAgent:
    def __init__(self, competition_id, output_dir):
        self.competition_id = competition_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MLEbench cache directory
        self.cache_dir = Path.home() / "Library/Caches/mle-bench/data" / competition_id
        self.prepared_public = self.cache_dir / "prepared/public"
        self.prepared_private = self.cache_dir / "prepared/private"

        # Calculate path relative to hexo/agent/run_agent.py
        # 1. Start at run_agent.py location (.../hexo/agent)
        agent_dir = Path(__file__).resolve().parent

        # 2. Go up to 'hexo', then down to 'mle-bench' repo root
        self.repo_root = agent_dir.parent / "mle-bench"

        # Add the repo root to Python's path
        if str(self.repo_root) not in sys.path:
            sys.path.append(str(self.repo_root))

        # 3. Construct path to config
        self.config_path = (
            self.repo_root
            / "mlebench"
            / "competitions"
            / competition_id
            / "config.yaml"
        )

        # Load config
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at: {self.config_path}")

        self.config = yaml.safe_load(open(self.config_path))

    # ----------------------------------------------------------------------
    # HELPER: Dynamic Module Loader (Symlink Strategy)
    # ----------------------------------------------------------------------
    def load_module_from_path(self, module_path_str):
        """
        Loads a module handling hyphenated directories by creating temporary symlinks.
        Input: 'mlebench.competitions.spooky-author-identification.grade:grade'
        """
        module_name_full, fn_name = module_path_str.split(":")

        # If there are no hyphens, just import normally
        if "-" not in module_name_full:
            module = importlib.import_module(module_name_full)
            return getattr(module, fn_name)

        # LOGIC: Walk the path, find hyphenated dirs, symlink them to underscore versions
        parts = module_name_full.split(".")
        current_path = self.repo_root

        # We rebuild the module path with underscores to import it later
        # e.g. mlebench.competitions.spooky_author_identification.grade
        clean_parts = []

        for part in parts:
            if "-" in part:
                hyphen_name = part
                underscore_name = part.replace("-", "_")

                hyphen_dir = current_path / hyphen_name
                underscore_symlink = current_path / underscore_name

                # Check if we need to create a symlink
                if hyphen_dir.exists() and hyphen_dir.is_dir():
                    if not underscore_symlink.exists():
                        print(
                            f"[INFO] Fixing import path: Linking {underscore_name} -> {hyphen_name}"
                        )
                        try:
                            underscore_symlink.symlink_to(hyphen_name)
                            # Invalidate cache so Python sees the new 'directory' immediately
                            importlib.invalidate_caches()
                        except OSError as e:
                            print(f"[WARN] Could not create symlink: {e}")

                current_path = underscore_symlink
                clean_parts.append(underscore_name)
            else:
                current_path = current_path / part
                clean_parts.append(part)

        # Import using the clean (underscored) path
        clean_module_path = ".".join(clean_parts)

        try:
            module = importlib.import_module(clean_module_path)
            return getattr(module, fn_name)
        except ImportError as e:
            # Re-raise with context
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
        self.log({"step": "modality_detection", "metadata": metadata, "result": result})
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
            )
        )
        print(f"[INFO] Training script generated: {script_path}")
        return script_path

    # ----------------------------------------------------------------------
    # HELPER: PACKAGE MAPPING
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
            "google.protobuf": "protobuf",
            "fitz": "pymupdf",
            "usb": "pyusb",
            "serial": "pyserial",
            "dotenv": "python-dotenv",
            "dateutil": "python-dateutil",
            "docx": "python-docx",
            "ppt": "python-pptx",
            "jq": "jq",
            "dns": "dnspython",
            "jwt": "pyjwt",
            "kafka": "kafka-python",
            "multipart": "python-multipart",
            "xgboost": "xgboost",
            "lxml": "lxml",
            "crypto": "pycryptodome",
            "Crypto": "pycryptodome",
            "py7zr": "py7zr",
        }

    # ----------------------------------------------------------------------
    # HELPER: INSTALL MISSING DEPENDENCY
    # ----------------------------------------------------------------------
    def _install_missing_dependency(self, error_msg):
        """Attempts to install a missing module based on the error message."""
        missing_module = self.extract_missing_module(error_msg)

        if not missing_module:
            return False

        package_to_install = self.package_mapping.get(missing_module, missing_module)

        print(f"[WARN] Missing module: '{missing_module}'")
        if package_to_install != missing_module:
            print(f"       Mapped to package: '{package_to_install}'")

        print(f"       Installing {package_to_install}...")

        try:
            subprocess.run(["uv", "pip", "install", package_to_install], check=True)
            print(f"[INFO] Installed {package_to_install} successfully.")
            importlib.invalidate_caches()
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {package_to_install}: {e}")
            raise e

    # ----------------------------------------------------------------------
    # STEP 4 — RUN TRAINING SCRIPT (subprocess)
    # ----------------------------------------------------------------------
    def run_training_script(self, script_path):
        print("[INFO] Running training script...")

        max_retries = 5

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
                print("[INFO] Training script executed successfully.")
                return

            except subprocess.CalledProcessError as e:
                error_output = e.stderr
                print(e.stdout)
                print(error_output, file=sys.stderr)

                if self._install_missing_dependency(error_output):
                    continue

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
            self.log({"step": "grading", "error": "submission.csv missing"})
            return None

        grader_path = self.config["grader"]["grade_fn"]
        answers_path = self.prepared_private / "test.csv"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load grader using symlink strategy
                grade_fn = self.load_module_from_path(grader_path)

                sample_sub = self.prepared_public / "sample_submission.csv"
                sample_cols = list(pd.read_csv(sample_sub).columns)
                submitted_cols = list(pd.read_csv(submission_path).columns)

                if submitted_cols != sample_cols:
                    print("[WARN] Submission columns do not match sample submission!")
                    self.log(
                        {
                            "step": "grading",
                            "warning": "column mismatch",
                            "expected": sample_cols,
                            "got": submitted_cols,
                        }
                    )

                # FIX: Load DataFrames before passing to grade_fn
                # The grader implementation expects objects with .shape, not Paths.
                submission_df = pd.read_csv(submission_path)
                answers_df = pd.read_csv(answers_path)

                score = grade_fn(submission_df, answers_df)
                print("[INFO] Score:", score)

                self.log({"step": "grading", "score": score})
                with open(self.output_dir / "grading_report.json", "w") as f:
                    json.dump(score, f, indent=4)

                return score

            except ImportError as e:
                print(f"[WARN] ImportError during grading: {e}")
                if self._install_missing_dependency(str(e)):
                    continue
                raise e

    # ------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------
    def log(self, entry: dict):
        log_path = self.output_dir / "reasoning_log.jsonl"
        entry_with_time = {"timestamp": time.time(), **entry}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry_with_time) + "\n")

    def extract_missing_module(self, error_msg: str):
        match = re.search(r"No module named '(.+?)'", error_msg)
        return match.group(1) if match else None

    # ----------------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------------
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
    parser.add_argument("-o", "--output", default="agent_output")
    args = parser.parse_args()

    agent = MLEAgent(args.competition, args.output)
    agent.run()
