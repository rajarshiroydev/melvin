import re
import yaml
import importlib
import subprocess
import argparse
import json
import sys
import asyncio
import time

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
        repo_root = agent_dir.parent / "mle-bench"

        # 3. Construct path to config (assuming standard structure: mle-bench/mlebench/competitions/...)
        self.config_path = (
            repo_root / "mlebench" / "competitions" / competition_id / "config.yaml"
        )

        # Load config
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at: {self.config_path}")

        self.config = yaml.safe_load(open(self.config_path))

    # ----------------------------------------------------------------------
    # STEP 1 — PREPARE DATA (run preparer)
    # ----------------------------------------------------------------------
    def prepare_data(self):
        if self.prepared_public.exists():
            print("[INFO] Prepared data already exists. Skipping prepare step.")
            return

        print("[INFO] Preparing dataset...")

        # Example preparer path:
        # "mlebench.competitions.spooky-author-identification.prepare:prepare"
        preparer_path = self.config["preparer"]
        module_path, fn_name = preparer_path.split(":")

        # module paths can't contain hyphens
        module_path = module_path.replace("-", "_")

        module = importlib.import_module(module_path)
        prepare_fn = getattr(module, fn_name)

        raw_dir = self.cache_dir / "raw"
        public_dir = self.prepared_public
        private_dir = self.prepared_private

        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)

        # Run preparer
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
        """
        Maps import names (module names) to PyPI package names.
        Add to this list as you discover more edge cases.
        """
        return {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
            "skimage": "scikit-image",
            "protobuf": "protobuf",  # frequent confusion with google.protobuf
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
            "crypto": "pycryptodome",  # Common confusion, 'pycrypto' is dead
            "Crypto": "pycryptodome",
        }

    # ----------------------------------------------------------------------
    # STEP 4 — RUN TRAINING SCRIPT (subprocess)
    # ----------------------------------------------------------------------
    def run_training_script(self, script_path):
        print("[INFO] Running training script...")

        max_retries = 5

        for attempt in range(max_retries):
            try:
                # Run with uv directly using 'uv run' is often cleaner,
                # but sticking to your current venv python approach:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)
                print("[INFO] Training script executed successfully.")
                return

            except subprocess.CalledProcessError as e:
                error_output = e.stderr
                print(e.stdout)
                print(error_output, file=sys.stderr)

                missing_module = self.extract_missing_module(error_output)

                if missing_module:
                    # CHECK MAPPING HERE
                    package_to_install = self.package_mapping.get(
                        missing_module, missing_module
                    )

                    print(f"[WARN] Missing module: '{missing_module}'")
                    if package_to_install != missing_module:
                        print(f"       Mapped to package: '{package_to_install}'")

                    print(f"       Installing {package_to_install}...")

                    # FIX: Use "uv" directly, not "sys.executable -m uv"
                    # Using 'uv pip install' is safer for existing venvs than 'uv add'
                    # unless you are strictly managing a pyproject.toml
                    install_cmd = ["uv", "pip", "install", package_to_install]

                    try:
                        subprocess.run(install_cmd, check=True)
                        print(f"[INFO] Installed {package_to_install} successfully.")
                        continue  # Retry the loop
                    except Exception as inst_err:
                        print(
                            f"[ERROR] Failed to install {package_to_install}: {inst_err}"
                        )
                        raise inst_err

                raise e

        raise RuntimeError("Training script failed too many times.")

    # ----------------------------------------------------------------------
    # STEP 5 — GRADE SUBMISSION (optional during dev)
    # ----------------------------------------------------------------------
    def grade_submission(self):
        print("[INFO] Grading submission...")

        submission_path = self.output_dir / "submission.csv"
        if not submission_path.exists():
            print("[ERROR] submission.csv not found!")
            return None

        grader_path = self.config["grader"]["grade_fn"]
        module_path, fn_name = grader_path.split(":")
        module_path = module_path.replace("-", "_")

        module = importlib.import_module(module_path)
        grade_fn = getattr(module, fn_name)

        answers_path = self.prepared_private / "test.csv"

        score = grade_fn(submission_path, answers_path)
        print("[INFO] Score:", score)

        with open(self.output_dir / "grading_report.json", "w") as f:
            json.dump(score, f, indent=4)

        return score

    # ------------------------------------------------------
    # LOGGING (Reasoning Trace)
    # ------------------------------------------------------
    def log(self, entry: dict):
        """Append a JSON reasoning log entry into agent_output directory."""

        log_path = self.output_dir / "reasoning_log.jsonl"
        entry_with_time = {"timestamp": time.time(), **entry}

        with open(log_path, "a") as f:
            f.write(json.dumps(entry_with_time) + "\n")

    # ------------------------------------------------------
    # Missing Module Detector
    # ------------------------------------------------------
    def extract_missing_module(self, error_msg: str):
        match = re.search(r"No module named '(.+?)'", error_msg)
        return match.group(1) if match else None

    # ----------------------------------------------------------------------
    # RUN EVERYTHING
    # ----------------------------------------------------------------------
    def run(self):
        # Step 1: Prepare dataset
        self.prepare_data()

        # Step 2: Detect modality/task
        modality, task_type, target_col, classes, metadata = self.full_detect()

        # Step 3: Generate training script
        script_path = self.full_codegen(
            modality, task_type, target_col, classes, metadata
        )

        # Step 4: Execute training script
        self.run_training_script(script_path)

        # Step 5: Grade submission
        self.grade_submission()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", required=True)
    parser.add_argument("-o", "--output", default="agent_output")
    args = parser.parse_args()

    agent = MLEAgent(args.competition, args.output)
    agent.run()
