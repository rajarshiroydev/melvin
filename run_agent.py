import yaml
import importlib
import subprocess
import argparse
import json
import sys

from pathlib import Path
from agents.modality_detector import collect_dataset_metadata, detect_modality_llm


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
    def detect(self):
        metadata = collect_dataset_metadata(self.prepared_public)
        result = detect_modality_llm(metadata)
        self.log({"step": "modality_detection", "result": result})
        return result

    # ----------------------------------------------------------------------
    # STEP 3 — GENERATE TRAINING CODE
    # ----------------------------------------------------------------------
    def generate_training_code(self, modality, task_type):
        print("[INFO] Generating training script...")

        train_script_path = self.output_dir / "generated_train_script.py"

        # placeholder — we will fill this in after skeleton
        code = """
        # Auto-generated training script
        # print("Training script placeholder. Will be filled in later.")
        """

        train_script_path.write_text(code)
        print(f"[INFO] Training script saved to {train_script_path}")

        return train_script_path

    # ----------------------------------------------------------------------
    # STEP 4 — RUN TRAINING SCRIPT (subprocess)
    # ----------------------------------------------------------------------
    def run_training_script(self, script_path):
        print("[INFO] Running training script...")
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("[INFO] Training script executed successfully.")

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

    # ----------------------------------------------------------------------
    # RUN EVERYTHING
    # ----------------------------------------------------------------------
    def run(self):
        self.prepare_data()
        modality, task_type = self.detect_task()
        script_path = self.generate_training_code(modality, task_type)
        self.run_training_script(script_path)
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
