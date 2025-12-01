import sys
import yaml
import importlib
import subprocess
import argparse
import json
import asyncio
import time
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

# Ensure these files exist in the same directory
from code_generator import generate_candidate_script, generate_final_script, fix_training_script_llm
from modality_detector import collect_dataset_metadata, detect_modality_llm
from retriever import retrieve_model_candidates

class MLEAgent:
    def __init__(self, competition_id, runs_base_dir, seed=42):
        self.competition_id = competition_id
        self.seed = seed
        self.runs_base_dir = Path(runs_base_dir)

        self.output_dir = self._setup_output_dir()
        self.cache_dir = self._locate_dataset()
        self.prepared_public = self.cache_dir / "prepared/public"
        self.prepared_private = self.cache_dir / "prepared/private"

        self.repo_root, self.config = self._load_mlebench_config()
        self.maximize_metric = self.config.get("metric", {}).get("maximize", True)

        self.hardware = self.get_hardware_profile()

    # ------------------------------------------------------------------
    # HELPER: Dependencies & Error Recovery
    # ------------------------------------------------------------------
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

    def log(self, data):
        # Basic file logger
        log_path = self.output_dir / "agent_log.jsonl"
        with open(log_path, "a") as f:
            data["timestamp"] = time.time()
            f.write(json.dumps(data) + "\n")

    def _handle_execution_error(self, error_msg):
        # 1. NumPy 2.x Fix
        if "NumPy 1.x cannot be run in NumPy 2" in error_msg or "Failed to initialize NumPy" in error_msg:
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
                subprocess.run(["uv", "pip", "install", "--force-reinstall", "scipy"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False

        # 3. HuggingFace / Transformers Version Conflict (NEW FIX)
        if "huggingface-hub" in error_msg and "is required" in error_msg:
            print("[WARN] HuggingFace Version Conflict Detected. Upgrading stack...")
            self.log({"step": "error_recovery", "action": "upgrade_transformers"})
            try:
                # Upgrade key NLP libraries together to ensure compatibility
                subprocess.run(["uv", "pip", "install", "-U", "transformers", "huggingface-hub", "tokenizers", "accelerate"], check=True)
                importlib.invalidate_caches()
                return True
            except subprocess.CalledProcessError:
                return False

        # 4. Install Missing Module
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
            package = self.package_mapping.get(missing_module, missing_module)
            if "mlebench" in package: return False

            print(f"[WARN] Installing missing dependency: '{package}'...")
            self.log({"step": "error_recovery", "missing": missing_module, "installing": package})

            try:
                subprocess.run(["uv", "pip", "install", package], check=True)
                importlib.invalidate_caches()
                return True
            except subprocess.CalledProcessError:
                return False

        return False

    # ------------------------------------------------------------------
    # HELPER: Setup & Config
    # ------------------------------------------------------------------
    def _setup_output_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{self.competition_id}"
        out_dir = self.runs_base_dir.resolve() / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Initialized run: {out_dir}")
        return out_dir

    def _locate_dataset(self):
        home = Path.home()
        possible_roots = [
            Path("/teamspace/studios/this_studio/.cache/mle-bench/data"), 
            home / ".cache/mle-bench/data",
            Path("/home/zeus/.cache/mle-bench/data"),
            Path("/root/.cache/mle-bench/data"),
        ]
        
        for root in possible_roots:
            candidate = root / self.competition_id
            try:
                if candidate.exists():
                    print(f"[INFO] Found dataset at: {candidate}")
                    return candidate
            except PermissionError:
                continue
                
        return home / ".cache/mle-bench/data" / self.competition_id

    def _load_mlebench_config(self):
        agent_dir = Path(__file__).resolve().parent
        possible_repo_roots = [
            agent_dir.parent.parent / "mle-bench",
            agent_dir.parent / "mle-bench",
            Path("mle-bench").resolve()
        ]
        repo_root = None
        for r in possible_repo_roots:
            if r.exists():
                repo_root = r
                break
        
        if not repo_root: repo_root = Path("mle-bench")
        if str(repo_root) not in sys.path: sys.path.append(str(repo_root))

        config_path = repo_root / "mlebench" / "competitions" / self.competition_id / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at: {config_path}")

        return repo_root, yaml.safe_load(open(config_path))

    def get_hardware_profile(self):
        return {} 

    def load_module_from_path(self, module_path_str):
        module_name_full, fn_name = module_path_str.split(":")
        try:
            module = importlib.import_module(module_name_full)
            return getattr(module, fn_name)
        except ImportError:
            pass
        sys.path.append(str(self.repo_root))
        module = importlib.import_module(module_name_full)
        return getattr(module, fn_name)

    # ------------------------------------------------------------------
    # STEP 1: Prepare Data
    # ------------------------------------------------------------------
    def prepare_data(self):
        if self.prepared_public.exists():
            print("[INFO] Data already prepared.")
            return
        print("[INFO] Preparing dataset via MLE-Bench...")
        preparer_path = self.config["preparer"]
        prepare_fn = self.load_module_from_path(preparer_path)
        raw_dir = self.cache_dir / "raw"
        prepare_fn(raw_dir, self.prepared_public, self.prepared_private)
        print("[INFO] Dataset prepared.")

    # ------------------------------------------------------------------
    # MAIN PIPELINE (PHASE 1)
    # ------------------------------------------------------------------
    def run(self):
        self.prepare_data()
        
        # 1. Detect Task
        print("[1/5] Analyzing Task...")
        metadata = collect_dataset_metadata(self.prepared_public)
        modality_info = asyncio.run(detect_modality_llm(metadata))
        print(f"      Detected: {modality_info['modality']} | {modality_info['task_type']}")

        # 2. Retrieve Candidates
        print("[2/5] Searching for SOTA models...")
        candidates = asyncio.run(retrieve_model_candidates(
            metadata, self.competition_id, modality_info['task_type'], modality_info['modality']
        ))
        print(f"      Found {len(candidates)} candidates: {[c['model_name'] for c in candidates]}")

        # 3. Tournament
        print("[3/5] Running Candidate Tournament (Subsampled)...")
        best_candidate = None
        best_score = -float("inf") if self.maximize_metric else float("inf")
        best_code = None
        
        for i, cand in enumerate(candidates):
            print(f"      Evaluating Candidate {i+1}: {cand['model_name']}...")
            script_name = f"candidate_{i}.py"
            script_path = self.output_dir / script_name
            
            code = asyncio.run(generate_candidate_script(
                cand, modality_info, metadata, self.prepared_public, self.seed
            ))
            script_path.write_text(code)

            # Execute with Retries for Dependencies
            score = self.execute_candidate_robust(script_path, timeout=600)
            print(f"      -> Score: {score}")
            
            if score is not None:
                if best_candidate is None:
                    best_score = score
                    best_candidate = cand
                    best_code = code
                else:
                    better = (score > best_score) if self.maximize_metric else (score < best_score)
                    if better:
                        best_score = score
                        best_candidate = cand
                        best_code = code


        if best_candidate is None:
            print("[CRITICAL] All candidates failed. Falling back to Candidate 0.")
            best_candidate = candidates[0]
            best_code = script_path.read_text()

        print(f"[WINNER] Best Strategy: {best_candidate['model_name']} (Score: {best_score})")

        # 4. Final Training
        print("[4/5] Training Final Model on Full Data...")
        final_script_path = self.output_dir / "train.py"
        final_code = asyncio.run(generate_final_script(
            best_candidate, best_code, modality_info, metadata, self.prepared_public, self.seed
        ))
        final_script_path.write_text(final_code)

        success = self.run_training_script_robust(final_script_path, timeout=86400)
        
        # 5. Grading
        if success:
            print("[5/5] Grading...")
            self.grade_submission()
        else:
            print("[FAIL] Final training failed.")

    # ------------------------------------------------------------------
    # EXECUTION HELPERS (ROBUST & STREAMING)
    # ------------------------------------------------------------------
    def execute_candidate_robust(self, script_path, timeout=600):
        start_time = time.time()
        # Allow up to 5 repair attempts per candidate
        max_retries = 5

        for attempt in range(max_retries):
            try:
                # 1. Run the script
                proc = subprocess.Popen(
                    [sys.executable, "-u", script_path.name],
                    cwd=self.output_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                captured_lines = []
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        print(f"      [LOG] {line.strip()[:100]}") 
                        captured_lines.append(line)
                    
                    if time.time() - start_time > timeout:
                        proc.kill()
                        print("      [WARN] Timeout reached!")
                        return None 

                full_log = "".join(captured_lines)

                # 2. Success Check
                if proc.returncode == 0:
                    match = re.search(r"FINAL_SCORE:\s*([\d\.-]+)", full_log)
                    if match:
                        return float(match.group(1))
                    print("      [WARN] Script finished but no FINAL_SCORE printed.")
                    return None 

                # 3. Failure Handling
                print(f"      [WARN] Crash detected (Attempt {attempt+1}/{max_retries}). Analyzing...")

                # A. Dependency Fix?
                if self._handle_execution_error(full_log):
                    print(f"      [INFO] Environment fixed. Retrying...")
                    continue
                
                # B. AI Code Fix? (Enable this!)
                print(f"      [INFO] Applying AI Code Fix...")
                try:
                    current_code = script_path.read_text()
                    # Call the fixer we imported
                    fixed_code = asyncio.run(fix_training_script_llm(current_code, full_log))
                    script_path.write_text(fixed_code)
                    continue # Retry with new code
                except Exception as e:
                    print(f"      [ERR] AI Fixer failed: {e}")
                    return None

            except Exception as e:
                print(f"      [ERR] Execution exception: {e}")
                return None
        
        print("      [ERR] Candidate failed after max retries.")
        return None

    def run_training_script_robust(self, script_path, timeout=3600):
        print(f"[INFO] Executing {script_path.name}...")
        
        start_time = time.time()
        max_retries = 10 
        
        for attempt in range(max_retries):
            with open(self.output_dir / f"log_final_{attempt}.txt", "w") as logf:
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", script_path.name],
                        cwd=self.output_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    full_log = []
                    while True:
                        line = proc.stdout.readline()
                        if not line and proc.poll() is not None:
                            break
                        if line:
                            print(line, end="")
                            logf.write(line)
                            full_log.append(line)
                        
                        if time.time() - start_time > timeout:
                            proc.kill()
                            raise TimeoutError("Time limit exceeded")
                    
                    # 1. Success Check
                    if proc.returncode == 0:
                        if self._validate_submission(self.output_dir / "submission.csv"):
                            return True
                        else:
                            raise RuntimeError("CSV Validation Failed")
                    
                    # 2. Failure Handling
                    err_msg = "".join(full_log)
                    
                    # A. Dependency Fix?
                    if self._handle_execution_error(err_msg):
                        print(f"[INFO] Dependency fixed. Retrying...")
                        continue 
                    
                    # B. Code Logic Fix (LLM)
                    print(f"[WARN] Attempt {attempt} failed. Retrying with AI Fix...")
                    new_code = asyncio.run(fix_training_script_llm(script_path.read_text(), err_msg))
                    script_path.write_text(new_code)
                    
                except Exception as e:
                    print(f"[ERR] Attempt {attempt} error: {e}")
        
        return False

    def _validate_submission(self, csv_path):
        if not csv_path.exists():
            print("[ERR] submission.csv missing")
            return False
        try:
            df = pd.read_csv(csv_path)
            if df.isnull().values.any():
                print("[ERR] Submission contains NaNs")
                return False
            return True
        except:
            return False

    def grade_submission(self):
        print("[INFO] Grading submission...")
        submission_path = self.output_dir / "submission.csv"
        if not submission_path.exists():
            return None

        grader_module_str = self.config["grader"]["grade_fn"]
        answers_path = self.prepared_private / "test.csv"
        
        wrapper_path = Path(__file__).parent / "grader_wrapper.py"
        cmd = [
            sys.executable, str(wrapper_path),
            "--repo_root", str(self.repo_root),
            "--grader_module", grader_module_str,
            "--submission", str(submission_path),
            "--answers", str(answers_path),
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "SCORE_OUTPUT:" in line:
                    score = float(line.split("SCORE_OUTPUT:")[1])
                    print(f"[INFO] *** FINAL SCORE: {score} ***")
                    
                    with open(self.output_dir / f"grading_report_seed_{self.seed}.json", "w") as f:
                        json.dump({"score": score, "seed": self.seed}, f)
                    return score
            print(f"[WARN] No score found. Wrapper Output:\n{res.stdout}\n{res.stderr}")
        except Exception as e:
            print(f"[ERR] Grading crash: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", required=True)
    parser.add_argument("-o", "--output", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = MLEAgent(args.competition, args.output, args.seed)
    agent.run()