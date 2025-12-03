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
from refiner import propose_ablations, propose_refinements, apply_refinement_llm

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
        
        # --- LOGGING SETUP ---
        self.step_counter = 0
        self.trace_file = self.output_dir / "reasoning_trace.md"
        self._init_trace_log()

    # ------------------------------------------------------------------
    # HELPER: Reasoning Trace Logging
    # ------------------------------------------------------------------
    def _init_trace_log(self):
        """Initializes the Markdown trace file."""
        header = f"""# 🧠 Agent Reasoning Trace
        **Competition:** {self.competition_id}
        **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        **Seed:** {self.seed}

        ---
        """
        with open(self.trace_file, "w", encoding="utf-8") as f:
            f.write(header)

    def log_step(self, title, content, icon="🤖", code=None, output=None, tags=None):
        """
        Logs a reasoning step to the Markdown trace and JSONL.
        """
        self.step_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 1. Write to JSONL (Machine Readable) - Keeps code/output for debugging
        log_entry = {
            "step": self.step_counter,
            "timestamp": time.time(),
            "icon": icon,
            "title": title,
            "content": content,
            "code": code,
            "output": output,
            "tags": tags or []
        }
        self.log(log_entry)

        # 2. Write to Markdown (Human Readable - TEXT ONLY)
        md_entry = f"### Step {self.step_counter} {icon} **{title}** <span style='color:grey; font-size:0.8em'>({timestamp})</span>\n\n"
        md_entry += f"{content}\n\n"
        md_entry += "---\n"

        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(md_entry)
        
        # Print to console for real-time feedback
        print(f"\n[{self.step_counter}] {icon} {title}: {content[:100]}...")

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
            if "timestamp" not in data:
                data["timestamp"] = time.time()
            f.write(json.dumps(data) + "\n")

    def _handle_execution_error(self, error_msg):
        # 1. NumPy 2.x Fix
        if "NumPy 1.x cannot be run in NumPy 2" in error_msg or "Failed to initialize NumPy" in error_msg:
            self.log_step("Environment Error", "NumPy version conflict detected. Action: Downgrade NumPy to 1.x.", "⚠️")
            print("[WARN] Environment Conflict: Downgrading NumPy...")
            self.log({"step": "error_recovery", "action": "downgrade_numpy"})
            try:
                subprocess.run(["uv", "pip", "install", "numpy<2"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False

        # 2. Scipy/Numpy Linkage Fix
        if "numpy.char" in error_msg or "module named 'numpy.char'" in error_msg:
            self.log_step("Environment Error", "SciPy/NumPy mismatch detected. Action: Reinstall SciPy.", "⚠️")
            print("[WARN] SciPy/NumPy Mismatch Detected. Reinstalling SciPy...")
            self.log({"step": "error_recovery", "action": "reinstall_scipy"})
            try:
                subprocess.run(["uv", "pip", "install", "--force-reinstall", "scipy"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False

        # 3. HuggingFace / Transformers Version Conflict (NEW FIX)
        if "huggingface-hub" in error_msg and "is required" in error_msg:
            self.log_step("Environment Error", "HuggingFace version conflict detected. Action: Upgrade transformers stack.", "⚠️")
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
            self.log_step("Dependency Install", f"Missing module '{missing_module}'. Installing '{package}'.", "📦")
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
            self.log_step("Data Check", "Data already exists in cache. Skipping preparation.", "📂")
            return
        
        self.log_step("Data Preparation", "Preparing dataset via MLE-Bench standard preparer...", "🛠️")
        print("[INFO] Preparing dataset via MLE-Bench...")
        preparer_path = self.config["preparer"]
        prepare_fn = self.load_module_from_path(preparer_path)
        raw_dir = self.cache_dir / "raw"
        prepare_fn(raw_dir, self.prepared_public, self.prepared_private)
        print("[INFO] Dataset prepared.")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        self.prepare_data()
        
        # --- 1. ANALYSIS ---
        print("[1/5] Analyzing Task...")
        metadata = collect_dataset_metadata(self.prepared_public)
        
        # Log the description text
        self.log_step("Task Analysis", 
                      f"Reading dataset description and metadata.\n**Description Snippet:**\n_{metadata.get('description', '')[:200]}..._", 
                      "📚")
        
        modality_info = asyncio.run(detect_modality_llm(metadata))
        print(f"      Detected: {modality_info['modality']} | {modality_info['task_type']}")
        
        self.log_step("Modality Detection", 
                      f"Identified task properties:\n- **Modality:** {modality_info['modality']}\n- **Task Type:** {modality_info['task_type']}\n- **Target:** {modality_info.get('target_col', 'N/A')}", 
                      "🔍")

        # --- 2. RETRIEVAL ---
        print("[2/5] Searching & Strategizing...")
        self.log_step("Research & Retrieval", f"Searching for SOTA approaches for {modality_info['task_type']} on {modality_info['modality']} data...", "🌏")
        
        retrieval_data = asyncio.run(retrieve_model_candidates(
            metadata, self.competition_id, modality_info['task_type'], modality_info['modality']
        ))
        
        # EXTRACT METRIC DIRECTION
        task = modality_info["task_type"]

        if task in ["classification", "image_classification", "audio_classification"]:
            metric_dir = "maximize"
        elif task in ["regression"]:
            metric_dir = "minimize"
        elif task in ["seq2seq"]:
            metric_dir = "minimize"
        else:
            metric_dir = "maximize"

        print(f"      Metric Goal: {metric_dir.upper()} (based on task type)")

        candidates = retrieval_data["candidates"]
        print(f"      Metric Goal: {metric_dir.upper()}")
        print(f"      Found {len(candidates)} candidates.")

        self.log_step("Strategy Design", 
                      f"Found {len(candidates)} potential strategies. Metric goal: **{metric_dir}**.", 
                      "🧠")

        # --- 3. TOURNAMENT ---
        print("[3/5] Candidate Tournament...")
        
        # FIX: Increase Timeout to 30 minutes (1800s)
        CANDIDATE_TIMEOUT = 1800 
        
        best_candidate = None
        best_code = None
        
        is_minimizing = (metric_dir == "minimize")
        
        if is_minimizing:
            best_score = float('inf')
        else:
            best_score = -float('inf')
        
        for i, cand in enumerate(candidates):
            cand_name = cand['model_name']
            cand_reasoning = cand.get('reasoning', 'N/A')
            
            print(f"      Evaluating Candidate {i+1}: {cand_name}...")
            
            # Log the specific plan for this candidate
            self.log_step(f"Design: Candidate {i+1}", 
                          f"**Model:** {cand_name}\n**Reasoning:** {cand_reasoning}\n**Library:** {cand['library']}", 
                          "📝")

            path = self.output_dir / f"candidate_{i}.py"
            code = asyncio.run(generate_candidate_script(
                cand, modality_info, metadata, self.prepared_public, self.seed
            ))
            path.write_text(code)
            
            self.log_step(f"Implementation: Candidate {i+1}", f"Generated training script for {cand_name}.", "💻", code=code)

            # PASS INCREASED TIMEOUT HERE
            score = self.execute_candidate_robust(path, timeout=CANDIDATE_TIMEOUT)
            print(f"      -> Score: {score}")
            
            self.log_step(f"Evaluation: Candidate {i+1}", f"Training finished. **Score:** {score}", "📊")
            
            # SELECTION LOGIC
            if score is not None:
                if best_candidate is None:
                    best_score = score
                    best_candidate = cand
                    best_code = path.read_text() # Capture fixed code
                    print(f"      [NEW LEADER] Score: {best_score}")
                else:
                    if is_minimizing:
                        better = score < best_score
                    else:
                        better = score > best_score
                        
                    if better:
                        print(f"      [NEW LEADER] {score} is better than {best_score}")
                        best_score = score
                        best_candidate = cand
                        best_code = path.read_text()

        if not best_candidate:
            print("[CRITICAL] All candidates failed. Falling back to Candidate 0.")
            self.log_step("Critical Failure", "All candidates failed to return a score. Falling back to Candidate 0 default.", "❌")
            best_candidate = candidates[0]
            try:
                best_code = self.output_dir.joinpath("candidate_0.py").read_text()
            except:
                best_code = asyncio.run(generate_candidate_script(best_candidate, modality_info, metadata, self.prepared_public, self.seed))

        print(f"[WINNER] {best_candidate['model_name']} (Score: {best_score})")
        self.log_step("Tournament Winner", f"Selected strategy: **{best_candidate['model_name']}** with score {best_score}", "🏆")

        # ------------------------------------------------------------------
        # PHASE 2: REFINEMENT (MLE-STAR Implementation)
        # ------------------------------------------------------------------
        print("[3.5/5] Running Refinement Loop (MLE-STAR)...")
        from refiner import propose_ablations, propose_refinements, apply_refinement_llm
        
        self.log_step("Refinement Analysis", "Analyzing the winning code for potential hyperparameter ablations...", "🔬")
        ablations = asyncio.run(propose_ablations(best_code, modality_info['task_type'], metric_dir))
        
        if ablations:
            target = ablations[0] 
            print(f"      Refining Target: {target['component_name']}...")
            self.log_step("Refinement Proposal", f"Identified target: **{target['component_name']}**.\nReasoning: {target.get('reasoning', '')}", "💡")
            
            variations = asyncio.run(propose_refinements(target))
            
            for var in variations:
                print(f"      Testing Variation: {var['variant_name']} ({var['instruction']})...")
                self.log_step(f"Refinement: {var['variant_name']}", f"Applying instruction: {var['instruction']}", "🧪")
                
                refined_code = asyncio.run(apply_refinement_llm(best_code, var['instruction']))
                
                if refined_code:
                    script_name = f"refine_{var['variant_name']}.py"
                    path = self.output_dir / script_name
                    path.write_text(refined_code)
                    
                    # PASS INCREASED TIMEOUT HERE AS WELL
                    score = self.execute_candidate_robust(path, timeout=CANDIDATE_TIMEOUT)
                    print(f"      -> Score: {score}")
                    
                    if score is not None:
                        if is_minimizing:
                            better = score < best_score
                        else:
                            better = score > best_score
                            
                        if better:
                            print(f"      [IMPROVEMENT] New Best Score: {score}")
                            self.log_step("Refinement Success", f"Refinement improved score to {score}.", "✅")
                            best_score = score
                            best_code = refined_code 
                            best_candidate["model_name"] += f" ({var['variant_name']})"
                        else:
                            self.log_step("Refinement Result", f"Refinement did not improve score ({score}).", "📉")
        else:
            print("      [INFO] No obvious refinements found. Proceeding.")
            self.log_step("Refinement", "No high-confidence refinements found.", "⏩")

        # ------------------------------------------------------------------
        # PHASE 3: FINAL TRAIN
        # ------------------------------------------------------------------
        print("[4/5] Training Final Model on Full Data...")
        self.log_step("Final Production Build", "Generating final training script for full dataset training.", "🏭")
        final_path = self.output_dir / "train.py"
        
        if best_code is None: best_code = ""
        
        final_code = asyncio.run(generate_final_script(
            best_candidate, best_code, modality_info, metadata, self.prepared_public, self.seed
        ))
        final_path.write_text(final_code)
        
        self.log_step("Final Execution", "Running final training script...", "🚀", code=final_code)
        
        if self.run_training_script_robust(final_path, timeout=86400):
            print("[5/5] Grading...")
            self.log_step("Grading", "Validating submission file against test set...", "🎓")
            score = self.grade_submission()
            self.log_step("Completion", f"Final Score: **{score}**", "🏁")
        else:
            print("[FAIL] Final training failed.")
            self.log_step("Failure", "Final training failed to produce a valid submission.", "💀")

    # ------------------------------------------------------------------
    # EXECUTION HELPERS (ROBUST & STREAMING)
    # ------------------------------------------------------------------
    def execute_candidate_robust(self, script_path, timeout=600):
        # Allow up to 10 repair attempts per candidate
        max_retries = 10

        for attempt in range(max_retries):
            start_time = time.time()
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
                        self.log_step("Execution Warning", "Timeout reached during candidate execution.", "⏱️")
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
                self.log_step(f"Crash Detected (Attempt {attempt+1})", "Script crashed. Analyzing logs...", "💥", output=full_log[-1000:])

                # A. Dependency Fix?
                if self._handle_execution_error(full_log):
                    print(f"      [INFO] Environment fixed. Retrying...")
                    self.log_step("Recovery", "Environment dependency fixed. Retrying...", "🔄")
                    continue
                
                # B. AI Code Fix? (Enable this!)
                print(f"      [INFO] Applying AI Code Fix...")
                try:
                    current_code = script_path.read_text()
                    # Call the fixer we imported
                    fixed_code = asyncio.run(fix_training_script_llm(current_code, full_log))
                    script_path.write_text(fixed_code)
                    self.log_step("AI Repair", "Applied LLM-based code fix to resolve crash.", "🚑", code=fixed_code)
                    continue # Retry with new code
                except Exception as e:
                    print(f"      [ERR] AI Fixer failed: {e}")
                    return None

            except Exception as e:
                print(f"      [ERR] Execution exception: {e}")
                return None
        
        print("      [ERR] Candidate failed after max retries.")
        self.log_step("Failure", "Candidate failed after maximum retries.", "❌")
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
                            self.log_step("Final Validation", "submission.csv generated and validated successfully.", "✅")
                            return True
                        else:
                            raise RuntimeError("CSV Validation Failed")
                    
                    # 2. Failure Handling
                    err_msg = "".join(full_log)
                    self.log_step(f"Final Train Crash (Attempt {attempt})", "Final training script crashed.", "💥", output=err_msg[-500:])
                    
                    # A. Dependency Fix?
                    if self._handle_execution_error(err_msg):
                        print(f"[INFO] Dependency fixed. Retrying...")
                        continue 
                    
                    # B. Code Logic Fix (LLM)
                    print(f"[WARN] Attempt {attempt} failed. Retrying with AI Fix...")
                    new_code = asyncio.run(fix_training_script_llm(script_path.read_text(), err_msg))
                    script_path.write_text(new_code)
                    self.log_step("Final Repair", "Applying AI repair to final script.", "🚑")
                    
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

    # This now means NUMBER OF SEEDS, not a single seed value.
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Number of seeds to run. Example: --seed 3 runs seeds 42, 43, 44."
    )
    args = parser.parse_args()

    # Build seed list 42, 43, 44, ...
    base_seed = 43
    seeds_to_run = [base_seed + i for i in range(args.seed)]
    print(f"[INFO] Running seeds: {seeds_to_run}")

    # Track all run directories for appending to JSONL
    all_run_dirs = []

    for s in seeds_to_run:
        print(f"\n==============================")
        print(f"🚀 Running agent with seed {s}")
        print(f"==============================\n")

        agent = MLEAgent(args.competition, args.output, s)
        agent.run()

        # Find latest run directory created by this seed
        run_dirs = sorted(Path(args.output).glob("*"), key=lambda p: p.stat().st_mtime)
        latest_run = run_dirs[-1]
        all_run_dirs.append(latest_run)

    # ------------------------------------------------------------
    # Append all submissions from all seeds into ONE JSONL file
    # ------------------------------------------------------------
    jsonl_path = Path(args.output) / "submissions.jsonl"
    print(f"[INFO] Appending submission entries to: {jsonl_path}")

    with open(jsonl_path, "a") as f:
        for run_dir in all_run_dirs:
            submission_path = run_dir / "submission.csv"
            record = {
                "competition_id": args.competition,
                "submission_path": str(submission_path)
            }
            f.write(json.dumps(record) + "\n")

    print(f"[INFO] Done. Appended {len(all_run_dirs)} submissions.")