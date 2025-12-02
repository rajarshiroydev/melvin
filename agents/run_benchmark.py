import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure we can import the orchestrator from the same directory
sys.path.append(str(Path(__file__).parent))
from orchestrator import MLEAgent

# -----------------------------------------------------------------------------
# LOGGER: CAPTURE OUTPUT TO FILE AND CONSOLE
# -----------------------------------------------------------------------------
class Logger(object):
    def __init__(self, filename="benchmark_logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')  # Append mode

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout and stderr
sys.stdout = Logger()
sys.stderr = sys.stdout

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# The 4 datasets required by the assignment except whale
TARGET_DATASETS = [
    "the-icml-2013-whale-challenge-right-whale-redux",
    "text-normalization-challenge-english-language",
    "spooky-author-identification",
    "tabular-playground-series-may-2022",
    "siim-isic-melanoma-classification",
]

# Seeds to repeat evaluation
SEEDS = [42, 43, 44]

# Metric Thresholds logic (Calculated from Leaderboards)
MEDAL_THRESHOLDS = {
    "spooky-author-identification": {
        "metric": "LogLoss",
        "direction": "min",
        "threshold": 0.3752  # Rank 124 (Top 10% of 1245 teams)
    },
    "tabular-playground-series-may-2022": {
        "metric": "ROC-AUC",
        "direction": "max",
        "threshold": 0.9982  # Rank 117 (Top 10% of 1170 teams)
    },
    "text-normalization-challenge-english-language": {
        "metric": "Accuracy/StringMatch",
        "direction": "max",
        "threshold": 0.9855  # Rank 100 (Top 100 of 260 teams)
    },
    "siim-isic-melanoma-classification": {
        "metric": "ROC-AUC",
        "direction": "max",
        "threshold": 0.9205  # Rank 330 (Top 10% of 3308 teams)
    },
    "the-icml-2013-whale-challenge-right-whale-redux": {
        "metric": "ROC-AUC",
        "direction": "max",
        "threshold": 0.90521
    }
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def check_medal(dataset_id, score):
    """
    Returns True if the score qualifies for a medal based on thresholds.
    """
    if score is None:
        return False
    
    config = MEDAL_THRESHOLDS.get(dataset_id)
    if not config:
        # Fallback if dataset not in dict: Any valid score counts
        return True 

    threshold = config["threshold"]
    
    if config["direction"] == "max":
        return score >= threshold
    else:
        return score <= threshold

def calculate_sem(data):
    """
    Calculates Mean and Standard Error of the Mean.
    SEM = std_dev / sqrt(n)
    """
    if not data:
        return 0.0, 0.0
    
    arr = np.array(data, dtype=float)
    mean = np.mean(arr)
    if len(arr) > 1:
        sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
    else:
        sem = 0.0
        
    return mean * 100, sem * 100  # Return as percentages

# -----------------------------------------------------------------------------
# MAIN BENCHMARK LOOP
# -----------------------------------------------------------------------------

def run_benchmark():
    # Setup Paths
    current_file = Path(__file__).resolve()
    # Default output dir: Hexo/melvin/runs
    default_runs_dir = current_file.parent.parent / "runs"
    
    print("="*60)
    print(f"STARTING MLE-BENCH LITE EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(TARGET_DATASETS)}")
    print(f"Seeds per dataset: {len(SEEDS)}")
    print("="*60)

    final_results = {}

    for dataset_id in TARGET_DATASETS:
        print(f"\n\n>>> PROCESSING DATASET: {dataset_id}")
        print(f"    Threshold: {MEDAL_THRESHOLDS[dataset_id]['threshold']} ({MEDAL_THRESHOLDS[dataset_id]['direction']})")
        dataset_medals = [] # Store 1.0 (Medal) or 0.0 (No Medal) per seed
        
        for seed in SEEDS:
            print(f"\n--- Running Seed {seed} ---")
            
            try:
                # 1. Initialize Agent
                agent = MLEAgent(dataset_id, default_runs_dir, seed=seed)
                
                # 2. Run the full pipeline (Prep -> Gen -> Train -> Grade)
                # Orchestrator.run() runs everything but returns None.
                # However, it saves a grading report we can read.
                agent.run()
                
                # 3. Retrieve Score from the JSON report file
                # The orchestrator saves: output_dir / f"grading_report_seed_{seed}.json"
                report_path = agent.output_dir / f"grading_report_seed_{seed}.json"
                
                score = None
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        data = json.load(f)
                        score = data.get("score")
                
                # 4. Check Medal Status
                is_medal = check_medal(dataset_id, score)
                medal_int = 1.0 if is_medal else 0.0
                dataset_medals.append(medal_int)
                
                status_str = "MEDAL 🏅" if is_medal else "NO MEDAL ❌"
                print(f"Seed {seed} Result: Score={score} [{status_str}]")

            except Exception as e:
                print(f"[CRITICAL FAILURE] Seed {seed} crashed: {e}")
                import traceback
                traceback.print_exc()
                dataset_medals.append(0.0) # Crash = No Medal

        # Calculate Statistics for this Dataset
        mean_pct, sem_pct = calculate_sem(dataset_medals)
        final_results[dataset_id] = f"{mean_pct:.1f}% ± {sem_pct:.1f}%"

    # -----------------------------------------------------------------------------
    # FINAL REPORT GENERATION
    # -----------------------------------------------------------------------------
    print("\n\n")
    print("="*80)
    print(f"{'DATASET':<50} | {'ANY MEDAL (%) MEAN ± SEM':<25}")
    print("-" * 80)
    
    for ds, res in final_results.items():
        print(f"{ds:<50} | {res:<25}")
    
    print("="*80)
    print("Evaluation Complete.")

if __name__ == "__main__":
    run_benchmark()