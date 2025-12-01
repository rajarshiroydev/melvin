import os
import json
from pathlib import Path
from litellm import completion

# ---------------------------------------------------------
# PROMPT: Candidate Generator (Strict Subsampling)
# ---------------------------------------------------------
CANDIDATE_PROMPT = """
You are an ML Engineer implementing a specific strategy found via research.

STRATEGY: {model_name}
LIBRARY: {library}
TIPS: {implementation_tips}

DATASET: {dataset_dir}
TARGET: {target_col}
TASK: {task_type}
METADATA: {metadata_json}

Your goal: Write a Python Training Script to EVALUATE this specific strategy.

CRITICAL CONSTRAINTS (SPEED IS #1):
1. **SUBSAMPLING IS MANDATORY**: 
   - You MUST load ONLY the first 20,000 rows or sample 20% of data (whichever is smaller).
   - `df = pd.read_csv(..., nrows=20000)` or `df = df.sample(n=20000)`.
   - DO NOT train on the full dataset. This is a quick viability test.
2. **VALIDATION**:
   - Use a simple 80/20 Holdout split.
   - Calculate the metric on the 20% holdout.
3. **OUTPUT FORMAT**:
   - The script MUST print the final score on the LAST LINE exactly like this:
     `FINAL_SCORE: 0.1234`
   - Do NOT generate a submission.csv yet. We are just testing the model.

BOILERPLATE:
- Set random seeds ({seed}).
- Handle missing values (SimpleImputer) and Categoricals (LabelEncoder) robustly.
- **IMPORTS**: Just import the libraries you need (e.g. `import catboost`). 
  - **DO NOT** try to install them with `subprocess` or `pip`. 
  - If a library is missing, the environment will handle it automatically.

Return ONLY valid Python code.
"""

# ---------------------------------------------------------
# PROMPT: Final Full-Scale Trainer
# ---------------------------------------------------------
FINAL_TRAIN_PROMPT = """
You are an ML Engineer. You have identified the WINNING strategy.
Now, write the FINAL PRODUCTION SCRIPT to train on the FULL DATASET and generate the submission.

WINNING STRATEGY: {model_name}
CODE REFERENCE (The prototype that worked):
{prototype_code}

INSTRUCTIONS:
1. **FULL DATA**: Load the ENTIRE dataset. Do NOT subsample.
2. **ROBUSTNESS**: Add error handling.
3. **SUBMISSION**:
   - Predict on `test.csv`.
   - Ensure ID column types match `sample_submission.csv`.
   - Save to `submission.csv`.
   - Check if multiclass probabilities sum to 1.
4. **TIME LIMIT**:
   - Implement a time check. If training > 45 mins, stop and predict with current weights.
5. **IMPORTS**: Just import what you need. Do not try to install packages.

Return ONLY valid Python code.
"""

async def generate_candidate_script(candidate_info, modality_info, metadata, dataset_dir, seed=42):
    prompt = CANDIDATE_PROMPT.format(
        model_name=candidate_info["model_name"],
        library=candidate_info["library"],
        implementation_tips=candidate_info.get("implementation_tips", ""),
        dataset_dir=str(dataset_dir),
        target_col=modality_info.get("target_col"),
        task_type=modality_info.get("task_type"),
        metadata_json=json.dumps(metadata),
        seed=seed
    )

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    raw = response["choices"][0]["message"]["content"]
    raw = raw.replace("```python", "").replace("```", "").strip()
    return raw

async def generate_final_script(best_candidate, prototype_code, modality_info, metadata, dataset_dir, seed=42):
    prompt = FINAL_TRAIN_PROMPT.format(
        model_name=best_candidate["model_name"],
        prototype_code=prototype_code,
        dataset_dir=str(dataset_dir),
        target_col=modality_info.get("target_col"),
        task_type=modality_info.get("task_type"),
        metadata_json=json.dumps(metadata),
        seed=seed
    )

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    raw = response["choices"][0]["message"]["content"]
    return raw.replace("```python", "").replace("```", "").strip()


# =========================================================
#  🔥 INSERTED BELOW: OLD FIXER LOGIC (PERFECT COPY)
# =========================================================

from langgraph.graph import StateGraph

class FixState(dict):
    script: str
    error_log: str
    fixed_script: str

# -----------------------------
# FIXER NODE
# -----------------------------
def llm_script_fixer(state: FixState):
    script = state["script"]
    error_log = state["error_log"]

    prompt = f"""
    You are an expert Python debugger for Machine Learning scripts.
    The following script crashed during execution.

    YOUR TASK:
    Fix the code to resolve the specific error found in the logs.
    Return the FULL, corrected script.

    BROKEN SCRIPT:
    ```python
    {script}
    ```
    ERROR LOG:
    {error_log}

    ANALYSIS:
    Identify the line causing the error.
    - If error is "DataLoader worker exited" or "multiprocessing": Force `num_workers=0`.
    - If error is "CUDA out of memory" or device issue: Force `device='cpu'`.
    - If error involves shape mismatch in Linear layers: Re-calculate the input features dynamically.
    - If error involves T5 or Seq2Seq tensor shapes: ensure padding & max_length small (e.g., 64).

    Ensure train_df.sample() logic is preserved if present.

    OUTPUT:
    Return ONLY valid Python code. No markdown fences.
    """

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    fixed_script = response["choices"][0]["message"]["content"]
    fixed_script = fixed_script.replace("```python", "").replace("```", "").strip()
    state["fixed_script"] = fixed_script
    return state

# -----------------------------
# FIXER GRAPH
# -----------------------------
def build_fix_graph():
    graph = StateGraph(FixState)
    graph.add_node("code_fixer", llm_script_fixer)
    graph.set_entry_point("code_fixer")
    graph.set_finish_point("code_fixer")
    return graph.compile()

# -----------------------------
# PUBLIC FIXER API
# -----------------------------
async def fix_training_script_llm(current_script: str, error_log: str):
    graph = build_fix_graph()
    final = await graph.ainvoke({"script": current_script, "error_log": error_log})
    return final["fixed_script"]
