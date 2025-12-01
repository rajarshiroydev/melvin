import os
import json
from pathlib import Path

from litellm import completion
from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# State Definitions
# ---------------------------------------------------------
class CodegenState(dict):
    modality: str
    task_type: str
    target_col: str
    classes: list
    metadata: dict
    dataset_dir: str
    seed: int
    script: str
    hardware: dict

class FixState(dict):
    script: str
    error_log: str
    fixed_script: str


# ---------------------------------------------------------
# NODE: Script Generator (Creator)
# ---------------------------------------------------------
def llm_script_generator(state: CodegenState):
    modality = state["modality"]
    task_type = state["task_type"]
    target_col = state["target_col"]
    classes = state.get("classes", [])
    metadata = state["metadata"]
    dataset_dir = state["dataset_dir"]
    seed = state.get("seed", 42)

    prompt = f"""
    You are an advanced ML engineering agent optimizing for SPEED and VALIDITY.
    Your job is to generate a FULL, SELF-CONTAINED Python training script that runs in <15 MINUTES.

    DATASET LOCATION: "{dataset_dir}"
    
    TASK DESCRIPTION (from README):
    \"\"\"{metadata.get("description", "No description provided.")}\"\"\"

    METADATA & PROFILING:
    {json.dumps({k: v for k, v in metadata.items() if k != "description"}, indent=2)}

    TASK:
    - Modality: {modality}
    - Type: {task_type}
    - Target: {target_col}
    - Classes: {classes}
    - Total Rows: {metadata.get('num_train_rows', 'Unknown')}

    HARDWARE INFO (Use to set batch size):
    {json.dumps(state["hardware"], indent=2)}

    DATA SUBSAMPLING REQUIREMENT (MANDATORY):
    - You MUST explicitly execute:
        train_df = train_df.sample(frac=0.05, random_state={seed})

    immediately after loading the training data.
    - Perform this BEFORE any preprocessing, splitting, tokenization, encoding, or DataLoader creation.
    - Every downstream component MUST use only this reduced DataFrame.
    - You MUST NOT reload or reference the full dataset anywhere else in the script.

    ### 0. SPEED & MEMORY RULES (CRITICAL FOR DEADLINE):
    - **EPOCHS**: Train for **ONLY 1 EPOCH**. (Strict Limit).
    - **PRECISION**: You MUST use Mixed Precision (FP16) via `torch.cuda.amp.GradScaler`.
    - **BATCH SIZE**: Maximize this! Use 32+ for images, 64+ for text.
    - **EVALUATION**: Run validation ONLY at the end of the epoch.
    - **DATA**: Use 100% of the data (NO subsampling), but rely on 1 epoch + fast models to finish on time.

    ### 1. MODEL SELECTION (FAST ARCHITECTURES):
    Analyze `metadata.complexity` but bias towards speed.
    
    * **IF IMAGE**:
        - Use `resnet18` (pretrained). It is fast and robust.
        - Avoid ResNet50 or EfficientNet unless resolution is tiny (<32x32).
    
    * **IF TEXT**:
        - **Task: Classification**:
            - Use xgboost
        - **Task: Seq2Seq / Text-Norm**:
            - Use `t5-small` (CRITICAL: t5-base is too slow).
    
    * **IF AUDIO**:
        - Use the specific robust loading logic below (Train/Test csv + zip extraction).
        - Use a lightweight 1D CNN or `resnet18` on MelSpectrograms.
        - **Audio Loader Rules**: 
            1. Load train.csv from dataset_dir.
            2. Extract train2.zip/test2.zip to temp folders.
            3. Fallback: try `torchaudio` -> `librosa` -> `soundfile`.
            4. Skip corrupted files (return zeros) to prevent crashes.
    
    * **IF TABULAR**:
        - Use XGBoost.
        - Use `.fit(X_train, y_train)` minimal API.
        **NORMALIZATION RULES (DO NOT MISS):**
            - **Multi-Class Classification**: 
            1. Use `softmax(dim=1)`.
            2. `softmax` -> `div(sum)` to ensure row sum=1.0.

    ### 2. SEEDING REQUIREMENTS:
    - You MUST set all random seeds to `{seed}`.
    - `random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed})`

    ### 3. LIBRARY BEST PRACTICES:
    - **Optimizers**: NEVER import `AdamW` from `transformers`. ALWAYS use `torch.optim.AdamW`.
    - **DataLoaders**: Set `num_workers=4`, `pin_memory=True`.
    - **Safe Execution**: Wrap main logic in `if __name__ == "__main__":`.
    - **Imports**: Handle `xgboost`/`transformers` imports gracefully.

    ### 1. TRAINING LOOP STRATEGY (THE "TIME LIMIT" LOGIC):
    - **Standard**: Train for 1 Epoch.
    - **HUGE DATASET EXCEPTION (Important)**: 
      - If `len(train_loader) > 5000`:
      - You MUST implement a break: `if batch_idx >= 2000: break`
      - Print "Max steps reached (2000), stopping training."
      - This allows "seeing" the full dataset structure without timing out.

    ### 4. MANDATORY SUBMISSION LOGIC:
    - Generate `submission.csv`.
    - Load `sample_submission.csv` to ensure correct ID sorting.
    - **Multi-Class**: Use `softmax(dim=1)` so rows sum to 1.
    - **Binary**: Use `sigmoid`.
    - Save to current directory (`.`).
    - Print "Submission saved to submission.csv".

    OUTPUT:
    - Return ONLY valid Python code. No markdown backticks.
    """

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw_script = response["choices"][0]["message"]["content"]

    # Stronger Markdown Cleanup
    raw_script = raw_script.strip()
    if raw_script.startswith("```"):
        raw_script = raw_script.split("\n", 1)[1]
    if raw_script.endswith("```"):
        raw_script = raw_script.rsplit("\n", 1)[0]
    raw_script = raw_script.replace("```python", "").replace("```", "")

    state["script"] = raw_script
    return state


# ---------------------------------------------------------
# NODE: Script Fixer (Repairer)
# ---------------------------------------------------------
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
    ERROR LOG:
    {error_log}
    ANALYSIS:
    Identify the line causing the error.
    - If error is "DataLoader worker exited" or "multiprocessing": Force `num_workers=0`.
    - If error is "CUDA out of memory" or device issue: Force `device='cpu'`.
    - If error involves shape mismatch in Linear layers: Re-calculate the input features (flatten size) dynamically.
    - If error involves "T5" or "Seq2Seq" shapes: Ensure padding and max_length are small (e.g., 64).
    
    Ensure `train_df.sample(frac=0.05)` logic is preserved.
    
    OUTPUT:
    Return ONLY valid Python code. No markdown backticks.
    """
    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    fixed_script = response["choices"][0]["message"]["content"]
    # Clean markdown
    if fixed_script.strip().startswith(""):
        fixed_script = fixed_script.replace("python", "").replace("```", "").strip()
    state["fixed_script"] = fixed_script
    return state


def llm_reason_about_step(step, info, model="gemini/gemini-2.5-flash"):
    """
    Generates short reflective reasoning for logs:
    - how the agent interprets the task/modality
    - why it chose a strategy
    - what it will try next time
    
    Output is SHORT and SAFE (no chain-of-thought, only conclusions).
    """
    prompt = f"""
    You are an ML planning agent. Produce a brief but deep explanation 
    (no chain-of-thought) of the following:

    STEP: {step}
    INFO: {json.dumps(info, indent=2)}

    Provide:
    1. How did you decide the task & modality of the dataset?
    2. Why the chosen strategy makes sense for this task?
    3. How the agent would self-improve next time?
    """

    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Reasoning unavailable due to LLM error."


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------
def build_gen_graph():
    graph = StateGraph(CodegenState)
    graph.add_node("code_generator", llm_script_generator)
    graph.set_entry_point("code_generator")
    graph.set_finish_point("code_generator")
    return graph.compile()


def build_fix_graph():
    graph = StateGraph(FixState)
    graph.add_node("code_fixer", llm_script_fixer)
    graph.set_entry_point("code_fixer")
    graph.set_finish_point("code_fixer")
    return graph.compile()


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
async def generate_training_script_llm(
    modality,
    task_type,
    target_col,
    classes,
    metadata,
    dataset_dir,
    output_path: Path,
    seed: int = 42,
    hardware=None,
):

    dataset_dir_str = str(dataset_dir.resolve())

    graph = build_gen_graph()

    final_state = await graph.ainvoke(
        {
            "modality": modality,
            "task_type": task_type,
            "target_col": target_col,
            "classes": classes,
            "metadata": metadata,
            "dataset_dir": dataset_dir_str,
            "seed": seed,
            "hardware": hardware or {},   # <-- NEW
        }
    )


    script = final_state["script"]
    output_path.write_text(script)
    return output_path


# ---------------------------------------------------------
# Public API: Fix
# ---------------------------------------------------------
async def fix_training_script_llm(current_script: str, error_log: str):
    graph = build_fix_graph()
    final_state = await graph.ainvoke(
        {"script": current_script, "error_log": error_log}
    )
    return final_state["fixed_script"]
