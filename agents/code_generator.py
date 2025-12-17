import os
import json
from pathlib import Path
from litellm import completion
from langgraph.graph import StateGraph

# env variable
os.environ['GROQ_API_KEY']

# ---------------------------------------------------------
# PROMPT: Candidate Generator
# ---------------------------------------------------------
CANDIDATE_PROMPT = """
    You are a Senior Kaggle Grandmaster implementing a specific strategy found via research.

    STRATEGY: {model_name}
    LIBRARY: {library}
    TIPS: {implementation_tips}
    STRATEGY: {model_name}
    LIBRARY: {library}
    TIPS: {implementation_tips}

    Use the following dataset paths. Do not change them.
    TRAIN_PATH = "{train_path}"
    TEST_PATH = "{test_path}"
    SAMPLE_SUBMISSION_PATH = "{sample_sub_path}"

    Data Schema:
    TRAIN_COLUMNS: {train_columns}
    TEST_COLUMNS: {test_columns}
    SUBMISSION_COLUMNS: {submission_columns}
    - Very important to understand the column schema of submission as
    sometimes is is not id and rather a combination of other column names.
    
    Hardware Constraints: 
    DETECTED GPU VRAM: {vram_gb} GB
    GPU NAME: {gpu_name}
    
    Maximize Utilization: Increase batch_size until near VRAM limit.
    - Ref: 16GB VRAM -> Batch 16
    - Ref: 24GB VRAM -> Batch 32
    - Ref: 80GB VRAM -> Batch 128 or more

    TARGET: {target_col}
    TASK: {task_type}
    METADATA: {metadata_json}

    UNSTRUCTURED DATA HANDLING:
    {unstructured_hint}

    Your goal: Write a Python Training Script to EVALUATE this specific strategy.

    CRITICAL CONSTRAINTS (SPEED IS #1):
    1. **SUBSAMPLING IS MANDATORY**: 
    - You MUST load ONLY the first 5000 rows or sample 5% of data.
    - `df = pd.read_csv(TRAIN_PATH, nrows=5000)`
    2. **VALIDATION**:
    - Use a 80/20 Holdout split.
    3. **Efficiency**: Implement Early Stopping (patience=3).
    4. **OUTPUT FORMAT**:
    - Print `FINAL_SCORE: 0.1234` on the last line.
    5. **EARLY STOPPING (MANDATORY)**: Implement early stopping with `patience=3` 
    monitoring validation loss/metric.
    
    BOILERPLATE:
    - Set random seeds ({seed}).
    - **IMPORTS**: Import what you need. DO NOT install.

    Return ONLY valid Python code.
    """

# ---------------------------------------------------------
# PROMPT: Final Full-Scale Trainer
# ---------------------------------------------------------
FINAL_TRAIN_PROMPT = """
    You are an Senior Kaggle Grandmaster.
    You have a prototype which performed the best.
    {prototype_code}
    Now, write the final training script to train on the full dataset and generate the submission.
    Use the exact same logic as the prototype code.
    The only changes should be training on the full dataset and the predicting on the test set.

    Use the following dataset paths. Do not change them.
    TRAIN_PATH = "{train_path}"
    TEST_PATH = "{test_path}"
    SAMPLE_SUBMISSION_PATH = "{sample_sub_path}"

    Data Schema:
    TRAIN_COLUMNS: {train_columns}
    TEST_COLUMNS: {test_columns}
    SUBMISSION_COLUMNS: {submission_columns}
    - Very important to understand the column schema of submission as
    sometimes is is not id and rather a combination of other column names.
    
    Hardware Constraints:
    DETECTED GPU VRAM: {vram_gb} GB
    GPU NAME: {gpu_name}

    Maximize Utilization: Increase batch_size until near VRAM limit.
    - Ref: 16GB VRAM -> Batch 16
    - Ref: 24GB VRAM -> Batch 32
    - Ref: 80GB VRAM -> Batch 128 or more
    
    Instructions:
        INHERIT SETUP: Copy imports and setup from REFERENCE.
        DATA LOAD: Load the ENTIRE dataset using `TRAIN_PATH`.

        **CRITICAL: RESUMABLE EXECUTION / CRASH RECOVERY**:
        - Before starting training, check: `if os.path.exists('best_model.pt') or os.path.exists('emergency_model.pt'):`
        - If a saved model exists: **SKIP TRAINING ENTIRELY**. Load the weights, print "Checkpoint found, skipping training...", and proceed directly to the SUBMISSION step.
        - This allows the script to recover instantly if it crashes during the prediction phase.
        
        **TIME LIMIT & FLOW CONTROL (CRITICAL)**:
        - **TIME LIMIT: 30 MINUTES (1800 seconds).**
        - Inside the loop, check: `if time.time() - start_time > 1800: break`.
        - **IF LIMIT REACHED**:
            1. **STOP TRAINING IMMEDIATELY**.
            2. **SAVE MODEL** to 'emergency_model.pt'.
            3. **SKIP EVALUATION/VALIDATION** if time is up.
            4. **PROCEED DIRECTLY TO PREDICTION**.

        VALIDATION STRATEGY: 90% Train, 10% Validation. But keep validation checking frequent
        so that unnecessary training can be stopped.
        TRAINING ROBUSTNESS: Early Stopping (patience=3), Model Checkpointing.
            
        SUBMISSION:
            Predict on `TEST_PATH`.
            **CRITICAL**: Load `SAMPLE_SUBMISSION_PATH` as the SOURCE OF TRUTH.
            - The submission file MUST have exactly the same number of rows and same order as `SAMPLE_SUBMISSION_PATH`.
            - Do not rely solely on the generated test file count.
            Ensure your submission.csv matches {sample_sub_path} format exactly.
            Save to submission.csv.
            
    Return ONLY valid Python code.
    """

def get_unstructured_hint(metadata):
    if metadata.get("has_filepath_col"):
        return """
        ** ATTENTION: UNSTRUCTURED DATA DETECTED **
        - The CSVs contain a 'filepath' column pointing to raw media files.
        - YOU MUST IMPLEMENT A CUSTOM DATASET CLASS.
        - Load raw files inside `__getitem__`.
        - For Audio: use librosa/torchaudio. For Images: use PIL.

        ** CRITICAL FOR INFERENCE (TESTING) **:
        1. **SOURCE OF TRUTH**: Use `SAMPLE_SUBMISSION_PATH` to define the test set items, NOT `TEST_PATH`.
        2. **DYNAMIC PATHS**: 
           - Read `sample_submission.csv` to get the list of IDs/filenames.
           - Infer the directory from `TEST_PATH` (e.g. `test_dir = os.path.dirname(df_test['filepath'].iloc[0])`).
           - Construct filepaths dynamically: `path = os.path.join(test_dir, row['clip'])`.
        3. **MISSING FILES**:
           - If a file listed in sample_submission is missing from disk, YOU MUST NOT CRASH.
           - **Catch the Exception** and return a zero-array (silence/black image).
        """
    return ""


async def generate_candidate_script(candidate_info, modality_info, metadata, dataset_dir, seed=42, hardware_stats=None):
    if hardware_stats is None: hardware_stats = {"vram_gb": 16, "gpu_name": "Unknown"}

    # EXTRACT EXACT FILENAMES
    train_file = metadata.get("train_filename", "train.csv")
    test_file = metadata.get("test_filename", "test.csv")
    sample_sub_file = metadata.get("sample_submission_filename", "sample_submission.csv")
    
    # CONSTRUCT FULL PATHS
    train_path = Path(dataset_dir) / train_file
    test_path = Path(dataset_dir) / test_file
    sample_sub_path = Path(dataset_dir) / sample_sub_file

    prompt = CANDIDATE_PROMPT.format(
        model_name=candidate_info["model_name"],
        library=candidate_info["library"],
        implementation_tips=candidate_info.get("implementation_tips", ""),
        train_path=str(train_path),
        test_path=str(test_path),
        sample_sub_path=str(sample_sub_path),
        train_columns=metadata.get("train_columns"),
        test_columns=metadata.get("test_columns"),
        submission_columns=metadata.get("submission_columns"),
        dataset_dir=str(dataset_dir),
        target_col=modality_info.get("target_col"),
        task_type=modality_info.get("task_type"),
        unstructured_hint=get_unstructured_hint(metadata),
        metadata_json=json.dumps(metadata),
        seed=seed,
        vram_gb=hardware_stats.get("vram_gb", 0),
        gpu_name=hardware_stats.get("gpu_name", "CPU")
    )

    response = completion(
        model="deepseek/deepseek-chat",
        # api_key=os.getenv("GROQ_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    raw = response["choices"][0]["message"]["content"]
    raw = raw.replace("```python", "").replace("```", "").strip()
    return raw

async def generate_final_script(best_candidate, prototype_code, modality_info, metadata, dataset_dir, seed=42, hardware_stats=None):
    if hardware_stats is None: hardware_stats = {"vram_gb": 16, "gpu_name": "Unknown"}

    # EXTRACT EXACT FILENAMES
    train_file = metadata.get("train_filename", "train.csv")
    test_file = metadata.get("test_filename", "test.csv")
    sample_sub_file = metadata.get("sample_submission_filename", "sample_submission.csv")
    
    train_path = Path(dataset_dir) / train_file
    test_path = Path(dataset_dir) / test_file
    sample_sub_path = Path(dataset_dir) / sample_sub_file

    prompt = FINAL_TRAIN_PROMPT.format(
        model_name=best_candidate["model_name"],
        prototype_code=prototype_code,
        train_path=str(train_path),
        test_path=str(test_path),
        sample_sub_path=str(sample_sub_path),
        train_columns=metadata.get("train_columns"),
        test_columns=metadata.get("test_columns"),
        submission_columns=metadata.get("submission_columns"),
        dataset_dir=str(dataset_dir),
        target_col=modality_info.get("target_col"),
        task_type=modality_info.get("task_type"),
        metadata_json=json.dumps(metadata),
        unstructured_hint=get_unstructured_hint(metadata),
        seed=seed,
        vram_gb=hardware_stats.get("vram_gb", 0),
        gpu_name=hardware_stats.get("gpu_name", "CPU")
    )

    response = completion(
        model="deepseek/deepseek-chat",
        # api_key=os.getenv("GROQ_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    raw = response["choices"][0]["message"]["content"]
    return raw.replace("```python", "").replace("```", "").strip()


# =========================================================
#  FIXER WITH HISTORY MEMORY
# =========================================================

class FixState(dict):
    script: str
    error_log: str
    history: list
    fixed_script: str

def llm_script_fixer(state: FixState):
    script = state["script"]
    error_log = state["error_log"]
    history = state.get("history", [])

    history_text = ""
    if history:
        history_text = "\n\n".join([f"--- ATTEMPT {i+1} ERROR ---\n{err[:500]}..." for i, err in enumerate(history)])

    prompt = f"""
    You are an expert Python debugger. The script crashed.
    
    CRITICAL INSTRUCTION:
    We have already tried to fix this script {len(history)} times.
    
    PAST FAILURE HISTORY (DO NOT REPEAT THESE MISTAKES):
    {history_text}

    CURRENT ERROR LOG:
    {error_log}

    BROKEN SCRIPT:
    ```python
    {script}
    ```

    YOUR TASK:
    Fix the code to resolve the error.
    If you see "Expected 2D/3D got 4D" (LSTM/RNN errors):
       - Check your `collate_fn` or `__getitem__`.
       - You might be unsqueezing unnecessarily.
    If you see "CUDA OOM": Force CPU.
    
    RETURN ONLY THE FIXED PYTHON CODE.
    """

    response = completion(
        model="deepseek/deepseek-chat",
        # api_key=os.getenv("GROQ_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, 
    )

    fixed_script = response["choices"][0]["message"]["content"]
    fixed_script = fixed_script.replace("```python", "").replace("```", "").strip()
    state["fixed_script"] = fixed_script
    return state

def build_fix_graph():
    graph = StateGraph(FixState)
    graph.add_node("code_fixer", llm_script_fixer)
    graph.set_entry_point("code_fixer")
    graph.set_finish_point("code_fixer")
    return graph.compile()

async def fix_training_script_llm(current_script: str, error_log: str, history: list = None):
    if history is None: history = []
    graph = build_fix_graph()
    final = await graph.ainvoke({
        "script": current_script, 
        "error_log": error_log,
        "history": history
    })
    return final["fixed_script"]