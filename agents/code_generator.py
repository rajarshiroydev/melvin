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
    You are an advanced ML engineering agent.
    Your job is to generate a FULL, SELF-CONTAINED Python training script.

    DATASET LOCATION:
    "{dataset_dir}"

    METADATA:
    {json.dumps(metadata, indent=2)}

    TASK:
    - Modality: {modality}
    - Type: {task_type}
    - Target: {target_col}
    - Classes: {classes}

    REQUIREMENTS:
    1.  **Reproducibility**: Set `random_state={seed}` and `torch.manual_seed({seed})` everywhere.
    2.  **Data Loading**: Use `pd.read_csv` with absolute paths.
    3.  **Submission**: Write `submission.csv` to current working directory (NOT dataset dir).
    4.  **No Placeholders**: Code must be runnable immediately.
    
    ### CRITICAL RESOURCE CONSTRAINTS (CPU MODE):
    5.  **DATA SUBSETTING**: We are testing on CPU. You MUST load/use only **5%** of the training data.
        - Example: `train_df = train_df.sample(frac=0.05, random_state={seed})`
        - Do NOT skip this. Full dataset training will crash.
    6.  **EPOCH LIMIT**: Train for a MAXIMUM of **3 epochs**.

    CRITICAL LIBRARY BEST PRACTICES (Avoid Deprecated APIs):
    - **Optimizers**: NEVER import `AdamW` from `transformers`. ALWAYS use `torch.optim.AdamW`.
    - **Transformers**: Use `AutoTokenizer` and `AutoModel...` classes where possible.
    - **DataLoaders**: ALWAYS set `num_workers=0` in `torch.utils.data.DataLoader`. Do NOT use multiprocessing (it causes crashes on macOS/Windows in this environment).
    - **Preprocessing**: If using `T5Tokenizer`, requires `sentencepiece`.
    - **Evaluation**: Ensure predictions match the sample_submission format exactly.
    - **Pandas vs Datasets**: 
        - Pandas DataFrames use `.columns` (list).
        - HuggingFace Datasets use `.column_names` (list).
        - DO NOT mix them up.

    SPECIFIC MODEL GUIDANCE:
    - TEXT CLASSIFICATION: TfidfVectorizer + LogisticRegression.
    - TABULAR: LightGBM or RandomForest.
    - SEQ2SEQ (Text Normalization): Use `transformers` (T5-small). 
        - Input: Text column. Target: Target column.
        - Use `Seq2SeqTrainer` or standard PyTorch training loop.
        - **Important**: Clean text data (handle NaNs) before tokenizing.
    - IMAGE: torchvision (ResNet18).
    - AUDIO: torchaudio + CNN.

    OUTPUT:
    - Return ONLY valid Python code. No markdown backticks.
    """

    response = completion(
        model="gemini/gemini-2.5-flash",  # Switched to Flash for faster code generation
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw_script = response["choices"][0]["message"]["content"]

    # Clean markdown
    if raw_script.strip().startswith("```"):
        raw_script = raw_script.replace("```python", "").replace("```", "").strip()

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
    If the error involves "DataLoader worker exited" or "multiprocessing", CHANGE `num_workers` to 0.
    Fix logic errors (e.g., pandas df.column_names -> df.columns).
    Fix import errors (e.g., deprecated APIs).
    Fix shape mismatches.
    DO NOT remove the logic that saves submission.csv.
    Ensure `train_df.sample(frac=0.05)` or equivalent subsetting logic is preserved to keep it fast.
    OUTPUT:
    Return ONLY valid Python code. No markdown backticks.
    """
    response = completion(
        model="gemini/gemini-2.5-pro",
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
