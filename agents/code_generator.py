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
    You are an advanced ML engineering agent (Resource-Constrained).
    Your job is to generate a FULL, SELF-CONTAINED Python training script.

    DATASET LOCATION: "{dataset_dir}"
    METADATA & PROFILING: {json.dumps(metadata, indent=2)}

    TASK:
    - Modality: {modality}
    - Type: {task_type}
    - Target: {target_col}
    - Classes: {classes}

    ### 1. MODEL SELECTION LOGIC (ARCHITECT PHASE):
    Analyze `metadata.complexity` and `dataset_size_mb` to choose the most efficient model.
    **Constraint: CPU-Only Training.**
    
    *   **IF IMAGE**:
        - **Task: Classification**:
            - Low Resolution (<64x64): Build simple Custom CNN.
            - High Resolution: Use `torchvision.models.mobilenet_v3_small(pretrained=True)`.
        - **Task: Regression / Image-to-Image (Denoising)**:
            - **CRITICAL**: Do NOT use a classifier backbone. Build a simple **Autoencoder** or **U-Net** (Conv2d -> ReLU -> MaxPool -> Upsample -> Conv2d). Output must match input shape.
    
    *   **IF TEXT**:
        - **Task: Classification**:
            - Short Text (<100 words): Use `sklearn` TfidfVectorizer + LogisticRegression.
            - Long Text: Use `DistilBERT` (Freeze body, train head) or LSTM.
        - **Task: Seq2Seq (Normalization/Translation)**:
            - Use `transformers.T5ForConditionalGeneration` (t5-small) + `T5Tokenizer`.
            - **Constraint**: Use `max_length=64` to keep CPU memory low.
    
    *   **IF AUDIO**:
        - Use `torchaudio`. Convert waveform to MelSpectrogram.
        - Model: Simple 2D CNN (treat Spectrogram as single-channel image).
    
    *   **IF TABULAR**:
        - Small (<10k rows): `RandomForestClassifier` / `Regressor`.
        - Large (>10k rows): `LightGBM` (Force `n_jobs=1` for stability if needed).

    ### 2. CRITICAL RESOURCE CONSTRAINTS (CPU MODE):
    - **DATA SUBSETTING**: We are testing on CPU. You MUST load/use only **5%** of the training data.
        - Example: `train_df = train_df.sample(frac=0.05, random_state={seed})`
    - **EPOCH LIMIT**: Train for a MAXIMUM of **3 epochs**.

    ### 3. LIBRARY BEST PRACTICES:
    - **Optimizers**: ALWAYS use `torch.optim.AdamW` (not transformers.AdamW).
    - **DataLoaders**: ALWAYS set `num_workers=0`. NO Multiprocessing.
    - **Inference (Critical)**: 
        - Multi-Class Classification: Apply `softmax(dim=1)` to outputs.
        - Binary/Multi-Label: Apply `sigmoid`.
        - Regression / Image-to-Image: **No activation** (Raw values) or `sigmoid` if normalized to 0-1.

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
