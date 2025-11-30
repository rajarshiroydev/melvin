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
    You are an advanced ML engineering agent (High-Performance/GPU-Enabled).
    Your job is to generate a FULL, SELF-CONTAINED Python training script.

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

    HARDWARE INFO YOU MUST USE TO SET BATCH SIZE, MODEL SIZE, AND EPOCHS:
    {json.dumps(state["hardware"], indent=2)}

    Rules:
    - If GPU VRAM < 6GB → Use smallest model (ResNet18, T5-small) and batch_size ≤ 8
    - If GPU VRAM 6–12GB → Medium models allowed, batch_size ≤ 32
    - If GPU VRAM > 12GB → Larger models allowed, batch_size 32–128
    - If NO GPU: avoid heavy models, reduce batch_size drastically
    - If RAM < 8GB: avoid loading full dataset into memory
    - If dataset is >5M rows: reduce epochs to 1–3
    - Always decide epochs + batch size based on hardware profile

    SEEDING REQUIREMENTS:
    - You MUST set all random seeds for Python, NumPy, PyTorch (CPU & CUDA) using the seed `{seed}`.
    - Ensure deterministic behavior:
        import random, numpy as np, torch
        random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed})
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all({seed})
    - Ensure DataLoader uses:
        generator=torch.Generator().manual_seed({seed})



    ### 1. MODEL SELECTION LOGIC (ARCHITECT PHASE):
    Analyze `metadata.complexity` to choose a high-accuracy model.
    
    *   **IF IMAGE**:
        - **Task: Classification**:
            - Low Resolution (<64x64): Use `resnet18` or `resnet34` (modify first conv layer if needed).
            - High Resolution: Use `torchvision.models.resnet50(weights='DEFAULT')` or `efficientnet_b0`.
        - **Task: Regression / Image-to-Image**:
            - Build a **U-Net** architecture or deep Autoencoder with ResNet backbone.
    
    *   **IF TEXT**:
        - **Task: Classification**:
            - Use `bert-base-uncased` or `roberta-base`.
            - **Fine-Tuning**: Train the FULL model (do NOT freeze body).
        - **Task: Seq2Seq**:
            - Use `t5-base` or `bart-base`.
    
    *   **IF AUDIO**:
        - Use `torchaudio`. Convert to MelSpectrogram.
        - Backbone: ResNet34 modified for 1-channel input (Spectrogram).
    
    *   **IF TABULAR**:
        - Use XGBoost or LightGBM.
        - To ensure compatibility across environments, ALWAYS use the minimal .fit() API:
              model.fit(X_train, y_train)
          Do NOT use:
              - early_stopping_rounds
              - callbacks
              - eval_set
              - EarlyStopping()
              - custom objectives
        - If cross-validation is needed, manually split folds and call model.fit() inside each fold.
        - For binary classification, use model.predict_proba(test)[:, 1].
        - For multi-class classification, use model.predict_proba(test) and follow sample_submission.csv column order.

    ### 2. TRAINING CONFIGURATION:
    - **DATA USAGE**: Use **100%** of the training data. Do NOT subset.
    - **EPOCHS**: Train for **5 epochs**.
    - **EARLY STOPPING**: Implement logic to stop training if validation loss doesn't improve for 3 epochs.
    - **DEVICE**: Explicitly check `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move models/data to device.
    - **AMP**: Use `torch.cuda.amp.GradScaler` (Mixed Precision) to speed up training and save VRAM.

    ### 3. LIBRARY BEST PRACTICES:
    - **Optimizers**: NEVER import `AdamW` from `transformers`. ALWAYS use `torch.optim.AdamW`.
    - **DataLoaders**: Set `num_workers=4` and `pin_memory=True`. 
        - **CRITICAL**: Wrap the main execution logic in `if __name__ == "__main__":` to prevent multiprocessing crashes.
    - **Logging**: Print training progress EVERY 10 BATCHES (e.g., "Epoch 1, Batch 10/X, Loss: ...").
    - **Inference**: 
        - Multi-Class: `softmax(dim=1)`.
        - Binary/Multi-Label: `sigmoid`.
        - Regression: No activation.
    - For **multi-class classification (len(classes) > 2)**:
        - Final predictions MUST be produced using `softmax(dim=1)` so that **each row sums to 1**, matching Kaggle submission requirements.
        - Ensure the output probabilities exactly follow the column order in `sample_submission.csv`.


    ### 4. MANDATORY SUBMISSION LOGIC:
    - You MUST generate a `submission.csv` file at the end.
    - Load `sample_submission.csv` to ensure correct ID sorting.
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
