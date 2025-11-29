# modality_detector.py

import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from litellm import completion

from langgraph.graph import StateGraph


load_dotenv()


# ---------------------------------------------------------
# Dataset Metadata Collector (sync, deterministic)
# ---------------------------------------------------------
def collect_dataset_metadata(public_dir: Path):
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    metadata = {
        "train_columns": list(df_train.columns),
        "test_columns": list(df_test.columns),
        "sample_submission_columns": list(df_sample.columns),
        "dtypes": df_train.dtypes.astype(str).to_dict(),
        "sample_rows": df_train.head(5).to_dict(orient="records"),
        "num_train_rows": len(df_train),
        "num_test_rows": len(df_test),
        "directory_files": [p.name for p in public_dir.iterdir()],
    }

    return metadata


# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------
class ModalityState(dict):
    metadata: dict
    output: dict


# ---------------------------------------------------------
# Node: LLM Modality Detector using Gemini 2.5 Flash
# ---------------------------------------------------------
def llm_modality_detector(state: ModalityState):
    metadata = state["metadata"]

    prompt = f"""
    You are an ML engineering agent.

    You receive dataset metadata:
    {json.dumps(metadata, indent=2)}

    Determine:
    - modality (one of: text, tabular, image, audio, seq2seq, multimodal)
    - task_type (classification, regression, seq2seq, image_classification, audio_classification)
    - target_col
    - classes (list)

    RULES:
    1. Always output VALID JSON. No explanation.
    2. **HIERARCHY OF MODALITIES**:
       - If IMAGE files (.jpg, .png) are present -> Modality is 'image' (even if CSVs exist).
       - If AUDIO files (.wav, .flac) are present -> Modality is 'audio'.
       - Only select 'tabular' if NO images/audio/long-text are present.
    3. **TASK TYPE LOGIC**:
       - If target is a continuous number -> 'regression'.
       - If target is a class/label -> 'classification'.
       - If target is text (translation/summary) -> 'seq2seq'.
       - **CRITICAL**: If inputs are images AND targets are also images (e.g., denoising, super-resolution, file paths in clean/dirty folders) -> task_type is 'regression' (pixel-wise regression).

    JSON schema:
    {{
    "modality": "...",
    "task_type": "...",
    "target_col": "...",
    "classes": []
    }}
    """

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response["choices"][0]["message"]["content"]

    # Strategy: Find the first '{' and the last '}' to isolate the JSON object
    start_index = raw.find("{")
    end_index = raw.rfind("}")

    if start_index != -1 and end_index != -1:
        # Extract the substring between the braces
        raw = raw[start_index : end_index + 1]

    # Safe JSON parse
    try:
        parsed = json.loads(raw)
    except Exception:
        raise ValueError(f"Gemini did not return valid JSON:\n{raw}")

    state["output"] = parsed
    return state


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------
def build_graph():
    graph = StateGraph(ModalityState)
    graph.add_node("modality_detector", llm_modality_detector)
    graph.set_entry_point("modality_detector")
    graph.set_finish_point("modality_detector")
    return graph.compile()


# ---------------------------------------------------------
# Public API function
# ---------------------------------------------------------
async def detect_modality_llm(metadata: dict):
    graph = build_graph()
    final_state = await graph.ainvoke({"metadata": metadata})
    return final_state["output"]
