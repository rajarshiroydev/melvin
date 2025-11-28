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
    train_path = public_dir / "en_train.csv"
    test_path = public_dir / "en_test_2.csv"
    sample_path = public_dir / "en_sample_submission_2.csv"

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
    - Always output VALID JSON.
    - No explanation, ONLY JSON.

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

    # Clean the response string by removing markdown code fences
    if raw.strip().startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

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
