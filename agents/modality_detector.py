# modality_detector.py

import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from langgraph.graph import StateGraph

# Try importing PIL for image profiling, handle if missing
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

load_dotenv()


# ---------------------------------------------------------
# Helper: Profiling Functions
# ---------------------------------------------------------
def get_directory_size_mb(directory: Path):
    """Calculates total size of a directory in MB."""
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception:
        return 0
    return round(total_size / (1024 * 1024), 2)


def profile_data_complexity(df, public_dir):
    """Generates heuristic stats about data complexity."""
    stats = {
        "avg_text_length": 0,
        "image_resolution_hint": None,
        "is_complex_text": False,
    }

    # 1. Text Complexity Profiling
    # Check string columns to see if they look like long text (sentences) or short (categories)
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        # Sample first non-ID column
        col = object_cols[0]
        # Calculate avg word count of first 5 rows
        try:
            avg_words = df[col].astype(str).apply(lambda x: len(x.split())).mean()
            stats["avg_text_length"] = int(avg_words)
            stats["is_complex_text"] = bool(
                avg_words > 20
            )  # Threshold for NLP vs Categorical
        except:
            pass

    # 2. Image Complexity Profiling
    # Try to find an image folder and read one image to get dimensions
    if HAS_PIL:
        try:
            # Look for common image folders
            for sub in ["images", "train", "test", "."]:
                p = public_dir / sub
                if p.exists():
                    # Find first jpg/png
                    images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
                    if images:
                        with Image.open(images[0]) as img:
                            stats["image_resolution_hint"] = img.size  # (width, height)
                        break
        except Exception:
            pass

    return stats


# ---------------------------------------------------------
# Dataset Metadata Collector (sync, deterministic)
# ---------------------------------------------------------
def collect_dataset_metadata(public_dir: Path):
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"

    # Robust read (handle missing files for edge cases)
    df_train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    df_test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

    # Calculate Profiling Stats
    dataset_size_mb = get_directory_size_mb(public_dir)
    complexity_stats = profile_data_complexity(df_train, public_dir)

    metadata = {
        "train_columns": list(df_train.columns),
        "test_columns": list(df_test.columns),
        # "sample_submission_columns": ... (Optional, removed to save context window)
        "dtypes": df_train.dtypes.astype(str).to_dict(),
        "sample_rows": df_train.head(3).to_dict(orient="records"),
        "num_train_rows": len(df_train),
        "dataset_size_mb": dataset_size_mb,
        "complexity": complexity_stats,
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

    REASONING GUIDELINES:
    1. **Modality Hierarchy**: 
       - Prioritize complex modalities (Image/Audio) over Tabular. 
       - If a CSV contains filenames pointing to images -> Modality is 'image'.
    
    2. **Task Type Definitions**:
       - **Classification**: Target is a discrete label or class ID.
       - **Regression**: Target is a continuous number.
       - **Image Restoration/Generation**: If the target column points to *image files* (meaning Input Image -> Output Image), this is technically a **regression** task (predicting pixel values), NOT seq2seq.
       - **Seq2Seq**: Target is text (e.g., translation).

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
