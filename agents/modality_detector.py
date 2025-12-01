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
# Helper: Text & File Readers
# ---------------------------------------------------------
def read_description_text(public_dir: Path):
    """
    Looks for description.md or README.md in the prepared public directory.
    Returns truncated text content.
    """
    candidates = [
        public_dir / "description.md",
        public_dir / "README.md",
    ]

    for p in candidates:
        if p.exists():
            try:
                # Read and truncate to avoid huge prompts
                text = p.read_text(encoding="utf-8", errors="ignore")
                return text[:3000]  # Limit to 3000 chars
            except Exception:
                continue
    return "No description available."


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
# SMART FILE DETECTOR (The Fix)
# ---------------------------------------------------------
def find_dataset_files(public_dir: Path):
    """
    Intelligently identifies the correct train and test files 
    even if they are named 'en_train.csv' or 'test_2.csv'.
    """
    files = [f.name for f in public_dir.glob("*.csv")]
    
    train_file = "train.csv"
    test_file = "test.csv"
    
    # 1. Look for 'train' in filename (pick largest if multiple)
    train_candidates = [f for f in files if "train" in f.lower()]
    if train_candidates:
        # Sort by file size (largest is usually the real train set)
        train_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        train_file = train_candidates[0]
        
    # 2. Look for 'test' in filename
    test_candidates = [f for f in files if "test" in f.lower()]
    if test_candidates:
        test_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        test_file = test_candidates[0]

    return train_file, test_file

# ---------------------------------------------------------
# Dataset Metadata Collector (sync, deterministic)
# ---------------------------------------------------------
def collect_dataset_metadata(public_dir: Path):
    # Detect actual filenames
    train_fname, test_fname = find_dataset_files(public_dir)

    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    
    # --- FIX: Read only 100 rows to prevent IO Crash on Metadata Check ---
    try:
        df_train = pd.read_csv(train_path, nrows=100) if train_path.exists() else pd.DataFrame()
    except Exception:
        df_train = pd.DataFrame()
        
    try:
        df_test = pd.read_csv(test_path, nrows=100) if test_path.exists() else pd.DataFrame()
    except Exception:
        df_test = pd.DataFrame()

    # Estimate total rows safely
    num_train_rows = 0
    if train_path.exists():
        try:
            # Fast line count
            with open(train_path, "rb") as f:
                num_train_rows = sum(1 for _ in f) - 1
        except:
            num_train_rows = 1000

    dataset_size_mb = get_directory_size_mb(public_dir)
    complexity_stats = profile_data_complexity(df_train, public_dir)
    description_text = read_description_text(public_dir)

    metadata = {
        "description": description_text,
        "train_filename": train_fname,
        "test_filename": test_fname,
        "train_columns": list(df_train.columns),
        "test_columns": list(df_test.columns),
        "dtypes": df_train.dtypes.astype(str).to_dict(),
        "sample_rows": df_train.head(3).to_dict(orient="records"),
        "num_train_rows": num_train_rows, # Uses safe count
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

    You receive dataset metadata and a task description.
    
    TASK DESCRIPTION (from README/description.md):
    \"\"\"{metadata.get("description", "")}\"\"\"

    METADATA:
    {json.dumps({k: v for k, v in metadata.items() if k != "description"}, indent=2)}

    Determine:
    - modality (one of: text, tabular, image, audio, seq2seq, multimodal)
    - task_type (classification, regression, seq2seq, image_classification, audio_classification)
    - target_col
    - classes (list)

    REASONING GUIDELINES:
    1. **Use Description First**: If the description explicitly says "Classify audio files" or "Predict the housing price", trust that over file extensions.
    2. **Modality Hierarchy**: 
       - Prioritize complex modalities (Image/Audio) over Tabular. 
       - If a CSV contains filenames pointing to images -> Modality is 'image'.
    3. **Task Type Definitions**:
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
