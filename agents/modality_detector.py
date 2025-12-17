import os
import json
import zipfile
import glob
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
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        col = object_cols[0]
        try:
            avg_words = df[col].astype(str).apply(lambda x: len(x.split())).mean()
            stats["avg_text_length"] = int(avg_words)
            stats["is_complex_text"] = bool(avg_words > 20)
        except:
            pass

    # 2. Image Complexity Profiling
    if HAS_PIL:
        try:
            for sub in ["images", "train", "test", "."]:
                p = public_dir / sub
                if p.exists():
                    images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
                    if images:
                        with Image.open(images[0]) as img:
                            stats["image_resolution_hint"] = img.size 
                        break
        except Exception:
            pass

    return stats

# ---------------------------------------------------------
# SMART FILE DETECTOR & EXTRACTOR
# ---------------------------------------------------------

def extract_zips_in_place(directory: Path):
    """
    Scans the directory for .zip files and extracts them in place.
    """
    zip_files = list(directory.glob("*.zip"))
    if not zip_files:
        return

    print(f"[INFO] Found {len(zip_files)} zip files. Extracting...")
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                file_names = zip_ref.namelist()
                if file_names and (directory / file_names[0]).exists():
                    continue
                print(f"       Extracting {zf.name}...")
                zip_ref.extractall(directory)
        except Exception as e:
            print(f"[WARN] Failed to extract {zf.name}: {e}")

def find_dataset_files(public_dir: Path):
    """
    Intelligently identifies the correct train and test files.
    """
    # Force glob to re-scan directory after potential extraction
    files = [f.name for f in public_dir.glob("*.csv")]
    
    train_file = "train.csv"
    test_file = "test.csv"
    
    # 1. Look for 'train'
    train_candidates = [f for f in files if "train" in f.lower() and "sample" not in f.lower()]
    if train_candidates:
        train_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        train_file = train_candidates[0]
        
    # 2. Look for 'test'
    test_candidates = [f for f in files if "test" in f.lower() and "sample" not in f.lower()]
    if test_candidates:
        test_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        test_file = test_candidates[0]
    
    # Fallback for test
    if not test_candidates and train_file:
        others = [f for f in files if f != train_file and "sample" not in f.lower()]
        if others:
             others.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
             test_file = others[0]

    return train_file, test_file

# ---------------------------------------------------------
# Dataset Metadata Collector
# ---------------------------------------------------------
def collect_dataset_metadata(public_dir: Path):
    
    # --- STEP 0: HANDLE ZIPPED CSVs ---
    extract_zips_in_place(public_dir)

    # Detect actual filenames
    train_fname, test_fname = find_dataset_files(public_dir)
    
    # --- NEW: DETECT SAMPLE SUBMISSION ---
    # We look for files with "sample" and "submission" in the name
    files = [f.name for f in public_dir.glob("*.csv")]
    sample_sub_fname = "sample_submission.csv" # Default
    
    # Priority: contains both "sample" and "submission"
    subs = [f for f in files if "sample" in f.lower() and "submission" in f.lower()]
    if subs:
        sample_sub_fname = subs[0]
    else:
        # Fallback: just "submission" (rare but possible)
        subs = [f for f in files if "submission" in f.lower()]
        if subs:
            sample_sub_fname = subs[0]

    train_path = public_dir / train_fname if train_fname else None
    test_path = public_dir / test_fname if test_fname else None
    sample_sub_path = public_dir / sample_sub_fname if sample_sub_fname else None
    
    print(f"[INFO] Dataset Files:\n       Train: {train_fname}\n       Test:  {test_fname}\n       Sub:   {sample_sub_fname}")

    # --- FAULT TOLERANT CSV READ ---
    try:
        df_train = pd.read_csv(train_path, nrows=100)
    except Exception:
        df_train = pd.DataFrame()

    try:
        df_test = pd.read_csv(test_path, nrows=100)
    except Exception:
        df_test = pd.DataFrame()

    df_sub = pd.DataFrame()
    if sample_sub_path and sample_sub_path.exists():
        try: df_sub = pd.read_csv(sample_sub_path, nrows=100)
        except: pass


    # --- AUDIO FILE DETECTION ---
    audio_dirs = {}
    for dname in ["train2", "test2", "audio", "train_audio", "test_audio"]:
        dpath = public_dir / dname
        if dpath.exists() and dpath.is_dir():
            audio_files = []
            for ext in ["*.aif", "*.aiff", "*.wav", "*.flac", "*.mp3", "*.ogg"]:
                audio_files.extend([str(p) for p in dpath.rglob(ext)])
            
            if audio_files:
                audio_dirs[dname] = {
                    "path": str(dpath),
                    "num_audio_files": len(audio_files),
                    "example_audio": audio_files[:3],
                }

    # Row estimation
    num_train_rows = 0
    try:
        if train_path.exists():
            with open(train_path, "rb") as f:
                num_train_rows = max(0, sum(1 for _ in f) - 1)
    except:
        num_train_rows = 0

    dataset_size_mb = get_directory_size_mb(public_dir)
    complexity_stats = profile_data_complexity(df_train, public_dir)
    description_text = read_description_text(public_dir)

    metadata = {
        "description": description_text,
        "train_filename": train_fname,
        "test_filename": test_fname,
        "sample_submission_filename": sample_sub_fname,
        "train_columns": list(df_train.columns),
        "test_columns": list(df_test.columns),
        "submission_columns": list(df_sub.columns),
        "dtypes": df_train.dtypes.astype(str).to_dict(),
        "sample_rows": df_train.head(3).to_dict(orient="records"),
        "num_train_rows": num_train_rows,
        "dataset_size_mb": dataset_size_mb,
        "complexity": complexity_stats,
        "directory_files": [p.name for p in public_dir.iterdir()],
        "audio_dirs": audio_dirs,
        "has_audio": len(audio_dirs) > 0,
    }

    return metadata

# ---------------------------------------------------------
# LangGraph State & Detector
# ---------------------------------------------------------
class ModalityState(dict):
    metadata: dict
    output: dict

def llm_modality_detector(state: ModalityState):
    metadata = state["metadata"]
    if metadata.get("has_audio", False):
        metadata["forced_modality_hint"] = "audio"

    prompt = f"""
    You are an ML engineering agent.
    TASK DESCRIPTION: \"\"\"{metadata.get("description", "")}\"\"\"
    METADATA: {json.dumps({k: v for k, v in metadata.items() if k != "description"}, indent=2)}

    Determine: modality, task_type, target_col, classes.
    
    JSON schema:
    {{
    "modality": "...",
    "task_type": "...",
    "target_col": "...",
    "classes": []
    }}
    """

    response = completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response["choices"][0]["message"]["content"]
    start_index = raw.find("{")
    end_index = raw.rfind("}")
    if start_index != -1 and end_index != -1:
        raw = raw[start_index : end_index + 1]

    try:
        parsed = json.loads(raw)
    except Exception:
        raise ValueError(f"Gemini did not return valid JSON:\n{raw}")

    state["output"] = parsed
    return state

def build_graph():
    graph = StateGraph(ModalityState)
    graph.add_node("modality_detector", llm_modality_detector)
    graph.set_entry_point("modality_detector")
    graph.set_finish_point("modality_detector")
    return graph.compile()

async def detect_modality_llm(metadata: dict):
    graph = build_graph()
    final_state = await graph.ainvoke({"metadata": metadata})
    return final_state["output"]