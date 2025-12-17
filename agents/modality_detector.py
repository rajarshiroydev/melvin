import os
import json
import zipfile
import glob
import re
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
# UNSTRUCTURED DATA NORMALIZER (NEW LOGIC)
# ---------------------------------------------------------
def normalize_unstructured_dataset(public_dir: Path):
    """
    If no CSVs are found, this scans directories to create 'synthetic' 
    generated_train.csv and generated_test.csv by inferring labels.
    """
    print("[INFO] No CSVs found. Attempting to normalize unstructured data...")

    # 1. Find Directories
    dirs = [d for d in public_dir.iterdir() if d.is_dir()]
    train_dir = next((d for d in dirs if "train" in d.name.lower()), None)
    test_dir = next((d for d in dirs if "test" in d.name.lower()), None)

    if not train_dir:
        return None, None 

    # 2. Scan Train Files
    exts = {".jpg", ".png", ".jpeg", ".tif", ".tiff", ".wav", ".mp3", ".aif", ".aiff", ".flac", ".ogg"}
    train_files = []
    for p in train_dir.rglob("*"):
        if p.suffix.lower() in exts:
            train_files.append(p)

    if not train_files:
        return None, None

    print(f"       Found {len(train_files)} raw training files in {train_dir.name}")

    # 3. Infer Labels
    data = []
    # Strategy A: Filename Suffix (Whale Challenge: clip_0.aif, clip_1.aif)
    has_suffix_label = any(re.search(r"_([01])\.", f.name) for f in train_files[:10])
    # Strategy B: Subfolder Name (ImageNet: train/dog/x.jpg)
    has_folder_label = any(f.parent.name != train_dir.name for f in train_files[:10])

    for f in train_files:
        label = None
        if has_suffix_label:
            match = re.search(r"_([01])\.", f.name)
            if match:
                label = int(match.group(1))
        elif has_folder_label:
            label = f.parent.name
        
        data.append({
            "filepath": str(f.absolute()),
            "filename": f.name,
            "label": label if label is not None else -1
        })
    
    df_train = pd.DataFrame(data)
    gen_train_path = public_dir / "generated_train.csv"
    df_train.to_csv(gen_train_path, index=False)
    print(f"       Generated {gen_train_path.name}")

    # 4. Scan Test Files
    gen_test_path = None
    if test_dir:
        test_files = []
        for p in test_dir.rglob("*"):
            if p.suffix.lower() in exts:
                test_files.append(p)
        
        if test_files:
            test_data = []
            for f in test_files:
                test_data.append({
                    "filepath": str(f.absolute()),
                    "id": f.name # Default ID to filename
                })
            
            df_test = pd.DataFrame(test_data)
            gen_test_path = public_dir / "generated_test.csv"
            df_test.to_csv(gen_test_path, index=False)
            print(f"       Generated {gen_test_path.name}")

            # 5. Synthetic Sample Submission
            if not list(public_dir.glob("*sample_submission*.csv")):
                sub_df = pd.DataFrame({
                    "id": df_test["id"],
                    "prediction": 0
                })
                sub_path = public_dir / "generated_sample_submission.csv"
                sub_df.to_csv(sub_path, index=False)

    return gen_train_path.name, (gen_test_path.name if gen_test_path else None)

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
    
    train_file = None
    test_file = None
    
    # Filter out sample submissions and previously generated files to find originals first
    candidates = [f for f in files if "sample" not in f.lower() and "generated" not in f.lower()]

    # 1. Look for 'train'
    train_candidates = [f for f in candidates if "train" in f.lower()]
    if train_candidates:
        train_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        train_file = train_candidates[0]
        
    # 2. Look for 'test'
    test_candidates = [f for f in candidates if "test" in f.lower()]
    if test_candidates:
        test_candidates.sort(key=lambda x: (public_dir / x).stat().st_size, reverse=True)
        test_file = test_candidates[0]
    
    # Fallback for test
    if not test_candidates and train_file:
        others = [f for f in candidates if f != train_file]
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
    
    # --- STEP 1: HANDLE UNSTRUCTURED DATA (NEW) ---
    if not train_fname:
        train_fname, test_fname = normalize_unstructured_dataset(public_dir)
    # ----------------------------------------------

    # --- NEW: DETECT SAMPLE SUBMISSION ---
    # We look for files with "sample" and "submission" in the name
    files = [f.name for f in public_dir.glob("*.csv")]
    sample_sub_fname = "sample_submission.csv" # Default
    
    # Priority: contains both "sample" and "submission"
    subs = [f for f in files if "sample" in f.lower() and "submission" in f.lower()]
    if subs:
        sample_sub_fname = subs[0]
    elif [f for f in files if "submission" in f.lower()]: 
        sample_sub_fname = [f for f in files if "submission" in f.lower()][0]

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
        if train_path and train_path.exists():
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
        "has_filepath_col": "filepath" in df_train.columns # Key flag for Code Gen
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
    # Check for Audio/Image signals including generated filepath column
    if metadata.get("has_audio", False) or metadata.get("audio_dirs") or metadata.get("has_filepath_col", False):
        metadata["forced_modality_hint"] = "audio/image"

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