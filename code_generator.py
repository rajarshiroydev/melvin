# code_generator.py

import os
import json
from pathlib import Path

from litellm import completion
from langgraph.graph import StateGraph

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------
class CodegenState(dict):
    modality: str
    task_type: str
    target_col: str
    classes: list
    metadata: dict
    script: str


# ---------------------------------------------------------
# NODE: Gemini 2.5 Flash — Full Script Generator (One LLM Call)
# ---------------------------------------------------------
def llm_script_generator(state: CodegenState):
    modality = state["modality"]
    task_type = state["task_type"]
    target_col = state["target_col"]
    classes = state.get("classes", [])
    metadata = state["metadata"]

    prompt = f"""
You are an advanced ML engineering agent.
Your job is to generate a FULL, SELF-CONTAINED Python training script.

The script MUST:
- load train.csv, test.csv, sample_submission.csv from current directory
- use {modality} + {task_type} to decide model and preprocessing
- use target_col = "{target_col}"
- handle classes = {classes}
- fit a model
- generate predictions
- write submission.csv in correct sample_submission format
- run WITHOUT modification
- contain all imports
- NO placeholders
- Python ONLY
- DO NOT wrap in backticks
- DO NOT explain anything
- ONLY output executable Python code

Dataset metadata:
{json.dumps(metadata, indent=2)}

Follow these model rules:

TEXT CLASSIFICATION:
- Use TfidfVectorizer + LogisticRegression (sklearn)

TABULAR CLASSIFICATION:
- Use LightGBM OR RandomForestClassifier

TABULAR REGRESSION:
- Use LightGBM OR RandomForestRegressor

SEQ2SEQ (text normalization):
- Use transformers (T5-small)
- Tokenize input/output columns
- Predict normalized text
- Produce submission file

IMAGE CLASSIFICATION:
- Use torchvision (ResNet18)
- Standard transforms, DataLoader

AUDIO CLASSIFICATION:
- Use torchaudio, CNN-based classifier

Remember:
- You MUST write submission.csv
- Column order MUST match sample_submission.csv
- Ensure probabilities sum to 1 for classification
"""

    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw_script = response["choices"][0]["message"]["content"]

    # Remove accidental ``` python fences
    if raw_script.strip().startswith("```"):
        raw_script = raw_script.replace("```python", "").replace("```", "").strip()

    state["script"] = raw_script
    return state


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------
def build_graph():
    graph = StateGraph(CodegenState)
    graph.add_node("code_generator", llm_script_generator)
    graph.set_entry_point("code_generator")
    graph.set_finish_point("code_generator")
    return graph.compile()


# ---------------------------------------------------------
# Public API: generate training script file
# ---------------------------------------------------------
async def generate_training_script_llm(
    modality, task_type, target_col, classes, metadata, output_path: Path
):
    graph = build_graph()
    final_state = await graph.ainvoke(
        {
            "modality": modality,
            "task_type": task_type,
            "target_col": target_col,
            "classes": classes,
            "metadata": metadata,
        }
    )

    script = final_state["script"]

    output_path.write_text(script)

    return output_path
