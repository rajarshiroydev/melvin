import os
import json
import re
from litellm import completion

# ---------------------------------------------------------
# PROMPT: Ablation Designer (What to tune?)
# ---------------------------------------------------------
ABLATION_PROMPT = """
You are a Kaggle Grandmaster optimizing a script.
SCRIPT PURPOSE: {task_type} (Metric: {metric_direction})

CURRENT SCRIPT SNIPPET (First 200 lines):
```python
{script_head}
TASK:
Identify 2 distinct "Hyperparameters" or "Logic Blocks" that are most likely to improve the score if tuned.
Focus on: Epochs, Learning Rate, Model Architecture arguments, or Preprocessing constants.
Output JSON ONLY:
[
{{
"component_name": "Number of Epochs",
"reasoning": "Model might be overfitting given the small dataset.",
"code_snippet_to_find": "num_train_epochs=..."
}},
...
]
"""

# ---------------------------------------------------------
# PROMPT: Refinement Planner (How to tune?)
# ---------------------------------------------------------
PLANNER_PROMPT = """
You are an ML Engineer.
Target: {component}
Reasoning: {reasoning}
Suggest 2 specific, distinct variations to try.
Keep values realistic for a "Lite" dataset.
Output JSON ONLY:
[
{{ "variant_name": "LowEpochs", "instruction": "Change num_train_epochs to 2" }},
{{ "variant_name": "HighLR", "instruction": "Change learning_rate to 5e-5" }}
]
"""

# ---------------------------------------------------------
# PROMPT: Smart Patcher (Apply the fix)
# ---------------------------------------------------------
PATCHER_PROMPT = """
You are an expert Code Patcher.
Your job is to apply a specific change to a Python script without breaking anything else.
CHANGE INSTRUCTION: {instruction}
ORIGINAL SCRIPT:
{script}
RULES:
Apply the instruction EXACTLY.
DO NOT remove the subsampling logic (e.g. nrows= or .sample()). WE NEED SPEED.
Return the FULL updated script.
Return ONLY valid Python code.
"""


async def propose_ablations(script, task_type, metric_direction):
    # Pass only the head to save tokens, usually hyperparameters are at the top
    script_head = "\n".join(script.splitlines()[:200])
    prompt = ABLATION_PROMPT.format(
        task_type=task_type,
        metric_direction=metric_direction,
        script_head=script_head
    )

    try:
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response["choices"][0]["message"]["content"]
        start = raw.find("[")
        end = raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"[ERR] Ablation parsing failed: {e}")
        return []


async def propose_refinements(component_info):
    prompt = PLANNER_PROMPT.format(
        component=component_info['component_name'],
        reasoning=component_info['reasoning']
    )
    try:
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response["choices"][0]["message"]["content"]
        start = raw.find("[")
        end = raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except:
        return []


async def apply_refinement_llm(script, instruction):
    """
    Uses LLM to rewrite the script with the requested change.
    Safer than string replace for complex changes.
    """
    prompt = PATCHER_PROMPT.format(instruction=instruction, script=script)
    response = completion(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response["choices"][0]["message"]["content"]
    cleaned = raw.replace("```python", "").replace("```", "").strip()
    return cleaned