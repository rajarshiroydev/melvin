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