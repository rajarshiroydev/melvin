import os
import json
import prompts
from litellm import completion

os.environ['GROQ_API_KEY']

async def propose_ablations(script, task_type, metric_direction):
    # Pass only the head to save tokens, usually hyperparameters are at the top
    script_head = "\n".join(script.splitlines()[:200])
    prompt = prompts.ABLATION_PROMPT.format(
        task_type=task_type,
        metric_direction=metric_direction,
        script_head=script_head
    )

    try:
        response = completion(
            model="deepseek/deepseek-chat",
            # api_key=os.getenv("GROQ_API_KEY"),
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
    prompt = prompts.PLANNER_PROMPT.format(
        component=component_info['component_name'],
        reasoning=component_info['reasoning']
    )
    try:
        response = completion(
            model="deepseek/deepseek-chat",
            # api_key=os.getenv("GROQ_API_KEY"),
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
    prompt = prompts.PATCHER_PROMPT.format(instruction=instruction, script=script)
    response = completion(
        model="deepseek/deepseek-chat",
        # api_key=os.getenv("GROQ_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response["choices"][0]["message"]["content"]
    cleaned = raw.replace("```python", "").replace("```", "").strip()
    return cleaned