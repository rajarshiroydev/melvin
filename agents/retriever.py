import os
import json
from ddgs import DDGS
from litellm import completion

def search_web(query, max_results=5):
    """
    Uses DuckDuckGo to find relevant discussions/code.
    """
    print(f"[SEARCH] Querying: {query}")
    results = []
    try:
        # Use the synchronous DDGS context manager
        with DDGS() as ddgs:
            # .text() returns an iterator of results
            search_gen = ddgs.text(query, max_results=max_results)
            for r in search_gen:
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
    except Exception as e:
        print(f"[WARN] Search failed: {e}")
        return []
    return results

async def retrieve_model_candidates(metadata, competition_id, task_type, modality):
    """
    Returns a list of dictionaries: [{'model_name': '...', 'reasoning': '...', 'libs': '...'}]
    """
    
    # 1. Construct Queries
    queries = [
        f"{competition_id} kaggle solution github",
        f"{competition_id} kaggle winning approach",
        f"state of the art model for {task_type} {modality} python",
    ]
    
    # 2. Execute Search
    search_context = []
    for q in queries:
        results = search_web(q)
        search_context.extend(results)
    
    context_str = "\n---\n".join(search_context[:15]) # Limit context

    # 3. LLM Synthesis (The "Researcher")
    prompt = f"""
    You are a Senior ML Researcher. 
    Analyze the search results below for the Kaggle competition "{competition_id}" (or similar {task_type} tasks).
    
    Identify 3 DISTINCT, HIGH-PERFORMING approaches.
    
    SEARCH CONTEXT:
    {context_str}
    
    DATASET META:
    Modality: {modality}
    Task: {task_type}
    Rows: {metadata.get('num_train_rows', 'Unknown')}
    
    RULES:
    1. Diversity: Don't propose 3 variants of the same model. Propose diverse robust approaches (e.g. 1 Gradient Boosting, 1 Neural Net, 1 Linear/Baseline).
    2. Feasibility: Must be implementable in Python with standard libraries.
    3. Output JSON ONLY.
    
    JSON SCHEMA:
    [
      {{
        "model_name": "Name of approach (e.g. CatBoost with Optuna)",
        "library": "Main library (e.g. catboost)",
        "reasoning": "Why this works based on search results",
        "implementation_tips": "Key hyperparameters or preprocessing mentioned in search"
      }},
      ...
    ]
    """

    try:
        response = completion(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        raw = response["choices"][0]["message"]["content"]
        
        # Cleaning JSON
        start = raw.find("[")
        end = raw.rfind("]") + 1
        json_str = raw[start:end]
        candidates = json.loads(json_str)
        return candidates
    except Exception as e:
        print(f"[ERR] Failed to parse retriever JSON: {e}")
        # Fallback candidates
        return [
            {"model_name": "Baseline XGBoost", "library": "xgboost", "reasoning": "Robust baseline", "implementation_tips": "Use early stopping"},
            {"model_name": "LightGBM", "library": "lightgbm", "reasoning": "Fast and accurate", "implementation_tips": "Use leaf-wise growth"}
        ]