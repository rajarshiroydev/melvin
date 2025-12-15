import os
import json
import asyncio
from tavily import TavilyClient
from litellm import completion

# Ensure TAVILY_API_KEY is in your .env
# os.environ['GROQ_API_KEY'] 
# os.environ['TAVILY_API_KEY']

def search_web_tavily(query):
    """
    Uses Tavily to get deep, context-rich search results from a single complex query.
    """
    print(f"[SEARCH] Querying Tavily: {query[:100]}...")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    try:
        # We use a single advanced search call.
        # Tavily's AI will parse the complex requirements in the query string.
        response = tavily.search(
            query=query,
            search_depth="advanced", 
            max_results=7, # Increased slightly since we are doing 1 shot
            include_domains=["kaggle.com", "github.com", "medium.com", "towardsdatascience.com"],
            exclude_domains=[]
        )
        
        results = []
        for r in response.get("results", []):
            # We capture the content to give the LLM the "Secret Sauce"
            results.append(f"Title: {r['title']}\nSource: {r['url']}\nContent: {r['content'][:2000]}") 
            
        return results
    except Exception as e:
        print(f"[WARN] Tavily search failed: {e}")
        return []

async def retrieve_model_candidates(metadata, competition_id, task_type, modality):
    """
    Retrieves winning strategies using a single, high-density Tavily query.
    Returns: JSON with metric direction and 3 distinct candidates.
    """
    
    # 1. Construct ONE Comprehensive Query
    # This matches the logic of your new agent: asking for everything in context.
    query = (
        f"Kaggle {competition_id} winning solutions."
        f"Identify the core techniques used by winners and list the "
        f"current 2025 SOTA models for those specific techniques. "
        f"Also explicitly state if the evaluation metric for the competition is maximized or minimized."
    )
    
    # 2. Execute Search (Single Call)
    search_results = search_web_tavily(query)
    
    if not search_results:
        print("[WARN] No search results found. Retrying with simpler query...")
        # Fallback for obscure competitions
        fallback_query = f"Kaggle {competition_id} solution github"
        search_results = search_web_tavily(fallback_query)

    context_str = "\n\n---\n\n".join(search_results)

    # 3. LLM Synthesis
    prompt = f"""
    You are a Senior Kaggle Grandmaster. 
    Analyze the search results below for the competition "{competition_id}".
    
    SEARCH CONTEXT:
    {context_str}
    
    DATASET META:
    Modality: {modality}
    Task: {task_type}
    
    YOUR GOAL:
    Extract the winning strategy.
    1. **Metric Direction**: strictly "maximize" or "minimize".
    2. **Candidates**: Propose 3 distinct, high-performance approaches found in the text.
    
    CRITICAL:
    - In "implementation_tips", do NOT write generic advice like "use cross validation".
    - Write SPECIFIC findings, e.g., "Use GroupKFold on 'user_id', clip targets to (0,20), use lag features."
    
    OUTPUT JSON ONLY:
    {{
      "metric_direction": "maximize" OR "minimize",
      "candidates": [
          {{
            "model_name": "Name of approach (e.g. Stacked LightGBM)",
            "library": "Primary library (e.g. lightgbm)",
            "reasoning": "Why this specific approach won",
            "implementation_tips": "Technical details: specific features, loss functions, or CV strategies found in search."
          }},
          ...
      ]
    }}
    """

    response = completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response["choices"][0]["message"]["content"]
    
    # Robust JSON extraction
    start = raw.find("{")
    end = raw.rfind("}") + 1
    json_str = raw[start:end]
    result = json.loads(json_str)
    
    # Validation
    if "candidates" not in result: 
        raise ValueError("Missing candidates key")
        
    return result