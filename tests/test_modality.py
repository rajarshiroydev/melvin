import pytest
import pandas as pd
import json

# Import the functions we want to test
from melvin.agents.modality_detector import collect_dataset_metadata, detect_modality_llm


# --------------------------------------------------------------------------
# TEST 1: File System & Metadata Collection (No LLM)
# --------------------------------------------------------------------------
def test_collect_dataset_metadata(tmp_path):
    """
    Tests if the function can correctly read CSV files and extract metadata.
    We use 'tmp_path' (built-in pytest fixture) to create a fake directory.
    """

    # 1. Setup: Create a fake dataset directory in memory/temp
    data_dir = tmp_path / "prepared_public"
    data_dir.mkdir()

    # Create dummy CSVs
    df_train = pd.DataFrame({"text_col": ["a", "b", "c"], "target": [0, 1, 0]})
    df_test = pd.DataFrame({"text_col": ["d", "e"]})
    df_sample = pd.DataFrame({"id": [1, 2], "target": [0, 0]})

    df_train.to_csv(data_dir / "train.csv", index=False)
    df_test.to_csv(data_dir / "test.csv", index=False)
    df_sample.to_csv(data_dir / "sample_submission.csv", index=False)

    # 2. Action: Run the function
    metadata = collect_dataset_metadata(data_dir)

    # 3. Assertions: Check if logic is correct
    assert metadata["num_train_rows"] == 3
    assert metadata["num_test_rows"] == 2
    assert "text_col" in metadata["train_columns"]
    assert "target" in metadata["train_columns"]
    # Check that it captured file names
    assert "train.csv" in metadata["directory_files"]


# --------------------------------------------------------------------------
# TEST 2: LLM Logic (Happy Path)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_detect_modality_llm_happy_path(mocker):
    # 1. Define Input
    dummy_metadata = {
        "train_columns": ["pixel_1", "pixel_2", "label"],
        "num_train_rows": 100,
    }

    # --- PRINT INPUT ---
    print("\n\n=== [1] INPUT DATA (Metadata) ===")
    print(json.dumps(dummy_metadata, indent=2))

    # 2. Setup Mock
    expected_output = {
        "modality": "tabular",
        "task_type": "classification",
        "target_col": "label",
        "classes": [],
    }
    mock_response = {"choices": [{"message": {"content": json.dumps(expected_output)}}]}

    # Capture the mock object so we can inspect it later
    mock_llm = mocker.patch("modality_detector.completion", return_value=mock_response)

    # 3. Action
    result = await detect_modality_llm(dummy_metadata)

    # --- PRINT THE PROMPT (What was sent to the Agent?) ---
    # This grabs the actual arguments sent to the mocked Gemini function
    args, kwargs = mock_llm.call_args
    # kwargs['messages'][0]['content'] contains the full prompt string
    actual_prompt = kwargs["messages"][0]["content"]

    print("\n=== [2] GENERATED PROMPT (Sent to LLM) ===")
    print(actual_prompt)

    # --- PRINT THE RESULT (What the Agent returned) ---
    print("\n=== [3] AGENT OUTPUT (Parsed Result) ===")
    print(json.dumps(result, indent=2))

    # 4. Assertion
    assert result["modality"] == "tabular"


# --------------------------------------------------------------------------
# TEST 3: LLM Logic (Dirty JSON / Markdown Fix)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_detect_modality_llm_markdown_cleaning(mocker):
    """
    Tests if the agent is robust enough to clean ```json ... ``` markdown tags
    which LLMs frequently output.
    """

    dummy_metadata = {"some": "data"}

    # The LLM returns markdown fences + json
    dirty_string = """
    Here is your JSON:
    ```json
    {
        "modality": "text",
        "task_type": "classification",
        "target_col": "sentiment",
        "classes": ["pos", "neg"]
    }
    ```
    """

    mock_response = {"choices": [{"message": {"content": dirty_string}}]}

    mocker.patch("modality_detector.completion", return_value=mock_response)

    # Action
    result = await detect_modality_llm(dummy_metadata)

    # Assertion: Did it clean the markdown and parse correctly?
    assert result["modality"] == "text"
    assert result["target_col"] == "sentiment"
    assert len(result["classes"]) == 2
