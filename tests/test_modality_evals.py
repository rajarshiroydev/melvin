import pytest
import asyncio
from modality_detector import detect_modality_llm

# --------------------------------------------------------------------------
# THE GOLDEN DATASET (Ground Truth)
# --------------------------------------------------------------------------
# Format: (Simulated Metadata, Expected Modality, Expected Task Type)

EVAL_CASES = [
    # ---------------------------------------------------------
    # 1. TABULAR (Classification & Regression)
    # ---------------------------------------------------------
    (
        # Competition: tabular-playground-series-may-2022
        # Characteristics: Anonymized features (f_00, f_01...), binary target
        {
            "train_columns": ["id"] + [f"f_{i:02d}" for i in range(10)] + ["target"],
            "test_columns": ["id"] + [f"f_{i:02d}" for i in range(10)],
            "sample_submission_columns": ["id", "target"],
            "dtypes": {"id": "int64", "f_00": "float64", "target": "int64"},
            "num_train_rows": 900000,
            "directory_files": ["train.csv", "test.csv", "sample_submission.csv"],
            "sample_rows": [
                {"id": 1, "f_00": 0.43, "target": 0},
                {"id": 2, "f_00": -1.2, "target": 1},
            ],
        },
        "tabular",
        "classification",
    ),
    (
        # Competition: new-york-city-taxi-fare-prediction
        # Characteristics: Lat/Long, Datetime, Target is continuous (fare_amount)
        {
            "train_columns": [
                "key",
                "fare_amount",
                "pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "passenger_count",
            ],
            "test_columns": [
                "key",
                "pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "passenger_count",
            ],
            "sample_submission_columns": ["key", "fare_amount"],
            "dtypes": {"fare_amount": "float64", "passenger_count": "int64"},
            "num_train_rows": 1000000,
            "directory_files": ["train.csv", "test.csv"],
            "sample_rows": [{"fare_amount": 12.50, "pickup_longitude": -73.99}],
        },
        "tabular",
        "regression",
    ),
    # ---------------------------------------------------------
    # 2. IMAGE CLASSIFICATION
    # ---------------------------------------------------------
    (
        # Competition: siim-isic-melanoma-classification
        # Characteristics: CSV points to image filenames (image_name), binary target
        {
            "train_columns": [
                "image_name",
                "patient_id",
                "sex",
                "age_approx",
                "target",
            ],
            "test_columns": ["image_name", "patient_id", "sex", "age_approx"],
            "sample_submission_columns": ["image_name", "target"],
            "directory_files": [
                "train.csv",
                "test.csv",
                "train/",
                "test/",
                "train/ISIC_01.jpg",
                "train/ISIC_02.jpg",
            ],
            "num_train_rows": 33126,
            "sample_rows": [{"image_name": "ISIC_2637011", "target": 0}],
        },
        "image",
        "classification",  # Note: Task type usually 'image_classification' or 'classification'
    ),
    (
        # Competition: aerial-cactus-identification
        # Characteristics: Directory of images, ID maps to has_cactus (0/1)
        {
            "train_columns": ["id", "has_cactus"],
            "test_columns": ["id"],
            "directory_files": [
                "train.csv",
                "sample_submission.csv",
                "train/",
                "test/",
                "train/0004be2cfeaba1c0361d32e2767df5a9.jpg",
            ],
            "dtypes": {"id": "object", "has_cactus": "int64"},
            "num_train_rows": 17500,
            "sample_rows": [
                {"id": "0004be2cfeaba1c0361d32e2767df5a9.jpg", "has_cactus": 1}
            ],
        },
        "image",
        "classification",
    ),
    # ---------------------------------------------------------
    # 3. TEXT CLASSIFICATION
    # ---------------------------------------------------------
    (
        # Competition: spooky-author-identification
        # Characteristics: Long text string, target is Author Name (EAP, HPL, MWS)
        {
            "train_columns": ["id", "text", "author"],
            "test_columns": ["id", "text"],
            "sample_submission_columns": ["id", "EAP", "HPL", "MWS"],
            "dtypes": {"text": "object", "author": "object"},
            "num_train_rows": 19579,
            "directory_files": ["train.csv", "test.csv", "sample_submission.csv"],
            "sample_rows": [
                {
                    "text": "This process, however, afforded me no means of ascertaining...",
                    "author": "EAP",
                },
                {"text": "It was the radiator of the heating system.", "author": "HPL"},
            ],
        },
        "text",
        "classification",
    ),
    (
        # Competition: jigsaw-toxic-comment-classification-challenge
        # Characteristics: Multi-label classification (toxic, severe_toxic, etc.)
        {
            "train_columns": [
                "id",
                "comment_text",
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ],
            "test_columns": ["id", "comment_text"],
            "dtypes": {"comment_text": "object", "toxic": "int64"},
            "directory_files": ["train.csv", "test.csv"],
            "sample_rows": [
                {"comment_text": "You are garbage.", "toxic": 1, "obscene": 1}
            ],
        },
        "text",
        "classification",
    ),
    # ---------------------------------------------------------
    # 4. AUDIO CLASSIFICATION
    # ---------------------------------------------------------
    (
        # Competition: the-icml-2013-whale-challenge-right-whale-redux
        # Characteristics: Directory of .aiff or .wav files, target is binary or class
        {
            "train_columns": ["clip_name", "label"],
            "test_columns": ["clip_name"],
            "directory_files": [
                "train.csv",
                "train/",
                "test/",
                "train/train1.aiff",
                "train/train2.aiff",
            ],
            "dtypes": {"clip_name": "object", "label": "int64"},
            "sample_rows": [{"clip_name": "train1.aiff", "label": 1}],
            "num_train_rows": 10000,
        },
        "audio",
        "classification",  # Or "audio_classification" depending on your prompt
    ),
    # ---------------------------------------------------------
    # 5. SEQ2SEQ (Text Normalization)
    # ---------------------------------------------------------
    (
        # Competition: text-normalization-challenge-english-language
        # Characteristics: Input token (before), Output token (after). String-to-String.
        {
            "train_columns": ["sentence_id", "token_id", "class", "before", "after"],
            "test_columns": ["sentence_id", "token_id", "before"],
            "sample_submission_columns": ["id", "after"],
            "dtypes": {"before": "object", "after": "object"},
            "directory_files": ["en_train.csv", "en_test.csv"],
            "sample_rows": [
                {"before": "2015", "after": "twenty fifteen", "class": "DATE"},
                {"before": "kg", "after": "kilograms", "class": "MEASURE"},
            ],
        },
        "seq2seq",  # or "text" depending on granularity, but prompt allows seq2seq
        "seq2seq",
    ),
    # ---------------------------------------------------------
    # 6. IMAGE REGRESSION / RESTORATION
    # ---------------------------------------------------------
    (
        # Competition: denoising-dirty-documents
        # Characteristics: Train folder (dirty), Clean folder (target).
        # NOTE: This is tricky for LLMs. It might map to Image Regression or Multimodal.
        {
            "train_columns": ["id", "dirty_path", "clean_path"],
            "directory_files": [
                "train/",
                "train_cleaned/",
                "test/",
                "train/1.png",
                "train_cleaned/1.png",
            ],
            "dtypes": {"id": "int64"},
            "sample_rows": [
                {
                    "id": 1,
                    "dirty_path": "train/1.png",
                    "clean_path": "train_cleaned/1.png",
                }
            ],
        },
        "image",
        # Task type might be ambiguous (regression/generation).
        # We accept either regression or image_classification if prompt is limited,
        # but ideally this is 'image_to_image' or 'regression'.
        "regression",
    ),
]


# --------------------------------------------------------------------------
# THE EVALUATION TEST FUNCTION
# --------------------------------------------------------------------------
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("metadata,expected_modality,expected_task", EVAL_CASES)
async def test_modality_accuracy(metadata, expected_modality, expected_task):
    """
    Benchmarks the Agent's reasoning capabilities against the Golden Dataset.
    Uses 'exact match' or 'substring match' to validate intelligence.
    """

    print("\n[WAIT] Sleeping 30s to respect Gemini free tier limits...")
    await asyncio.sleep(30)

    print(f"\n[EVAL] Testing: Expecting {expected_modality} | {expected_task}")

    # 1. Run the REAL Agent (Live API Call)
    result = await detect_modality_llm(metadata)

    print(f"[EVAL] Agent Predicted: {result['modality']} | {result['task_type']}")

    # 2. Soft Assertions (Case insensitive + Partial match handling)
    # We map common LLM variations to standard keys

    pred_modality = result["modality"].lower()
    pred_task = result["task_type"].lower()

    # Handle variations like "image_classification" vs "classification"
    task_match = (expected_task in pred_task) or (pred_task in expected_task)

    assert expected_modality in pred_modality, (
        f"Modality Mismatch! Expected {expected_modality}, got {pred_modality}"
    )

    assert task_match, f"Task Type Mismatch! Expected {expected_task}, got {pred_task}"

    # 3. Check consistency
    if expected_task == "classification":
        # If classification, agent should have found classes or a target column
        assert result.get("target_col"), "Classification task missing 'target_col'"
