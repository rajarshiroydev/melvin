# test_modality.py

import asyncio
from pathlib import Path
from modality_detector import collect_dataset_metadata, detect_modality_llm


async def main():
    public_dir = (
        Path.home()
        / "Library/Caches/mle-bench/data/text-normalization-challenge-english-language/prepared/public"
    )

    metadata = collect_dataset_metadata(public_dir)
    print("\n=== METADATA ===")
    print(metadata)

    result = await detect_modality_llm(metadata)
    print("\n=== LLM RESULT ===")
    print(result)


asyncio.run(main())
