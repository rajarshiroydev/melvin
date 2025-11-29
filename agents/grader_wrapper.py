import sys
import argparse
import importlib
import importlib.util
import pandas as pd
from pathlib import Path


# ----------------------------------------------------------------------
# Logic to load MLEbench modules dynamically (handles symlinks)
# ----------------------------------------------------------------------
def load_module_from_path(repo_root, module_path_str):
    repo_root = Path(repo_root)
    module_name_full, fn_name = module_path_str.split(":")

    # Case 1: Simple module (e.g., "mlebench.metrics:accuracy")
    if "-" not in module_name_full:
        try:
            module = importlib.import_module(module_name_full)
            return getattr(module, fn_name)
        except ImportError:
            pass  # Fall through to manual loading if simple import fails

    # Case 2: Complex path with symlinks (MLEbench specific)
    parts = module_name_full.split(".")
    clean_parts = []
    for part in parts:
        if "-" in part:
            underscore_name = part.replace("-", "_")
            clean_parts.append(underscore_name)
        else:
            clean_parts.append(part)

    clean_module_path = ".".join(clean_parts)
    try:
        module = importlib.import_module(clean_module_path)
        return getattr(module, fn_name)
    except ImportError:
        # Final fallback: force import the original name
        module = importlib.import_module(module_name_full)
        return getattr(module, fn_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, help="Path to mle-bench root")
    parser.add_argument(
        "--grader_module",
        required=True,
        help="Import path string (e.g., mlebench...:grade)",
    )
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument("--answers", required=True, help="Path to private test.csv")
    args = parser.parse_args()

    # 1. Add repo root to sys.path so imports work
    sys.path.append(args.repo_root)

    try:
        # 2. Load the grader function
        grade_fn = load_module_from_path(args.repo_root, args.grader_module)

        # 3. Load DataFrames
        sub_df = pd.read_csv(args.submission)
        ans_df = pd.read_csv(args.answers)

        # 4. Data Hygiene: Ensure ID columns match types (str vs int vs float)
        # This fixes common bugs where one is '1202' and the other is 1202.0
        if "id" in sub_df.columns and "id" in ans_df.columns:
            if sub_df["id"].dtype != ans_df["id"].dtype:
                try:
                    sub_df["id"] = sub_df["id"].astype(ans_df["id"].dtype)
                except Exception:
                    pass

        # 5. Run Grading
        score = grade_fn(sub_df, ans_df)

        # 6. Print Score with a specific delimiter for the Orchestrator to catch
        print(f"SCORE_OUTPUT:{score}")

    except Exception as e:
        # Print the full traceback to stderr so Orchestrator logs it
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
