"""Create a deduplicated Project 2 copy without changing the Project 1 source."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "project1" / "data" / "online_shoppers_intention.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "online_shoppers_intention_deduplicated.csv"


def main() -> None:
    if SOURCE_PATH.resolve() == OUTPUT_PATH.resolve():
        raise ValueError("Output must be a new file; the source dataset must remain unchanged.")

    source = pd.read_csv(SOURCE_PATH)
    deduplicated = source.drop_duplicates(keep="first")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(OUTPUT_PATH, index=False)

    removed = len(source) - len(deduplicated)
    print("╭────────────────────────────────────────────╮")
    print("│  PROJECT 2  /  BUILD CLEAN DATASET          │")
    print("╰────────────────────────────────────────────╯")
    print(f"Source rows       {len(source):>8,}")
    print(f"Duplicates removed{removed:>8,}")
    print(f"Rows saved        {len(deduplicated):>8,}")
    print(f"Output            {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
