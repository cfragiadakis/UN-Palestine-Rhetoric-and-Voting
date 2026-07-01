"""Build the consolidated UN General Debate speeches CSV.

This script reads the raw text files in Data/TXT and writes Data/AllSpeeches.csv.
Run it from the project root with:

    python scripts/build_all_speeches_csv.py
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def build_speeches_csv(output_path: Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Data" / "TXT"
    if output_path is None:
        output_path = project_root / "Data" / "AllSpeeches.csv"

    rows = []

    for session in range(1, 80):
        session_str = f"{session:02d}"
        directory = data_dir / f"Session {session_str} - {1945 + session}"

        if not directory.exists():
            continue

        for file_path in sorted(directory.iterdir()):
            if file_path.name.startswith(".") or not file_path.is_file():
                continue

            speech_text = file_path.read_text(encoding="utf-8")

            country_code = file_path.name.split("_", 1)[0]
            rows.append([session, 1945 + session, country_code, speech_text])

    df_speech = pd.DataFrame(
        rows,
        columns=["Session", "Year", "ISO-alpha3 Code", "Speech"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_speech.to_csv(output_path, index=False)

    print(f"Saved {len(df_speech)} speeches to {output_path}")
    return output_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the generated CSV. Defaults to Data/AllSpeeches.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_speeches_csv(args.output)
