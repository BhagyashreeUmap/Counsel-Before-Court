import json
import os
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INPUT_FILE = Path("/aikart/input.json")
OUTPUT_FILE = Path("/aikart/output.json")


def main():
    # Read aiKart input
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # aiKart may also provide the same JSON through this env variable
        data = json.loads(os.environ.get("AIKART_INPUT", "{}"))

    case_description = data.get("case_description", "").strip()
    language = data.get("language", "English")

    if not case_description:
        response = "Please provide a description of your legal issue."
    else:
        # Lightweight aiKart entry point for the Counsel Before Court agent.
        # Full interactive workflow is available in the Streamlit prototype.
        response = f"""# Counsel Before Court

**Language:** {language}

## Case received
{case_description}

Your case has been received for legal-information and case-preparation support.

Counsel Before Court helps citizens organize their situation, identify information and documents that may matter, learn from similar case patterns, and prepare questions for a legal professional.

> **Important:** This system provides legal information and case preparation only. It does not provide legal advice, replace a lawyer, or predict case outcomes.
"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "markdown",
                "response": response
            },
            f,
            ensure_ascii=False
        )


if __name__ == "__main__":
    main()