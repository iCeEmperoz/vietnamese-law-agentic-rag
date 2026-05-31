"""Root-level entrypoint for the Vietnamese Law Ingestion & Indexing Pipeline."""
import sys
import argparse
from pathlib import Path

# Add project root directory to Python path
# Path(__file__).resolve().parent is the 'scripts' folder, so .parent.parent is the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.scripts.ingest import run_ingestion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Law Ingestion Pipeline")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit the number of documents processed (e.g. --sample 1000)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to raw Vietnamese Law document files"
    )
    
    args = parser.parse_args()
    run_ingestion(sample_size=args.sample, data_path=args.data_path)
