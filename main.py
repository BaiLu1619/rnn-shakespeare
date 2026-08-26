"""Run the complete Character RNN pipeline with one command."""

import argparse

from train import run_pipeline
from util.config import load_config
from util.data_loader import download_tiny_shakespeare


def main() -> None:
    parser = argparse.ArgumentParser(description="Character-RNN")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="download and verify Tiny Shakespeare, then exit",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.download_data:
        download_tiny_shakespeare(config["data_dir"])
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
