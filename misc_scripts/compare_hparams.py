import argparse
import difflib
import json

import wandb
import wandb.apis


def get_config(api, project_name, run_name):
    run = api.run(f"{project_name}/{run_name}")
    config = run.config
    config_str = json.dumps(config, indent=2)
    return config_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", default="chord_tones_musicbert")
    parser.add_argument("run1")
    parser.add_argument("run2")
    args = parser.parse_args()
    return args


def main():
    # Initialize the W&B API
    api = wandb.Api()

    args = parse_args()

    # Specify the project name and run name you want to fetch
    # project_name = "chord_tones_musicbert"
    config1 = get_config(api, args.project_name, args.run1)
    config2 = get_config(api, args.project_name, args.run2)

    # Create a Differ object
    differ = difflib.Differ()

    # Compute the difference between the two strings
    diff = list(
        differ.compare(
            config1.splitlines(keepends=True), config2.splitlines(keepends=True)
        )
    )
    diff = [l for l in diff if l[:2] in {"- ", "+ "}]
    print("".join(diff))


if __name__ == "__main__":
    main()
