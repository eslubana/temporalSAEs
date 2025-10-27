#!/usr/bin/env python3
"""
Upload a directory to HuggingFace Hub as a model repository.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, login
from huggingface_hub.utils import RepositoryNotFoundError


def upload_directory_to_hf(config):
    """Upload a local directory to HuggingFace Hub."""

    local_path = Path(config['local_dir'])
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        print('Repository found, checking...')
        api.repo_info(repo_id=config['repo_id'])
    except RepositoryNotFoundError:
        print('Repository not found, creating...')
        create_repo(repo_id=config['repo_id'])

    # Upload the directory
    upload_folder(
        folder_path=str(local_path),
        repo_id=config['repo_id'],
        repo_type="model",
        path_in_repo=config['path_in_repo']
    )


def main():
    list_saes = os.listdir("final_saes/llama/layer_15/")
    for sae in list_saes:
        config = {
            'local_dir': "" + sae,
            'repo_id': "",
            'path_in_repo': 'layer_15/' + sae,
        }
        upload_directory_to_hf(config)
        print(f"Successfully uploaded {config['local_dir']} to https://huggingface.co/{config['repo_id']}/tree/main/{config['path_in_repo']}")


if __name__ == "__main__":
    login()
    main()