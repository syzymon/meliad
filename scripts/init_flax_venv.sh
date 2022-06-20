#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y python3-venv

python3 -m venv ~/venvs/memorizing
source ~/venvs/memorizing/bin/activate
pip install --upgrade pip setuptools wheel

# initializes jax and installs ray on cloud TPUs
pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r flax_codebase/requirements.txt
