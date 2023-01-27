#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Destroy and recreate the venv
./dev/venv-delete.sh
./dev/venv-create.sh

# Remove caches
./dev/rm-caches.sh

# Remove poetry.lock file
rm -f poetry.lock

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==22.2
python -m pip install -U setuptools==63.2.0
python -m pip install -U poetry==1.1.14

# Use `poetry` to install project dependencies
python -m poetry install
