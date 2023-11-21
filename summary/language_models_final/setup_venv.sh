#!/usr/bin/env bash

python -m venv venv || exit 1
source venv/bin/activate
python -m pip install -r requirements.txt
