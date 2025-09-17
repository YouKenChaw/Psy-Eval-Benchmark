#!/bin/bash

export PYTHONPATH=(dirname "$0")/..:${PYTHONPATH:-}

python ./experiments/run_example.py