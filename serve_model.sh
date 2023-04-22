#!/bin/bash

export MLFLOW_TRACKING_URI=http://0.0.0.0:5000

mlflow models serve -m $1 -p 8000 -h 0.0.0.0 --no-conda