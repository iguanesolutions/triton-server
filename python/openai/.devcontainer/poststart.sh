#!/usr/bin/env bash
set -e
pip install /opt/tritonserver/python/triton*.whl
pip install -r requirements-test.txt
pip install -r requirements.txt
pip install uvicorn
