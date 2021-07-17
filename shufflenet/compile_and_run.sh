#!/bin/bash
aipubuild config/onnx_shufflenet_build.cfg
aipubuild config/onnx_shufflenet_run.cfg
python3 validate_output.py
