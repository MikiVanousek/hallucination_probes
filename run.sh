#!/bin/bash
CUDA_VISIBLE_DEVICES=0 ../.local/binuv run python -m probe.train --config $1