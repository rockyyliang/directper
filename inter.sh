#!/bin/bash
salloc --account=def-cogdrive --gres=gpu:p100:2 --cpus-per-task=20 --mem=36G --time=0-02:00:00

