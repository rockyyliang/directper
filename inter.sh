#!/bin/bash
salloc --account=def-cogdrive --gres=gpu:v100:2 --cpus-per-task=20 --mem=47G --time=0-02:00:00

