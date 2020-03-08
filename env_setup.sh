#!/bin/bash
#automate temp env setup in a compute node
#since no-index does not give versions of packages we want, we'll create the env in the login node
#then load it here
#tf, keras, numpy, plt, torch

module load python/3.6
module load scipy-stack

#make env in temp
#virtualenv --no-download $SLURM_TMPDIR/env

#activate pre made env
activate(){
  . $HOME/envs/torch/bin/activate
}
activate

#$SLURM_TMPDIR/env/bin/activate

#pip install --no-index --upgrade pip
#pip install --no-index tensorflow_gpu
#pip install --no-index keras
#pip install --no-index imgaug
#pip install --no-index torch
