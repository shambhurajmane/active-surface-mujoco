# Active Surface MuJoCo Simulation Platform

This repository contains the MuJoCo-based simulation platform developed for our manipulation research paper.  
It supports regrasping and in-hand manipulation tasks with active surface grippers and object-centric task definitions.

## Features
- MuJoCo models for active surface grippers, Variable friciton gripper and Belt-orienting Phalanges  
- Lightweight evaluation and visualization scripts  

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate mujoco_env


### Pip 
pip install -r requirements.txt

### Run a minimal example:
VF experiments
python .\scripts\VF\vf_experiments.py


BOP experiments
python .\scripts\BOP\BOP_experiments.py

### Citation
TBD


### License
MIT License
