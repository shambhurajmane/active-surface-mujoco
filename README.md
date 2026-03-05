# Active Surface MuJoCo Simulation Platform
![Active Surface Simulation](docs/intro.png)

This repository contains the MuJoCo-based simulation platform developed for our manipulation research paper.  
It supports regrasping and in-hand manipulation tasks with active surface grippers and object-centric task definitions.


### CAD Model Attribution

The CAD models of the BOP gripper used in this project are derived from the work of **Gregory Xie**, originally published in the MIT thesis:

*Xie, G. (2023). "Don’t Over Think It: Mechanically Intelligent Manipulation."  
Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology.*

Source: https://dspace.mit.edu/handle/1721.1/151343

All credit for the original mechanical design and CAD files belongs to the author. The models are used here solely for academic research and benchmarking of the manipulation planning algorithm.

## Features
- MuJoCo xml files for active surface grippers, Variable friciton gripper and Belt-orienting Phalanges  
- Lightweight evaluation and visualization scripts  

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate mujoco_env
```

### Pip 
```bash
pip install -r requirements.txt
```
 
##  Repository Structure
```text
mujoco-active-surface/
│
├── assets/ # Meshes and CAD models
│ ├── BOP_CAD/ # CAD models for BOP gripper
│ ├── VF_CAD/ # CAD models for VF gripper
│ ├── Object_Assets/ # Simple test objects
│ └── YCB_Objects_Test_Cases/ # YCB object models
│
├── mujoco/ # MuJoCo MJCF/XML models
│ ├── XML_files_for_gripper/ # Gripper XML definitions
│ ├── test_cases/ # Task-specific MuJoCo setups
│ └── scenes/ # Complete simulation scenes
│
├── scripts/ # Run, evaluation, and visualization scripts
│ ├── BOP/ # Scripts for BOP gripper experiments
│ └── VF/ # Scripts for VF gripper experiments
│
├── configs/ # YAML/JSON configuration files
├── docs/ # Diagrams, videos, and additional documentation
│
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
├── LICENSE # License information
└── CITATION.cff # Citation metadata
```

## Run a minimal example:
VF experiments
```bash
python .\scripts\VF\vf_experiments.py
```

BOP experiments
```bash
python .\scripts\BOP\BOP_experiments.py
```

### Citation
TBD

BOP CAD files were taken from Xie, Gregory's thesis work posted at Don’t Over Think It: Mechanically Intelligent Manipulation, MIT library.
https://dspace.mit.edu/handle/1721.1/151343?show=full
Please cite their work.
@article{xie2023hand,
  title={In-hand manipulation with a simple belted parallel-jaw gripper},
  author={Xie, Gregory and Holladay, Rachel and Chin, Lillian and Rus, Daniela},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  number={2},
  pages={1334--1341},
  year={2023},
  publisher={IEEE}
}

### License
MIT License
