# Axial Turbine Optimization

A direct optimization framework for axial turbine design using automated simulation workflows and evolutionary algorithms.

This repository provides a structured pipeline for optimizing axial turbine configurations through parameter exploration, simulation execution, and objective evaluation. It is designed for research and engineering workflows and supports HPC cluster execution.

---

## Overview

This project enables:

- Automated optimization of axial turbine design parameters  
- Integration with simulation workflows via XML state templates  
- Differential Evolution–based optimization using pygmo  
- Reproducible experiments through controlled random seeds  
- HPC-ready execution using batch submission scripts  

The framework is modular and can be adapted for different turbine models, objective functions, and optimization strategies.

---

## Repository Structure

    axialTurbine_optimization/
    │
    ├── boundaryData_RU_INLET/     # Boundary condition data
    ├── gmshMeshFile/              # Mesh files for geometry/simulation
    ├── models/
    │   └── axialTurbine.py        # Core axial turbine model definition
    │
    ├── xml/                       # XML configuration templates
    ├── templateState.xml          # Base simulation state template
    │
    ├── opt_build.py               # Main optimization driver
    ├── utils.py                   # Utility functions (seed setup, device config)
    ├── sbatch_opt_T5.pbs          # HPC batch submission script
    └── README.md

---

## Requirements

Before running the optimization, ensure the following are installed:

- Python 3.7+
- pygmo  
- torch  

Install required Python packages:

    pip install pygmo torch

Additional external tools may be required depending on how `models/axialTurbine.py` interfaces with simulation software.

---

## Running the Optimization

To start the optimization:

    python opt_build.py

The script will:

1. Set random seeds for reproducibility  
2. Initialize optimization bounds from the turbine model  
3. Run a Differential Evolution optimizer  
4. Evaluate turbine performance using the defined model  
5. Track and report optimization progress  

---

## Optimization Workflow

The general workflow is:

1. Define design variables and bounds in `models/axialTurbine.py`
2. Generate candidate designs using an evolutionary algorithm
3. Update simulation state files (XML-based templates)
4. Run simulation or evaluation
5. Compute objective function (e.g., efficiency, head deviation, torque)
6. Iterate until convergence

---

## Customization

You can modify:

- Optimization parameters (population size, mutation, etc.) in `opt_build.py`
- Objective function definition in `models/axialTurbine.py`
- Simulation templates in `xml/` and `templateState.xml`
- HPC configuration in `sbatch_opt_T5.pbs`

---

## Reproducibility

The `utils.py` module ensures:

- Controlled random seed initialization
- Device setup (CPU/GPU detection if applicable)

This allows experiments to be replicated reliably.

---

## HPC Usage

For cluster execution:

    sbatch sbatch_opt_T5.pbs

Modify the PBS script as needed to match your cluster configuration.

---

## Extending the Project

Possible extensions include:

- Multi-objective optimization (e.g., NSGA-II)
- Surrogate modeling integration
- Latent-space optimization using generative models
- Automated post-processing of simulation results
- Logging and experiment tracking tools

---

## License



---

## Author

Developed for axial turbine design research and optimization workflows.

