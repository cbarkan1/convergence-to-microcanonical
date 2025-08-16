
# Generalized Coarse-Graining of a Billiard System

Code for the paper: *On the convergence of phase space distributions to microcanonical equilibrium: dynamical isometry and generalized coarse-graining*

Accepted to Journal of Physics A: Mathematical and Theoretical

📄 [Published article](https://doi.org/10.1088/1751-8121/ad7c9e) | 📄 [ArXiv version](https://arxiv.org/abs/2404.05123)

## What this does

Simulates an ensemble of billiard particle inside a triangular box and studies how they approach equilibrium. Compares the coarse-graining method introduced in the paper to the standard method.

## Usage

**Run the main simulation:**
```bash
python billiard_simulation.py
```

**Generate figures in paper:**
```bash
python Fig1ABC.py    # Figure 1 panels A, B, C
python Fig1D.py      # Figure 1 panel D  
python Fig2.py       # Figure 2
python FigS1.py      # Supplementary Figure 1
python FigS2.py      # Supplementary Figure 2
```

## Key components

- `billiard_utils.py` - Core billiard dynamics simulation (particle trajectories, collisions)
- `coarse_graining_classes.py` - Implementation of generalized coarse-graining methods
- `billiard_simulation.py` - Main simulation that combines both to study equilibration

## Requirements

- numpy
- scipy
- matplotlib  