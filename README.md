# ResiliAI: A Multi-Agent Reinforcement Learning Framework for AI-Driven Economic Shock Resilience

## Repository Overview

This repository contains the code, data, and documentation for the ResiliAI framework, a novel approach that integrates Multi-Agent Reinforcement Learning (MARL), Digital Twin simulation, and Explainable AI to model, analyze, and enhance economic resilience.

## Framework Components

The ResiliAI framework consists of three main components:

1. **Digital Twin Economy**: Creates virtual replicas of economic systems using real-time data streams
2. **Resilience Scoring Engine**: Identifies key factors affecting economic stability using explainable AI
3. **Multi-Agent Reinforcement Learning Framework**: Simulates adaptive behavior of economic agents under various conditions

## Repository Structure

```
ESI2025_ResiliAI/
├── code/                       # Source code for the ResiliAI framework
│   ├── src/                    # Core modules
│   │   ├── data_preprocessing.py  # Data preprocessing module
│   │   ├── digital_twin.py     # Digital Twin Economy simulation
│   │   ├── marl_framework.py   # Multi-Agent Reinforcement Learning framework
│   │   ├── resilience_scoring.py  # Resilience Scoring Engine
│   │   └── experiment_runner.py   # Experiment orchestration
│   ├── notebooks/              # Jupyter notebooks for analysis and visualization
│   ├── tests/                  # Unit and integration tests
│   └── run_experiments.py      # Main script to run experiments
├── data/                       # Data directory
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── results/                    # Experiment results
│   ├── visualizations/         # Generated visualizations
│   └── report/                 # Summary reports
├── paper/                      # Research paper files
│   ├── extended_abstract.tex   # Extended abstract for ESI2025
│   ├── full_paper.tex          # Full paper for ESI2025
│   └── figures/                # Paper figures
├── analysis/                   # In-depth analysis documents
├── README.md                   # Repository overview (this file)
├── requirements.txt            # Python dependencies
└── LICENSE                     # License information
```

## Installation

```bash
# Clone the repository
git clone https://github.com/alosbeh/ResiliAI.git
cd ResiliAI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
# Run experiments with default parameters
python code/run_experiments.py

# Run experiments with custom parameters
python code/run_experiments.py --simulation_steps 60 --num_trials 5
```

### Available Command-line Arguments

- `--simulation_steps`: Number of simulation steps (default: 60)
- `--num_trials`: Number of trials for MARL experiments (default: 5)
- `--data_dir`: Directory for input data (default: data/)
- `--results_dir`: Directory for output results (default: results/)
- `--random_seed`: Random seed for reproducibility (default: 42)

## Key Features

- **Economic Shock Simulation**: Model pandemic, financial crisis, and climate shock scenarios
- **Policy Intervention Testing**: Evaluate fiscal stimulus, monetary easing, and targeted support policies
- **Resilience Factor Identification**: Identify key factors affecting economic stability
- **Agent Adaptation Analysis**: Study how economic agents adapt to changing conditions
- **Explainable Results**: Transparent insights through SHAP values and feature importance analysis

## Data Sources

The framework uses data from:
- IMF Fiscal Policies Database in Response to COVID-19
- World Bank economic indicators
- Additional synthetic data for testing and validation

## Paper Abstract

Economic resilience to shocks such as pandemics, financial crises, and climate disasters has become a critical concern for policymakers worldwide. Traditional economic models often fail to capture the complex, adaptive nature of economic systems under stress. This paper introduces ResiliAI, a novel framework that integrates Multi-Agent Reinforcement Learning (MARL), Digital Twin simulation, and Explainable AI to model, analyze, and enhance economic resilience. ResiliAI enables the simulation of heterogeneous economic agents adapting to shocks in realistic environments, provides real-time "what-if" policy experimentation without affecting real markets, and offers a resilience scoring engine that identifies key factors affecting economic stability. We demonstrate the framework's capabilities through comprehensive experiments using IMF fiscal policy data, showing how different policy interventions affect recovery trajectories across various shock scenarios. Results reveal that fiscal stimulus reduces pandemic recovery time by 37%, monetary easing improves liquidity by 42% during financial crises, and targeted support reduces inequality metrics by 31% across all shock types. The framework also identifies critical structural factors for resilience, including economic diversity, robust social safety nets, and flexible labor markets. ResiliAI advances the frontier of AI-driven economic modeling by enabling more adaptive, forward-looking policy design for sustainable and resilient economies.

## Citation

If you use this code or the ideas presented in the paper, please cite:

```
@inproceedings{alsobeh2025resiliai,
  title={ResiliAI: A Multi-Agent Reinforcement Learning Framework for AI-Driven Economic Shock Resilience and Adaptive Policy Simulation},
  author={ALsobeh, Anas},
  booktitle={8th International Conference on Entrepreneurship for Sustainability \& Impact (ESI2025)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Massachusetts Institute of Technology for research support
- International Monetary Fund for fiscal policy data
- The open-source community for tools and libraries
