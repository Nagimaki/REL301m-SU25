# Constitutional AI Training Pipeline

This repository provides a complete implementation of a Constitutional AI training pipeline using GPT-2, including data generation, training with RLAIF (Reinforcement Learning from AI Feedback), comprehensive evaluation, and visualization tools.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Run in Jupyter Notebook](#run-in-jupyter-notebook)
  - [Run as a Script](#run-as-a-script)
  - [Quick Setup Test](#quick-setup-test)
- [Project Structure](#project-structure)
- [Logging & Monitoring](#logging--monitoring)
- [License](#license)

## Features

- **Modular design**: Separate classes for metrics tracking, constitutional critique, and RLAIF reward modeling.
- **Automated data generation**: Generate large, diverse training prompts.
- **Comprehensive evaluation**: Pre- and post-training comparisons with detailed metrics.
- **Visualization tools**: Static plots and an interactive dashboard to monitor training progress.

## Prerequisites

- Python **3.8** or higher
- `git` for cloning the repository
- (Optional) A CUDA-enabled GPU for accelerated training

## Setup

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to isolate dependencies.

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install --upgrade pip
pip install transformers torch datasets accelerate peft trl wandb numpy matplotlib seaborn plotly
```

Alternatively, if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

## Configuration

- The training script uses Weights & Biases (`wandb`) for experiment logging. If you wish to track runs, log in with:
  ```bash
  wandb login
  ```
- You can adjust hyperparameters (e.g., learning rate, batch size) directly in the training cells or script.

## Usage

### Run in Jupyter Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open the provided notebook or script file.
3. Execute cells sequentially from **Cell 1** to **Cell 19**.

### Run as a Script

A main execution function is provided. To run the full demo end-to-end:

```bash
python run_training.py
```

Replace `run_training.py` with your script name that wraps the `run_complete_demo()` call.

### Quick Setup Test

To verify your environment setup without running full training, execute the `quick_test()` function:

```python
from main import quick_test
if quick_test():
    print("✅ Environment is set up correctly!")
```

## Project Structure

```
├── data/                  # (Optional) Custom data directory
├── main.py                # Entry point for training/demo scripts
├── metrics.py             # MetricsTracker class definition
├── critic.py              # ConstitutionalCritic class definition
├── reward_model.py        # RLAIFRewardModel class definition
├── trainer.py             # ConstitutionalAITrainer class definition
├── visualizer.py          # Visualization utilities
├── utils.py               # Data generation and evaluation functions
├── requirements.txt       # Python dependencies (if provided)
└── README.md              # Project documentation
```

## Logging & Monitoring

- Training logs are output via Python's `logging` module at INFO level.
- Metrics such as loss, constitutional scores, response lengths, and diversity are tracked and can be visualized.
- Static plots are saved (e.g., `training_progress.png`, `principle_comparison.png`).
- An interactive Plotly dashboard is also available through the `Visualizer.create_interactive_dashboard()` method.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

