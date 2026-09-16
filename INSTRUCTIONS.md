# Project Setup and Execution Instructions

This document provides step-by-step instructions to configure the environment and run each of the Reinforcement Learning projects in this repository.

---

## Prerequisites

* **Python Version:** Python 3.12 (required)
* **Operating System:** Linux, macOS, or Windows (WSL recommended)

---

## 1. Installing Python 3.12 (via pyenv)

If Python 3.12 is not already installed on your system, it is recommended to manage Python versions using `pyenv`.

### Installing `pyenv`

On Linux / macOS:
```bash
curl https://pyenv.run | bash
```

Add the following environment configuration to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Apply the changes:
```bash
source ~/.bashrc
```

### Installing Python 3.12

Install Python 3.12.0 using `pyenv` and set it locally for this repository:

```bash
pyenv install 3.12.0
pyenv local 3.12.0
```

Verify the active Python version:
```bash
python --version
```

---

## 2. Virtual Environment Setup

Navigate to the project root directory (`deep_rl`) and follow these steps to initialize and populate the virtual environment.

### Step 2.1: Create Virtual Environment

Create a virtual environment named `.venv`:

```bash
python3.12 -m venv .venv
```

### Step 2.2: Activate Virtual Environment

* **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell):**
  ```bash
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 2.3: Upgrade Package Installer and Install Dependencies

Install the required packages specified in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Projects

Ensure the virtual environment is activated (`source .venv/bin/activate`) before running any script.

### 3.1 Lunar Lander (Deep RL - PPO)

Trains a Proximal Policy Optimization (PPO) agent on `LunarLander-v3` across 16 parallel environments, saves the model checkpoint to `lunar_lander_test.zip`, and evaluates mean policy reward.

```bash
python lunar_lander.py
```

### 3.2 Taxi-v4 (Tabular Q-Learning)

Trains a custom tabular Q-Learning agent on `Taxi-v4` over 25,000 episodes, calculates benchmark metrics, and launches visual rendering mode (`render_mode="human"`).

```bash
python taxi.py
```

### 3.3 Frozen Lake (Tabular Q-Learning)

Trains a custom Q-Learning agent on `FrozenLake-v1` (4x4 non-slippery grid) using an exponential epsilon decay schedule and outputs performance metrics.

```bash
python frozen_lake.py
```

### 3.4 BEMS (Building Energy Management System Environment)

Initializes and runs the custom Gymnasium environment for energy management and task scheduling.

```bash
python BEMS.py
```

---

## 4. System Dependencies & Troubleshooting

* **Box2D / Swig Build Requirements:**
  If installing `Box2D` or running `lunar_lander.py` encounters build errors, ensure system build tools and `swig` are installed on your OS:
  ```bash
  # Debian / Ubuntu
  sudo apt-get update
  sudo apt-get install -y build-essential swig
  ```

* **Headless Display for Visual Evaluation:**
  `taxi.py` includes a visual evaluation function using `render_mode="human"`. If running on a headless server without a GUI display, you may need a virtual framebuffer (`xvfb-run python taxi.py`) or change `render_mode="human"` to `render_mode="rgb_array"` in `taxi.py`.
