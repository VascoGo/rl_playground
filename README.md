# Deep Reinforcement Learning Repository

## Overview

This repository contains a suite of Reinforcement Learning (RL) projects developed as a solo learning codebase. The repository demonstrates progression across core RL paradigms: from discrete tabular Q-Learning algorithms to deep policy gradient methods using modern frameworks, as well as domain-specific environment design using Gymnasium.

The primary goal of this codebase is to study, implement, and evaluate foundational and advanced RL algorithms across standard benchmark environments and custom problem formulations.

---

## Implemented Projects

### 1. Lunar Lander (Deep Reinforcement Learning)
* **Environment:** `LunarLander-v3` (Gymnasium)
* **Algorithm:** Proximal Policy Optimization (PPO) via `stable-baselines3`
* **Key Components:**
  * Vectorized parallel environment training (`make_vec_env` with 16 parallel environments).
  * Hyperparameter configuration tailored for continuous state navigation ($N_{\text{steps}}=1024$, batch size 64, $\gamma=0.999$, $\lambda=0.98$, $\text{ent\_coef}=0.01$).
  * Model weight serialization upon training completion and deterministic policy evaluation.

### 2. Taxi-v4 (Tabular Q-Learning)
* **Environment:** `Taxi-v4` (Gymnasium)
* **Algorithm:** Custom Tabular Q-Learning
* **Key Components:**
  * Custom Q-table initialization and Epsilon-Greedy action selection strategy.
  * Linear decay schedule for exploration rate ($\epsilon$).
  * Metric evaluation pipeline supporting both benchmark statistical evaluation and real-time visual playback (`render_mode="human"`).

### 3. Frozen Lake (Tabular Q-Learning)
* **Environment:** `FrozenLake-v1` (Gymnasium, 4x4 non-slippery variant)
* **Algorithm:** Custom Tabular Q-Learning
* **Key Components:**
  * Exponential exploration decay rate schedule:
    $$\epsilon(e) = \epsilon_{\text{min}} + (\epsilon_{\text{max}} - \epsilon_{\text{min}}) \cdot e^{-\lambda \cdot e}$$
  * Quantitative performance benchmark measuring mean reward and standard deviation across evaluation episodes.

### 4. Custom Environment: Building Energy Management System (BEMS)
* **Environment:** Custom Gymnasium Environment (`BEMS.py`)
* **Objective:** Models a dynamic energy management agent designed to schedule task queues while managing battery state of charge (SoC) and satisfying operational deadlines.
* **State & Action Spaces:**
  * **Observation Space:** Structured `Dict` space containing agent status parameters (SoC, charging status, active task index, remaining steps), task deadline array, and task mask.
  * **Action Space:** `MultiDiscrete([2, task_queue + 1])` enabling joint control over battery charging state and task execution slot assignment.
* **Reward Function:** Multi-objective penalty and reward formulation balancing step costs, battery exhaustion penalties, deadline violations, overcharging, and task completion bonuses.

---

## Repository Structure

```
deep_rl/
├── BEMS.py          # Custom Gymnasium environment implementation for energy management
├── frozen_lake.py   # Tabular Q-Learning algorithm applied to FrozenLake-v1
├── lunar_lander.py  # Deep RL (PPO) agent trained on LunarLander-v3
├── taxi.py          # Tabular Q-Learning algorithm applied to Taxi-v4 with visual evaluation
├── requirements.txt # Python dependencies for all projects
├── INSTRUCTIONS.md  # Setup, environment creation, and execution guide
└── README.md        # Repository documentation
```

---

## Technical Stack

* **Language:** Python 3.12
* **Reinforcement Learning:** Gymnasium, Stable-Baselines3
* **Deep Learning Framework:** PyTorch
* **Scientific Computing & Utilities:** NumPy, Matplotlib, Box2D, tqdm, Rich

---

## Setup and Running Instructions

To configure Python 3.12, set up the virtual environment, install dependencies, and run the scripts, please consult the execution guide:

[View Instructions Document](INSTRUCTIONS.md)

---

## Author

Developed as an independent research and portfolio project in Deep Reinforcement Learning.
