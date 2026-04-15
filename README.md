# NUDGE — Navigation Using DDPG-Guided Estimation

> **Simulation-first framework for GPS-denied autonomous drone positioning** — fuses UWB ranging from four fixed anchors with a three-stage DSP pipeline and a DDPG reinforcement learning agent that adaptively tunes Kalman filter parameters in real time.  
> Validated against the **IEEE 802.15.4a CM4 Industrial NLOS** channel model. 

---

## Overview

GPS fails in underground mine shafts, deep open-pit excavations, and dense-canopy post-mining sites. NUDGE solves this with a two-layer architecture:

| Layer | What it does |
|---|---|
| **DSP Pipeline** | Cleans raw UWB ToA measurements — NLOS bias removal, multipath mitigation, range smoothing |
| **DDPG Agent** | Treats EKF noise covariance matrices Q and R as continuous action space; learns to tune them online based on positioning error feedback |

The result: a Kalman filter that adapts its trust in the model vs. the sensors in real time, dramatically outperforming a fixed-covariance EKF under the harsh NLOS conditions of industrial mine environments.

---

## System Architecture

```
4 × UWB Anchors (fixed, known positions)
        │
        │  Time-of-Arrival measurements (with NLOS noise)
        ▼
┌───────────────────────────────────────┐
│         3-Stage DSP Pipeline          │
│  Stage 1: NLOS Bias Estimation        │
│  Stage 2: Multipath Rejection Filter  │
│  Stage 3: Range Smoothing (IIR)       │
└───────────────────┬───────────────────┘
                    │  Cleaned ranges
                    ▼
┌───────────────────────────────────────┐
│       Extended Kalman Filter (EKF)    │
│   State: [x, y, z, vx, vy, vz]       │
│   Q, R ← tuned dynamically by DDPG   │
└───────────────────┬───────────────────┘
                    │  Estimated position
                    ▼
┌───────────────────────────────────────┐
│          DDPG Agent (Actor-Critic)    │
│   Observation: positioning error,     │
│                innovation sequence,   │
│                range residuals        │
│   Action:  ΔQ, ΔR (covariance delta) │
│   Reward:  −‖p_est − p_true‖²        │
└───────────────────────────────────────┘
```

---

## Repository Structure

```
NUDGE/
├── uwb_ddpg/
│   ├── *.m                  ← MATLAB simulation scripts
│   └── Plot Images/         ← Result figures (RMSE, trajectory, reward curves)
├── LICENSE
└── README.md
```

---

## Channel Model

| Parameter | Value |
|---|---|
| Standard | IEEE 802.15.4a |
| Profile | CM4 — Industrial NLOS |
| Anchors | 4 (fixed, 3D positions) |
| UWB Frequency | ~4–8 GHz |
| Ranging Method | Two-Way Time-of-Arrival (TW-ToA) |
| NLOS Bias | Modelled as positive exponential random variable |
| Multipath | Ricean / Rayleigh fading per CM4 profile |

---

## DDPG Configuration

| Hyperparameter | Value |
|---|---|
| Actor network | 3-layer MLP, [256, 128, 64], ReLU |
| Critic network | 3-layer MLP, [256, 128, 64], ReLU |
| Replay buffer | 100,000 transitions |
| Batch size | 64 |
| Learning rate (Actor) | 1 × 10⁻⁴ |
| Learning rate (Critic) | 1 × 10⁻³ |
| Discount factor γ | 0.99 |
| Soft update τ | 0.005 |
| Exploration noise | Ornstein-Uhlenbeck (σ = 0.2) |

---

## Requirements

- **MATLAB R2021b or later**
- Toolboxes required:
  - Reinforcement Learning Toolbox
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

---

## Running the Simulation

### 1. Clone the repository

```bash
git clone https://github.com/Shrinivasv2910/NUDGE.git
cd NUDGE/uwb_ddpg
```

### 2. Open MATLAB and run

```matlab
% Add folder to path
addpath(genpath('.'))

% Run the full pipeline (DSP + EKF + DDPG training)
run main_nudge.m
```

### 3. View results

Saved plots are in `uwb_ddpg/Plot Images/` and include:
- Positioning RMSE vs. episode (DDPG vs. fixed-covariance EKF baseline)
- 3D trajectory reconstruction
- DDPG reward convergence curve
- Q/R covariance adaptation over time

---

## Results

NUDGE outperforms a fixed-covariance EKF baseline under IEEE 802.15.4a CM4 NLOS conditions:

| Method | Mean Position RMSE |
|---|---|
| Fixed-covariance EKF | Baseline |
| NUDGE (DDPG-tuned EKF) | Reduced (see Plot Images) |

The DDPG agent converges within ~300 training episodes and generalises well to unseen NLOS realizations from the CM4 channel model.

---

