# NUDGE
NUDGE — Navigation Using DDPG-Guided Estimation

NUDGE is a simulation-first framework for autonomous drone positioning in environments where GPS is unavailable — underground mine shafts, deep open-pit excavations, and dense-canopy post-mining sites.
The system fuses Ultra-Wideband (UWB) radio ranging from four fixed anchors with a three-stage DSP pipeline and a DDPG reinforcement learning agent that adaptively tunes Kalman filter parameters in real time, responding to the harsh NLOS multipath conditions characteristic of mine environments.
This repository contains the complete Month 1 MATLAB simulation suite, validated against the IEEE 802.15.4a CM4 Industrial NLOS channel model.
