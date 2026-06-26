# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning project for a two-wheeled self-balancing robot. A custom Gymnasium environment simulates the physics (cart-pole variant with motor lag), PPO from Stable-Baselines3 trains a policy, and the final model is exported to TensorFlow Lite for deployment on embedded hardware (Pixhawk).

## Commands

```bash
# Train a model (300k timesteps, saves to balancing_robot_model.zip)
python train.py

# Multi-stage training with checkpoints at 300k and 500k timesteps
python train_compare.py

# Test a trained model's recovery from large initial tilt angles
python test_model.py

# Side-by-side visual comparison of two models (300k vs 500k)
python test_side_by_side.py

# Export trained model to TFLite (PyTorch → ONNX → TF → TFLite with float16 quantization)
python tflite_exporter.py
```

### Environment setup

```bash
pip install gymnasium stable-baselines3 torch onnx onnx-tf tensorflow matplotlib numpy
```

## Architecture

**Environment** (`balancing_robot_env.py`): `BalancingRobotEnv` extends `gymnasium.Env`. Observation space is 4D `[x, x_dot, theta, theta_dot]`. Action space is 1D continuous `[-1, 1]` representing normalized motor force. Physics simulates a cart-pole with a **motor response lag** (first-order IIR filter via `motor_alpha=0.3`) and **reward shaping** that penalizes tilt, displacement, velocity, and control effort quadratically. Termination at `|x| > 2.4` or `|theta| > 20°`. Rendering uses Matplotlib with optional subplot support (`ax` parameter) for side-by-side comparison.

**Training** (`train.py`, `train_compare.py`): Uses Stable-Baselines3 PPO with `MlpPolicy`. Default hyperparams: `learning_rate=0.0003`, `n_steps=2048`, `batch_size=64`. `train.py` does a single 300k-step run. `train_compare.py` does incremental training (300k then +200k) saving intermediate checkpoints — this allows comparing how performance improves with more training.

**Export pipeline** (`tflite_exporter.py`): `OnnxablePolicy` wraps the trained PPO policy to extract only the deterministic actor (stripping critic and stochastic distribution). The pipeline is: PyTorch `.zip` → ONNX → TensorFlow SavedModel → TFLite (float16 quantized). The resulting `.tflite` file targets Pixhawk C++ deployment.

**Testing** (`test_model.py`, `test_side_by_side.py`): Both directly mutate `env.unwrapped.state` to set custom initial conditions (extreme tilt angles) for targeted recovery tests. `test_side_by_side.py` runs two environments simultaneously in a single Matplotlib figure with two subplots.

## Key details

- `env.unwrapped.state` can be mutated after `reset()` to inject custom starting states — used by test scripts for recovery-from-tilt benchmarks.
- The environment's `render()` accepts an optional `ax` Matplotlib axis and `title` string, enabling multi-environment side-by-side rendering.
- Models save/load from the repo root as `.zip` files (SB3 format) — these are gitignored along with TensorBoard logs and `rl_env/`.
- TensorBoard logs go to `./ppo_balancing_tensorboard/` for visualizing training curves.
