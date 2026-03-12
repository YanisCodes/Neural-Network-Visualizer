# 🧠 Neural Network Visualizer

> Real-time visualization of a feedforward neural network training from scratch — no ML frameworks, just raw NumPy and math.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-only-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)
![Built in one night](https://img.shields.io/badge/built%20in-one%20night-ff69b4?style=flat-square)

---

## 🌙 Origin Story

Built in one night. I wanted to understand what actually happens inside a neural network — so I built one from scratch.

The idea was simple: stop treating backpropagation as a black box — build it from scratch and *watch it learn* in real time. One night was enough.

---

## What is this?

A from-scratch implementation of a multi-layer perceptron (MLP) with **live visual feedback** as it learns.

Every weight update, every activation, every gradient — rendered in real time.

Built with **zero ML libraries**. Just NumPy for matrix ops and Pygame for rendering.

---

## Features

- 🔴 **Live weight visualization** — connection colors shift from red (negative) to teal (positive) as weights update
- ⚡ **Real-time backpropagation** — watch the network correct itself step by step
- 📈 **Live loss & accuracy curves** — plotted as the network trains
- 🔁 **Multiple datasets** — XOR problem and circular classification
- 🎛️ **Interactive controls** — pause, reset, change speed on the fly

---

## Architecture

```
Input Layer (2) → Hidden Layer (6) → Hidden Layer (6) → Output Layer (1)
```

- **Activation:** Sigmoid σ(x) = 1 / (1 + e⁻ˣ)
- **Loss:** Mean Squared Error  L = (1/n) Σ(ŷ − y)²
- **Optimizer:** Stochastic Gradient Descent
- **Weight init:** He initialization → W ~ N(0, √(2/nᵢₙ))

---

## Getting Started

```bash
git clone https://github.com/YanisCodes/neural-viz
cd neural-viz
pip install -r requirements.txt
python main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume training |
| `R` | Reset network |
| `↑ / ↓` | Increase / decrease training speed |
| `D` | Switch dataset |
| `ESC` | Quit |

---

## Why from scratch?

Using PyTorch or TensorFlow would've taken 10 lines.
The goal here was to understand and *show* what actually happens inside the black box — every dot product, every partial derivative, every weight nudge — live on screen.

One night. One file. Pure math.

---

## Requirements

```
pygame>=2.1.0
numpy>=1.23.0
```

---

*Made with curiosity & zero sleep — [@YanisCodes](https://github.com/YanisCodes)*
