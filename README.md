# 🧠 Neural Network Visualizer

> **Real-time training visualizer for a feedforward neural network — built from scratch with zero ML frameworks.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.x-green?style=flat-square)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/NumPy-only-orange?style=flat-square)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YanisCodes/Neural-Network-Visualizer?style=flat-square)](https://github.com/YanisCodes/Neural-Network-Visualizer)
[![Built in one night](https://img.shields.io/badge/built%20in-one%20night-ff69b4?style=flat-square)](https://github.com/YanisCodes/Neural-Network-Visualizer)

---

## ✨ Features

- 🔴 **Live weight visualization** — connection colors shift from red (negative) to teal (positive) as weights update in real time
- ⚡ **Real-time backpropagation** — watch the network correct itself step by step, particle by particle
- 📈 **Live loss & accuracy curves** — plotted live as the network trains on screen
- 🔁 **Multiple datasets** — switch between XOR and circular classification problems
- 🎛️ **Interactive controls** — pause, reset, toggle datasets, change training speed on the fly
- 🌊 **Glow & particle effects** — cyberpunk-styled UI with animated data flow particles
- 🧪 **No ML frameworks** — pure NumPy for math, Pygame for rendering. See every dot product and gradient update.

---

## 📸 Demo

> *Screenshot coming soon — run it yourself to see the live visualization in action!*

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YanisCodes/Neural-Network-Visualizer.git
cd Neural-Network-Visualizer

# Install dependencies
pip install -r requirements.txt

# Run the visualizer
python main.py
```

> **Note:** This requires a display (X11/Wayland). On headless servers, use `xvfb-run python main.py` or a VNC session.

---

## 🎮 Usage

Launch `python main.py` to open a fullscreen visualization. You'll see:

- The **network architecture** rendered as connected nodes (2 inputs → 6 hidden → 6 hidden → 1 output)
- **Animating particles** flowing along connections, showing forward propagation
- **Color-coded weights** — teal = positive, magenta = negative
- **Live graphs** tracking loss and accuracy over time
- A **status panel** with epoch count, loss, accuracy, and speed

### Controls

| Key | Action                |
|-----|-----------------------|
| `SPACE` | Pause / Resume training |
| `R`     | Reset network weights |
| `↑ / ↓` | Increase / decrease speed |
| `D`     | Switch dataset (XOR ↔ Circles) |
| `ESC`   | Quit |

---

## ⚙️ Architecture

```
Input Layer (2) → Hidden Layer (6) → Hidden Layer (6) → Output Layer (1)
```

| Component | Detail |
|-----------|--------|
| **Activation** | Sigmoid σ(x) = 1 / (1 + e⁻ˣ) |
| **Loss** | Mean Squared Error L = (1/n) Σ(ŷ − y)² |
| **Optimizer** | Stochastic Gradient Descent (manual) |
| **Weight init** | He initialization W ~ N(0, √(2/nᵢₙ)) |
| **Datasets** | XOR (4 samples) & Circle classification (300 samples) |

---

## 🛠 Tech Stack

- **Python 3.10+** — core language
- **NumPy** — matrix operations, forward/backward propagation
- **Pygame 2.x** — real-time 2D rendering, input handling
- **Matplotlib** *(optional)* — used only via Pygame's font rendering; all graph drawing is custom

No PyTorch. No TensorFlow. No scikit-learn. Just raw math.

---

## 📁 Project Structure

```
Neural-Network-Visualizer/
├── main.py             # Single-file application — network, renderer, UI, and main loop
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
├── .gitignore          # Standard Python ignores
└── README.md           # This file
```

> Everything is in **one file** — `main.py` contains the neural network class, dataset generators, particle system, renderer, and game loop.

---

## 🧠 Why from scratch?

Using PyTorch or TensorFlow would've taken 10 lines. The goal here was to understand and **show** what actually happens inside the black box — every dot product, every partial derivative, every weight nudge — live on screen.

One night. One file. Pure math.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🌟 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YanisCodes/Neural-Network-Visualizer/issues).

---

*Made with curiosity & zero sleep — [@YanisCodes](https://github.com/YanisCodes)*