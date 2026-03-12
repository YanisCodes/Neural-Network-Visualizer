import pygame
import numpy as np
import random
import math
import sys
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
FPS = 60
LAYER_SIZES = [2, 6, 6, 1]

# CYBERPUNK PALETTE
BG          = (4, 5, 12)
CYAN        = (0, 255, 220)
CYAN_DIM    = (0, 100, 85)
MAGENTA     = (220, 0, 255)
MAG_DIM     = (90, 0, 110)
YELLOW      = (255, 220, 0)
RED         = (255, 50, 80)
WHITE       = (220, 230, 255)
GREY        = (50, 60, 90)
PANEL_BG    = (6, 8, 18)
BORDER      = (0, 70, 60)

# ─── NEURAL NETWORK (ton code original inchangé) ──────────────────────────────
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, s)) for s in layer_sizes[1:]]
        self.activations = [np.zeros((1, s)) for s in layer_sizes]
        self.loss_history = []
        self.accuracy_history = []
        self.epoch = 0
        self.forward(np.zeros((1, layer_sizes[0])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.activations[0] = X
        self.zs = []
        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            self.zs.append(z)
            current = self.sigmoid(z)
            self.activations[i+1] = current
        return current

    def backward(self, X, y, lr=0.1):
        m = X.shape[0]
        output = self.forward(X)
        loss = float(np.mean((output - y) ** 2))
        delta = 2 * (output - y) * self.sigmoid_deriv(self.zs[-1]) / m
        deltas = [delta]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (deltas[0] @ self.weights[i+1].T) * self.sigmoid_deriv(self.zs[i])
            deltas.insert(0, delta)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * (self.activations[i].T @ deltas[i])
            self.biases[i]  -= lr * np.sum(deltas[i], axis=0, keepdims=True)
        return loss

    def train_step(self, X, y, lr=0.05):
        loss = self.backward(X, y, lr)
        pred = self.forward(X) > 0.5
        acc  = float(np.mean(pred == y))
        self.loss_history.append(loss)
        self.accuracy_history.append(acc)
        if len(self.loss_history) > 200:
            self.loss_history.pop(0)
            self.accuracy_history.pop(0)
        self.epoch += 1
        return loss, acc


# ─── DATASETS ─────────────────────────────────────────────────────────────────
def make_dataset():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    return X, y

def make_circle_dataset(n=300):
    angles = np.random.uniform(0, 2*math.pi, n)
    radii  = np.random.uniform(0, 1, n)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    y = (radii > 0.5).astype(float).reshape(-1, 1)
    return X, y


# ─── PARTICLES ────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, src, dst, color):
        self.src   = src
        self.dst   = dst
        self.t     = 0.0
        self.speed = random.uniform(0.008, 0.022)
        self.color = color
        self.size  = random.randint(2, 4)

    def update(self):
        self.t += self.speed
        return self.t < 1.0

    def pos(self):
        x = self.src[0] + (self.dst[0] - self.src[0]) * self.t
        y = self.src[1] + (self.dst[1] - self.src[1]) * self.t
        return int(x), int(y)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def glow_circle(surf, color, pos, radius, intensity=160):
    for r in range(radius * 3, radius, -3):
        alpha = int(intensity * ((1 - (r - radius) / max(radius * 2, 1)) ** 2))
        alpha = max(0, min(255, alpha))
        s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (r+1, r+1), r)
        surf.blit(s, (pos[0]-r-1, pos[1]-r-1))

def glowing_line(surf, color, p1, p2, width=1, alpha=160):
    if p1 == p2: return
    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.line(s, (*color, alpha), p1, p2, width)
    surf.blit(s, (0, 0))

def panel_rect_draw(surf, color, rect, radius=5, border_col=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border_col:
        pygame.draw.rect(surf, border_col, rect, 1, border_radius=radius)

def make_scanlines(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 3):
        pygame.draw.line(s, (0, 0, 0, 28), (0, y), (w, y))
    return s


# ─── RENDERER (redesigné) ─────────────────────────────────────────────────────
class Renderer:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fxs, self.fsm, self.fmd, self.flg, self.fxl = fonts
        self.scanlines = make_scanlines(WIDTH, HEIGHT)
        self.tick = 0
        self.particles = []
        self.last_spawn = 0
        self.start_time = time.time()

    def spawn_particles(self, node_pos, weights):
        if self.tick - self.last_spawn < 3:
            return
        self.last_spawn = self.tick
        for li in range(len(weights)):
            W  = weights[li]
            mx = max(abs(W.max()), abs(W.min()), 1e-9)
            for ni in range(len(node_pos[li])):
                for nj in range(len(node_pos[li+1])):
                    if abs(W[ni,nj]/mx) > 0.25 and random.random() < 0.07:
                        t   = (W[ni,nj]/mx + 1) / 2
                        col = lerp(MAGENTA, CYAN, t)
                        self.particles.append(Particle(node_pos[li][ni], node_pos[li+1][nj], col))
        self.particles = [p for p in self.particles if p.update()]

    def draw_bg(self):
        self.screen.fill(BG)
        for x in range(0, WIDTH, 50):
            a = 12 if x % 200 else 28
            s = pygame.Surface((1, HEIGHT), pygame.SRCALPHA)
            s.fill((0, 180, 150, a))
            self.screen.blit(s, (x, 0))
        for y in range(0, HEIGHT, 50):
            a = 12 if y % 200 else 28
            s = pygame.Surface((WIDTH, 1), pygame.SRCALPHA)
            s.fill((0, 180, 150, a))
            self.screen.blit(s, (0, y))

    def draw_network(self, nn, rect):
        x0, y0, w, h = rect
        n_layers = len(nn.layer_sizes)
        layer_x  = [x0 + int(w * (i+1) / (n_layers+1)) for i in range(n_layers)]

        # compute node positions
        node_pos = []
        for li, (lx, size) in enumerate(zip(layer_x, nn.layer_sizes)):
            col = []
            for ni in range(size):
                ny = y0 + int(h * (ni+1) / (size+1))
                col.append((lx, ny))
            node_pos.append(col)

        # spawn + draw particles
        self.spawn_particles(node_pos, nn.weights)
        for p in self.particles:
            px, py = p.pos()
            glow_circle(self.screen, p.color, (px, py), p.size, 130)
            pygame.draw.circle(self.screen, WHITE, (px, py), max(1, p.size-1))

        # connections
        for li in range(len(nn.weights)):
            W  = nn.weights[li]
            mx = max(abs(W.max()), abs(W.min()), 1e-9)
            for ni, src in enumerate(node_pos[li]):
                for nj, dst in enumerate(node_pos[li+1]):
                    w_val = W[ni, nj]
                    t     = (w_val / mx + 1) / 2
                    col   = lerp(MAG_DIM, CYAN_DIM, t)
                    alpha = int(25 + 130 * abs(w_val / mx))
                    thick = max(1, int(2.5 * abs(w_val / mx)))
                    glowing_line(self.screen, col, src, dst, thick, alpha)

        # nodes
        pulse = (math.sin(self.tick * 0.05) + 1) / 2
        for li, positions in enumerate(node_pos):
            acts = nn.activations[li][0]
            for ni, (nx, ny) in enumerate(positions):
                act = float(acts[ni]) if ni < len(acts) else 0.0
                col = lerp(MAGENTA, CYAN, act)

                glow_r = 26 + int(10 * act * pulse)
                glow_circle(self.screen, col, (nx, ny), glow_r, int(50 + 55*act))

                body = lerp((12, 12, 28), col, 0.25 + 0.7*act)
                pygame.draw.circle(self.screen, body, (nx, ny), 20)
                pygame.draw.circle(self.screen, lerp(GREY, col, act), (nx, ny), 20, 2)
                pygame.draw.circle(self.screen, lerp((20,20,40), WHITE, act), (nx, ny), 6)

                txt = self.fxs.render(f"{act:.2f}", True, lerp(GREY, WHITE, act))
                self.screen.blit(txt, (nx - txt.get_width()//2, ny + 25))

        # layer labels
        labels = ["INPUT"] + [f"H.{i+1}" for i in range(n_layers-2)] + ["OUTPUT"]
        for li, (lx, label) in enumerate(zip(layer_x, labels)):
            txt = self.fxs.render(label, True, CYAN_DIM)
            self.screen.blit(txt, (lx - txt.get_width()//2, y0 - 28))

    def draw_graph(self, data, rect, color, label, y_range=(0, 1)):
        x0, y0, w, h = rect
        panel_rect_draw(self.screen, PANEL_BG, rect, border_col=BORDER)
        pygame.draw.rect(self.screen, color, (x0, y0, w, 2), border_radius=4)

        if len(data) < 2:
            return
        lo, hi = y_range
        pts = []
        for i, v in enumerate(data):
            px = x0 + int(w * i / (len(data)-1))
            py = y0 + h - int(h * (v - lo) / max(hi - lo, 1e-9))
            py = max(y0+4, min(y0+h-2, py))
            pts.append((px, py))

        fill_pts = [(x0, y0+h)] + pts + [(x0+w, y0+h)]
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        local = [(p[0]-x0, p[1]-y0) for p in fill_pts]
        pygame.draw.polygon(s, (*color, 22), local)
        self.screen.blit(s, (x0, y0))
        pygame.draw.lines(self.screen, color, False, pts, 2)

        lbl = self.fxs.render(label, True, color)
        self.screen.blit(lbl, (x0+8, y0+8))
        val = self.fmd.render(f"{data[-1]:.4f}", True, color)
        self.screen.blit(val, (x0+w - val.get_width() - 8, y0+6))

    def draw_panel(self, nn, training, speed, rect):
        x0, y0, w, h = rect
        panel_rect_draw(self.screen, PANEL_BG, rect, border_col=BORDER)
        pygame.draw.rect(self.screen, CYAN, (x0, y0, w, 3), border_radius=4)

        # title
        t = self.flg.render("SYS.STATUS", True, CYAN)
        self.screen.blit(t, (x0+16, y0+14))
        pygame.draw.line(self.screen, BORDER, (x0+14, y0+46), (x0+w-14, y0+46))

        status_col = CYAN if training else RED
        rows = [
            ("EPOCH",    f"{nn.epoch:,}",                                             WHITE),
            ("LOSS",     f"{nn.loss_history[-1]:.6f}" if nn.loss_history else "—",    RED),
            ("ACCURACY", f"{nn.accuracy_history[-1]*100:.2f}%" if nn.accuracy_history else "—", CYAN),
            ("SPEED",    f"{speed}x",                                                  YELLOW),
            ("PARTICLES",f"{len(self.particles)}",                                    MAGENTA),
            ("STATUS",   "● TRAINING" if training else "● PAUSED",                   status_col),
        ]
        for i, (k, v, col) in enumerate(rows):
            ry = y0 + 56 + i * 46
            self.screen.blit(self.fxs.render(k, True, (55, 75, 95)), (x0+16, ry))
            vt = self.fmd.render(v, True, col)
            self.screen.blit(vt, (x0+w - vt.get_width() - 16, ry+2))
            pygame.draw.line(self.screen, (14, 18, 32), (x0+14, ry+38), (x0+w-14, ry+38))

        # controls
        ctrl_y = y0 + h - 160
        pygame.draw.line(self.screen, BORDER, (x0+14, ctrl_y-8), (x0+w-14, ctrl_y-8))
        self.screen.blit(self.fxs.render("CONTROLS", True, (38, 58, 78)), (x0+16, ctrl_y))
        controls = [
            ("SPACE", "pause / resume"),
            ("R",     "reset network"),
            ("↑ ↓",   "training speed"),
            ("D",     "switch dataset"),
            ("ESC",   "quit"),
        ]
        for i, (key, desc) in enumerate(controls):
            ky = ctrl_y + 18 + i * 26
            self.screen.blit(self.fxs.render(key, True, YELLOW), (x0+16, ky))
            self.screen.blit(self.fxs.render(desc, True, (55, 72, 95)), (x0+80, ky))

    def draw_header(self, ds_name):
        # glow
        glow = self.fxl.render("NEURAL VIZ", True, (0, 45, 38))
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            self.screen.blit(glow, (16+dx, 12+dy))
        self.screen.blit(self.fxl.render("NEURAL VIZ", True, CYAN), (16, 12))

        sub = self.fxs.render("// REAL-TIME BACKPROPAGATION VISUALIZER  //  github.com/YanisCodes", True, (0, 72, 60))
        self.screen.blit(sub, (16, 54))

        elapsed = int(time.time() - self.start_time)
        up = self.fxs.render(f"UPTIME {elapsed:05d}s", True, (0, 65, 55))
        self.screen.blit(up, (WIDTH - up.get_width() - 14, 14))
        ds_txt = self.fxs.render(f"DATASET: {ds_name}", True, (0, 65, 55))
        self.screen.blit(ds_txt, (WIDTH - ds_txt.get_width() - 14, 30))

    def finalize(self):
        self.screen.blit(self.scanlines, (0, 0))
        self.tick += 1


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("NEURAL VIZ  //  YanisCodes")
    clock  = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Courier New", 11, bold=True),   # fxs
        pygame.font.SysFont("Courier New", 13, bold=True),   # fsm
        pygame.font.SysFont("Courier New", 17, bold=True),   # fmd
        pygame.font.SysFont("Courier New", 22, bold=True),   # flg
        pygame.font.SysFont("Courier New", 36, bold=True),   # fxl
    )

    renderer = Renderer(screen, fonts)

    datasets   = [make_dataset, lambda: make_circle_dataset(300)]
    ds_names   = ["XOR", "CIRCLES"]
    ds_idx     = 0
    X, y       = datasets[ds_idx]()

    nn       = NeuralNetwork(LAYER_SIZES)
    training = True
    speed    = 5

    PANEL_W     = 260
    panel_rect  = (WIDTH - PANEL_W - 16, 76, PANEL_W, HEIGHT - 92)
    net_rect    = (16, 90, WIDTH - PANEL_W - 48, HEIGHT - 210)
    loss_rect   = (16, HEIGHT - 108, (WIDTH - PANEL_W - 48)//2 - 8, 92)
    acc_rect    = ((WIDTH - PANEL_W - 48)//2 + 24, HEIGHT - 108, (WIDTH - PANEL_W - 48)//2 - 8, 92)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    training = not training
                elif event.key == pygame.K_r:
                    nn = NeuralNetwork(LAYER_SIZES)
                    renderer.particles = []
                elif event.key == pygame.K_UP:
                    speed = min(50, speed + 5)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 5)
                elif event.key == pygame.K_d:
                    ds_idx = (ds_idx + 1) % len(datasets)
                    X, y   = datasets[ds_idx]()
                    nn     = NeuralNetwork(LAYER_SIZES)
                    renderer.particles = []

        if training:
            for _ in range(speed):
                idx = np.random.randint(0, len(X))
                nn.train_step(X[idx:idx+1], y[idx:idx+1], lr=0.1)
            idx = np.random.randint(0, len(X))
            nn.forward(X[idx:idx+1])

        # ── DRAW ──
        renderer.draw_bg()
        renderer.draw_header(ds_names[ds_idx])
        renderer.draw_network(nn, net_rect)
        renderer.draw_graph(nn.loss_history, loss_rect, RED, "LOSS", (0, max(nn.loss_history or [1])))
        renderer.draw_graph(nn.accuracy_history, acc_rect, CYAN, "ACCURACY", (0, 1))
        renderer.draw_panel(nn, training, speed, panel_rect)
        renderer.finalize()

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()