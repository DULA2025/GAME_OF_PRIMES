import cupy as cp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
import colorsys
import time

# Verify GPU availability
try:
    cp.cuda.Device(0).use()
    print("Using GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))
except Exception as e:
    print("GPU not available, falling back to CPU:", str(e))
    import numpy as cp

# Constants
ON = 255
OFF = 0
vals = [ON, OFF]

# Prime number check (CPU scalar)
def is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Simplified E8 mapping (CPU scalar)
def map_to_e8(n):
    n = int(n)
    idx = n % 8
    k = (n // 8) + 1
    vector = np.zeros(8)
    vector[idx] = 36 * k * (idx % 6 + 1)
    return vector

# Grid initialization functions
def randomGrid(N):
    """Returns a grid of NxN random values with prime bias"""
    grid = cp.zeros((N, N), dtype=cp.uint8)
    numbers = cp.arange(1, N * N + 1).reshape(N, N)
    grid_cpu = np.zeros((N, N), dtype=np.uint8)
    target_alive = int(N * N * 0.2)
    for i in range(N):
        for j in range(N):
            n = int(numbers[i, j].get())
            if is_prime(n) or (is_prime(n + 2) or (n - 2 > 0 and is_prime(n - 2))):
                grid_cpu[i, j] = ON
            elif cp.random.random() < 0.2 - cp.sum(grid) / (N * N):
                grid_cpu[i, j] = ON
            if np.sum(grid_cpu) >= target_alive:
                break
        if np.sum(grid_cpu) >= target_alive:
            break
    return cp.asarray(grid_cpu)

def addGlider(i, j, grid):
    """Adds a glider at (i, j)"""
    glider = cp.array([[OFF, OFF, ON],
                       [ON, OFF, ON],
                       [OFF, ON, ON]], dtype=cp.uint8)
    grid[i:i+3, j:j+3] = glider

def addGosperGliderGun(i, j, grid):
    """Adds a Gosper Glider Gun at (i, j)"""
    gun = cp.zeros((11, 38), dtype=cp.uint8)
    gun[5, 1:3] = ON
    gun[6, 1:3] = ON
    gun[3, 13:15] = ON
    gun[4, [12, 16]] = ON
    gun[5, [11, 17]] = ON
    gun[6, [11, 15, 17, 18]] = ON
    gun[7, [11, 17]] = ON
    gun[8, [12, 16]] = ON
    gun[9, 13:15] = ON
    gun[1, 25] = ON
    gun[2, [23, 25]] = ON
    gun[3:6, 21:23] = ON
    gun[6, [23, 25]] = ON
    gun[7, 25] = ON
    gun[3:5, 35:37] = ON
    grid[i:i+11, j:j+38] = gun

# GPU-accelerated neighbor counting
def count_neighbors_parallel(grid, N):
    neighbors = cp.zeros_like(grid, dtype=cp.int32)
    offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
    for di, dj in offsets:
        shifted = cp.roll(grid, (di, dj), axis=(0, 1))
        neighbors += shifted // 255  # Toroidal wrapping handled by cp.roll
    return neighbors

# GPU-accelerated update
def update(grid, N, numbers):
    new_grid = grid.copy()
    neighbors = count_neighbors_parallel(grid, N)
    
    # Conway's rules
    alive = grid == ON
    die = alive & ((neighbors < 2) | (neighbors > 3))
    survive = alive & (~die)
    dead = grid == OFF
    birth = dead & (neighbors == 3)
    
    # Vectorized prime/twin prime boost
    birth_indices = cp.where(dead & (neighbors >= 2) & (neighbors <= 3))  # Relaxed condition
    birth_nums = numbers[birth_indices].get()
    for idx, n in enumerate(birth_nums):
        n = int(n)
        if (is_prime(n) or (n + 2 <= N * N and is_prime(n + 2))) and n <= N * N:
            i, j = birth_indices[0][idx], birth_indices[1][idx]
            birth[i, j] = True
    
    new_grid[die] = OFF
    new_grid[survive] = ON
    new_grid[birth] = ON
    
    return new_grid

# Extract alive positions
def get_alive_positions(grid):
    alive_indices = cp.where(grid == ON)
    return cp.stack(alive_indices, axis=1)

# Particle effect system
class Particle:
    def __init__(self, x, y, color, lifetime=1.0, size=0.05, velocity=(0, 0), type='birth'):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.velocity = velocity
        self.type = type

particle_effects = []
cell_colors = {}
pulse_cells = []
alive_history = []

def create_particles(i, j, type):
    x = (j - N/2) * cell_size
    y = (N/2 - i) * cell_size
    if type == 'birth':
        color = cell_colors.get((i, j), (0.5, 0.8, 0.2))
        for _ in range(5):
            vx = random.uniform(-0.02, 0.02)
            vy = random.uniform(-0.02, 0.02)
            size = random.uniform(0.01, 0.03)
            lifetime = random.uniform(0.5, 1.0)
            particle_effects.append(Particle(x, y, color, lifetime, size, (vx, vy), type))
    else:
        color = (0.7, 0.3, 0.7)
        for _ in range(3):
            vx = random.uniform(-0.01, 0.01)
            vy = random.uniform(-0.01, 0.01)
            size = random.uniform(0.01, 0.025)
            lifetime = random.uniform(0.3, 0.7)
            particle_effects.append(Particle(x, y, color, lifetime, size, (vx, vy), type))

def update_particles(dt):
    global particle_effects
    new_particles = []
    for p in particle_effects:
        p.lifetime -= dt
        if p.lifetime > 0:
            p.x += p.velocity[0]
            p.y += p.velocity[1]
            p.size *= 0.98 if p.type == 'birth' else 0.97
            new_particles.append(p)
    particle_effects = new_particles

# Draw a single cell
cell_size = 0.15
def draw_cell(i, j):
    # Adjusted coordinates to match typical top-left origin
    x = (j - N/2) * cell_size
    y = (N/2 - i) * cell_size  # Flipped i to match standard orientation
    r, g, b = cell_colors.get((i, j), (0.1, 0.7, 0.9))
    half_size = cell_size / 2
    
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    glVertex2f(x - half_size, y - half_size)
    glVertex2f(x + half_size, y - half_size)
    glVertex2f(x + half_size, y + half_size)
    glVertex2f(x - half_size, y + half_size)
    glEnd()

def draw_particles():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    for p in particle_effects:
        alpha = p.lifetime / p.max_lifetime
        glColor4f(p.color[0], p.color[1], p.color[2], alpha)
        glBegin(GL_QUADS)
        glVertex2f(p.x - p.size, p.y - p.size)
        glVertex2f(p.x + p.size, p.y - p.size)
        glVertex2f(p.x + p.size, p.y + p.size)
        glVertex2f(p.x - p.size, p.y + p.size)
        glEnd()
    glDisable(GL_BLEND)

def draw_grid(alive_positions_gpu):
    alive_positions = alive_positions_gpu.get()
    for i, j in alive_positions:
        draw_cell(i, j)

# OpenGL initialization
def init_gl(N):
    glClearColor(0.1, 0.1, 0.2, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = N / N
    glOrtho(-10 * aspect_ratio, 10 * aspect_ratio, -10, 10, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Plotting
fig, ax = plt.subplots()
def update_plot(frame):
    ax.clear()
    ax.plot(alive_history, label='Alive Cells', color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Game of Life with Primes')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

# Main function
def main():
    global N, grid, numbers
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life with prime dynamics.")
    parser.add_argument('--grid-size', dest='N', required=False, type=int, default=100)
    parser.add_argument('--interval', dest='interval', required=False, type=int, default=50)
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--gosper', action='store_true', required=False)
    args = parser.parse_args()

    N = args.N if args.N and args.N > 8 else 100
    update_interval = args.interval / 1000  # Convert ms to seconds

    # Initialize grid
    grid = cp.zeros((N, N), dtype=cp.uint8)
    numbers = cp.arange(1, N * N + 1).reshape(N, N)
    if args.glider:
        addGlider(1, 1, grid)
    elif args.gosper:
        addGosperGliderGun(10, 10, grid)
    else:
        grid = randomGrid(N)

    # Pygame/OpenGL setup
    pygame.init()
    display = (1200, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("GPU-Accelerated Game of Life with Primes")
    init_gl(N)

    # Animation setup
    ani = FuncAnimation(fig, update_plot, interval=5000, cache_frame_data=False)
    plt.ion()
    plt.show(block=False)

    generation = 0
    last_time = time.time()
    last_evolution = 0
    running = True
    paused = False
    zoom = 1.0
    pan_x, pan_y = 0, 0

    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_r:
                    grid = randomGrid(N)
                    alive_history.clear()
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom *= 1.1
                elif event.button == 5:
                    zoom /= 1.1
            elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[1]:
                pan_x += event.rel[0] * 0.01
                pan_y -= event.rel[1] * 0.01

        if not paused and current_time - last_evolution > update_interval:
            old_alive = get_alive_positions(grid)
            grid = update(grid, N, numbers)
            new_alive = get_alive_positions(grid)
            old_set = set(zip(old_alive[:, 0].get(), old_alive[:, 1].get()))
            new_set = set(zip(new_alive[:, 0].get(), new_alive[:, 1].get()))
            births = new_set - old_set
            deaths = old_set - new_set
            for i, j in births:
                n = int(numbers[i, j].get())
                if is_prime(n) and is_prime(n + 2):
                    cell_colors[(i, j)] = (0.9, 0.1, 0.1)
                elif is_prime(n):
                    cell_colors[(i, j)] = (0.1, 0.7, 0.9)
                else:
                    hue = (n % 360) / 360.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                    cell_colors[(i, j)] = (r, g, b)
                create_particles(i, j, 'birth')
            for i, j in deaths:
                create_particles(i, j, 'death')
            alive_history.append(int(cp.sum(grid == ON)))
            generation += 1
            last_evolution = current_time
            print(f"Generation {generation}: {alive_history[-1]} alive cells")

        update_particles(dt)
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(pan_x, pan_y, 0)
        glScalef(zoom, zoom, 1.0)
        draw_grid(get_alive_positions(grid))
        draw_particles()
        pygame.display.flip()
        pygame.time.wait(5)

    pygame.quit()
    plt.close()

if __name__ == '__main__':
    main()
