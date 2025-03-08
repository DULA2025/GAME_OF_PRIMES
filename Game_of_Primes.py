import cupy as cp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Verify GPU availability
try:
    cp.cuda.Device(0).use()
    print("Using GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))
except Exception as e:
    print("GPU not available, falling back to CPU:", str(e))
    import numpy as cp

# Basic primality test (CPU scalar)
def is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Congruence class (CPU scalar)
def congruence_class(n):
    n = int(n)
    mod6 = n % 6
    return 1 if mod6 == 1 else -1 if mod6 == 5 else 0

# Simplified E8 mapping (CPU scalar)
def map_to_e8(n):
    n = int(n)
    idx = n % 8
    k = (n // 8) + 1
    vector = np.zeros(8)
    vector[idx] = 36 * k * (idx % 6 + 1)
    return vector

# 2D Grid setup (on GPU)
width = 100
height = 100
cell_size = 0.15
grid = cp.zeros((height, width), dtype=cp.int32)
numbers = cp.arange(1, width * height + 1).reshape(height, width)
particle_effects = []
cell_colors = {}
pulse_cells = []
alive_history = []

# Initialize 2D grid
target_alive = int(width * height * 0.0474)
grid_cpu = np.zeros((height, width), dtype=np.int32)
for i in range(height):
    for j in range(width):
        n = int(numbers[i, j].get())
        if is_prime(n) or (is_prime(n + 2) or (n - 2 > 0 and is_prime(n - 2))):
            grid_cpu[i, j] = 1
            if is_prime(n) and is_prime(n + 2):
                cell_colors[(i, j)] = (0.9, 0.1, 0.1)
            elif is_prime(n):
                cell_colors[(i, j)] = (0.1, 0.7, 0.9)
        elif np.random.random() < 0.0474 - np.sum(grid_cpu) / (width * height) and congruence_class(n) != 0:
            grid_cpu[i, j] = 1
        if np.sum(grid_cpu) >= target_alive:
            break
    if np.sum(grid_cpu) >= target_alive:
        break
grid = cp.asarray(grid_cpu)

# GPU-optimized 2D neighbor counting
def count_neighbors_parallel(grid, height, width):
    neighbors = cp.zeros_like(grid, dtype=cp.int32)
    offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
    
    for di, dj in offsets:
        shifted = cp.roll(grid, (di, dj), axis=(0, 1))
        if di == -1:
            shifted[:1, :] = 0
        elif di == 1:
            shifted[-1:, :] = 0
        if dj == -1:
            shifted[:, :1] = 0
        elif dj == 1:
            shifted[:, -1:] = 0
        neighbors += shifted
    
    return neighbors

# GPU-accelerated 2D evolution
def evolve_parallel(grid, numbers, height, width, rand_vals):
    new_grid = grid.copy()
    neighbors = count_neighbors_parallel(grid, height, width)
    
    mod_class = cp.where(numbers % 6 == 1, 1, cp.where(numbers % 6 == 5, -1, 0))
    is_twin = cp.array([is_prime(n) and is_prime(n + 2) if n + 2 <= width * height else False 
                        for n in numbers.get().flatten()], dtype=cp.bool_).reshape(height, width)
    v_n = cp.array([map_to_e8(n) for n in numbers.get().flatten()]).reshape(height, width, 8)
    v_n2 = cp.array([map_to_e8(n + 2) if n + 2 <= width * height else map_to_e8(n) 
                     for n in numbers.get().flatten()]).reshape(height, width, 8)
    dist = cp.linalg.norm(v_n - v_n2, axis=2)
    twin_bonus = cp.where((is_twin) | (dist < 200), 1, 0)
    
    alive = grid == 1
    die = alive & ((neighbors < 1) | (neighbors > 2))
    survive = alive & (~die)
    decay = survive & (rand_vals < 0.05)
    new_grid[die | decay] = 0
    new_grid[survive & ~decay] = 1
    
    dead = grid == 0
    birth_chance = cp.where(mod_class != 0, 0.25, 0.20)
    twin_boost = cp.where(twin_bonus & (neighbors >= 0), 0.50, 0)
    birth = dead & cp.isin(neighbors, cp.array([1, 2, 3, 4, 5])) & (rand_vals < (birth_chance + twin_boost))
    new_grid[birth] = 1
    
    return new_grid

# Extract alive cell positions on GPU
def get_alive_positions(grid):
    alive_indices = cp.where(grid == 1)
    return cp.stack(alive_indices, axis=1)  # [N_alive, 2] array of (i, j)

# Wrapper to handle evolution and side effects
def evolve(grid, numbers, height, width):
    global cell_colors, pulse_cells, alive_history
    old_grid = grid.copy()
    rand_vals = cp.random.random((height, width))
    new_grid = evolve_parallel(grid, numbers, height, width, rand_vals)
    
    old_alive = cp.where(old_grid == 1)
    new_alive = cp.where(new_grid == 1)
    old_set = set(zip(old_alive[0].get(), old_alive[1].get()))
    new_set = set(zip(new_alive[0].get(), new_alive[1].get()))
    
    births = list(new_set - old_set)
    deaths = list(old_set - new_set)
    pulse = []
    
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
    
    alive_cells = old_set & new_set
    for i, j in alive_cells:
        n = int(numbers[i, j].get())
        is_twin = is_prime(n) and is_prime(n + 2)
        v_n = map_to_e8(n)
        v_n2 = map_to_e8(n + 2) if n + 2 <= width * height else v_n
        dist = np.linalg.norm(v_n - v_n2)
        twin_bonus = 1 if (is_twin or dist < 200) else 0
        if twin_bonus and random.random() < 0.3:
            pulse.append(((i, j), random.uniform(0.8, 1.2), random.uniform(0.05, 0.15)))
    
    pulse_cells.extend(pulse)
    alive_history.append(int(cp.sum(new_grid)))
    
    return new_grid

# Particle effect system (2D)
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

def create_particles(i, j, type):
    x = (j - width/2) * cell_size
    y = (height/2 - i) * cell_size
    if type == 'birth':
        color = cell_colors.get((i, j), (0.5, 0.8, 0.2))
        for _ in range(5):  # Reduced for performance
            vx = random.uniform(-0.02, 0.02)
            vy = random.uniform(-0.02, 0.02)
            size = random.uniform(0.01, 0.03)
            lifetime = random.uniform(0.5, 1.0)
            particle_effects.append(Particle(x, y, color, lifetime, size, (vx, vy), type))
    else:
        color = (0.7, 0.3, 0.7)
        for _ in range(3):  # Reduced for performance
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
    
    global pulse_cells
    new_pulses = []
    for (pos, scale, rate) in pulse_cells:
        new_scale = scale - rate * dt
        if new_scale > 0.8:
            new_pulses.append((pos, new_scale, rate))
    pulse_cells = new_pulses

# Draw a single cell (2D)
def draw_cell(i, j, size_modifier=1.0):
    x = (j - width/2) * cell_size
    y = (height/2 - i) * cell_size
    if (i, j) in cell_colors:
        r, g, b = cell_colors[(i, j)]
    else:
        hue = ((i * j) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    
    for (pi, pj), scale, _ in pulse_cells:
        if pi == i and pj == j:
            size_modifier = scale
            break
    
    half_size = cell_size / 2 * size_modifier
    
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    glVertex2f(x - half_size, y - half_size)
    glVertex2f(x + half_size, y - half_size)
    glVertex2f(x + half_size, y + half_size)
    glVertex2f(x - half_size, y + half_size)
    glEnd()
    
    glColor3f(r * 0.8, g * 0.8, b * 0.8)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x - half_size, y - half_size)
    glVertex2f(x + half_size, y - half_size)
    glVertex2f(x + half_size, y + half_size)
    glVertex2f(x - half_size, y + half_size)
    glEnd()

# Draw all particles (2D)
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

# Draw grid with alive positions
def draw_grid(alive_positions_gpu):
    alive_positions = alive_positions_gpu.get()  # [N_alive, 2]
    for i, j in alive_positions:
        draw_cell(i, j)

# OpenGL initialization for 2D
def init_gl():
    glClearColor(0.1, 0.1, 0.2, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = width / height
    if aspect_ratio > 1:
        glOrtho(-10 * aspect_ratio, 10 * aspect_ratio, -10, 10, -1, 1)
    else:
        glOrtho(-10, 10, -10 / aspect_ratio, 10 / aspect_ratio, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Plotting function
fig, ax = plt.subplots()
def update_plot(frame):
    ax.clear()
    ax.plot(alive_history, label='Alive Cells', color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Alive Cells')
    ax.set_title('2D Game of Primes: Alive Cells Over Time')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

# Main game loop
def main():
    global grid
    pygame.init()
    display = (1200, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Fast 2D Game of Primes - GPU Accelerated (2x Speed)")
    
    init_gl()
    
    stable_count = 0
    last_alive = 0
    restart_count = 0
    max_restarts = 3
    running_avg = []
    window_size = 10
    
    paused = False
    generation = 0
    last_time = time.time()
    evolution_speed = 0.1  # 2x faster (was 0.2)
    last_evolution = 0
    
    zoom = 1.0
    pan_x, pan_y = 0, 0
    
    ani = FuncAnimation(fig, update_plot, interval=5000, cache_frame_data=False)
    plt.ion()
    plt.show(block=False)
    
    running = True
    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    grid = cp.where(cp.random.random((height, width)) < 0.10, 1, 0).astype(cp.int32)
                    alive_history.clear()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    evolution_speed = max(0.05, evolution_speed - 0.05)
                elif event.key == pygame.K_MINUS:
                    evolution_speed += 0.05
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom *= 1.1
                elif event.button == 5:
                    zoom /= 1.1
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[1]:
                pan_x += event.rel[0] * 0.01
                pan_y -= event.rel[1] * 0.01
        
        if not paused and current_time - last_evolution > evolution_speed:
            grid = evolve(grid, numbers, height, width)
            generation += 1
            last_evolution = current_time
            
            current_alive = int(cp.sum(grid))
            print(f"Generation {generation}: {current_alive} alive cells")
            
            if len(alive_history) >= window_size:
                avg = np.mean(alive_history[-window_size:])
                running_avg.append(avg)
                print(f"Running average (last {window_size} gens): {avg:.2f}")
            
            if current_alive == last_alive:
                stable_count += 1
                if stable_count > 10 and restart_count < max_restarts:
                    print("Simulation stabilized, restarting with noise.")
                    grid = cp.where(cp.random.random((height, width)) < 0.10, 1, 0).astype(cp.int32)
                    alive_history.clear()
                    stable_count = 0
                    restart_count += 1
            else:
                stable_count = 0
            last_alive = current_alive
        
        update_particles(dt)
        
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(pan_x, pan_y, 0)
        glScalef(zoom, zoom, 1.0)
        
        alive_positions = get_alive_positions(grid)
        draw_grid(alive_positions)
        draw_particles()
        
        pygame.display.flip()
        pygame.time.wait(5)  # Reduced from 10ms to 5ms for smoother rendering
    
    pygame.quit()
    plt.close()

if __name__ == "__main__":
    main()
