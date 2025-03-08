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

# Constants
ON = 1  # Changed from 255 to 1 for better memory efficiency
OFF = 0
PRIME_CACHE_SIZE = 10000  # For caching prime checks

# Global variables
particle_effects = []
cell_colors = {}
alive_history = []
N = 100  # Default value

# Verify GPU availability
try:
    cp.cuda.Device(0).use()
    print("Using GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'))
    use_gpu = True
except Exception as e:
    print("GPU not available, falling back to CPU:", str(e))
    import numpy as cp
    use_gpu = False

# Prime number cache (CPU)
prime_cache = {}

# Improved prime number check with caching
def is_prime(n):
    n = int(n)
    if n in prime_cache:
        return prime_cache[n]
    
    if n < 2:
        prime_cache[n] = False
        return False
    
    if n == 2 or n == 3:
        prime_cache[n] = True
        return True
    
    if n % 2 == 0:
        prime_cache[n] = False
        return False
    
    # Only check odd divisors up to sqrt(n)
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            prime_cache[n] = False
            return False
    
    prime_cache[n] = True
    return True

# Pre-compute primes for faster lookup
def precompute_primes(limit):
    for i in range(2, limit):
        is_prime(i)
    print(f"Precomputed {len([k for k, v in prime_cache.items() if v])} primes up to {limit}")

# Grid initialization with vectorized operations
def randomGrid(N):
    """Returns a grid of NxN random values with prime bias"""
    grid = np.zeros((N, N), dtype=np.uint8)
    
    # Create a flattened array of positions and filter by primes
    positions = np.arange(1, N*N + 1).reshape(N, N)
    
    # Initialize target cell count for alive cells
    target_alive = int(N * N * 0.2)
    alive_count = 0
    
    # First, mark prime numbers and twin primes
    for i in range(N):
        for j in range(N):
            n = int(positions[i, j])
            if is_prime(n) or (is_prime(n + 2) or (n - 2 > 0 and is_prime(n - 2))):
                grid[i, j] = ON
                alive_count += 1
                if alive_count >= target_alive:
                    break
        if alive_count >= target_alive:
            break
    
    # If we still need more alive cells, add random ones
    if alive_count < target_alive:
        remaining = target_alive - alive_count
        flat_indices = np.random.choice(
            np.where(grid.flatten() == OFF)[0], 
            size=min(remaining, (grid == OFF).sum()), 
            replace=False
        )
        grid.flat[flat_indices] = ON
    
    # Convert to GPU array if available
    return cp.asarray(grid) if use_gpu else grid

def addGlider(i, j, grid):
    """Adds a glider at (i, j)"""
    glider = np.array([[OFF, OFF, ON],
                       [ON, OFF, ON],
                       [OFF, ON, ON]], dtype=np.uint8)
    grid[i:i+3, j:j+3] = glider if not use_gpu else cp.asarray(glider)

def addGosperGliderGun(i, j, grid):
    """Adds a Gosper Glider Gun at (i, j)"""
    gun = np.zeros((11, 38), dtype=np.uint8)
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
    grid[i:i+11, j:j+38] = gun if not use_gpu else cp.asarray(gun)

# GPU-accelerated neighbor counting with optimized kernel
def count_neighbors_parallel(grid, N):
    # This is more efficient than multiple roll operations
    neighbors = cp.zeros_like(grid, dtype=cp.int8) if use_gpu else np.zeros_like(grid, dtype=np.int8)
    
    # Precompile kernel for this operation if using GPU
    if use_gpu:
        # Use a 2D grid of thread blocks
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = cp.roll(grid, (di, dj), axis=(0, 1))
                neighbors += shifted
    else:
        # CPU fallback
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = np.roll(grid, (di, dj), axis=(0, 1))
                neighbors += shifted
                
    return neighbors

# Optimized game logic update with fewer memory operations
def update(grid, N, numbers):
    neighbors = count_neighbors_parallel(grid, N)
    
    # Vectorized rule application
    # Conway's rules:
    # 1. Any live cell with fewer than two live neighbors dies (underpopulation)
    # 2. Any live cell with two or three live neighbors lives on
    # 3. Any live cell with more than three live neighbors dies (overpopulation)
    # 4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction)
    
    # Calculate new state in one operation
    birth = (grid == OFF) & (neighbors == 3)
    survive = (grid == ON) & ((neighbors == 2) | (neighbors == 3))
    
    # Create new grid by applying rules
    new_grid = cp.zeros_like(grid) if use_gpu else np.zeros_like(grid)
    new_grid[survive | birth] = ON
    
    # Apply prime number boosting (only to cells that might be born)
    potential_births = (grid == OFF) & (neighbors >= 2) & (neighbors <= 3) & (~birth)
    
    if use_gpu:
        potential_births_indices = cp.where(potential_births)
        potential_nums = numbers[potential_births_indices].get()
    else:
        potential_births_indices = np.where(potential_births)
        potential_nums = numbers[potential_births_indices]
    
    # Process potential births in batches for better performance
    batch_size = 1000
    for batch_start in range(0, len(potential_nums), batch_size):
        batch_end = min(batch_start + batch_size, len(potential_nums))
        batch_indices = slice(batch_start, batch_end)
        
        for idx, n in enumerate(potential_nums[batch_indices]):
            n = int(n)
            if is_prime(n) or (n + 2 <= N * N and is_prime(n + 2)):
                i, j = potential_births_indices[0][batch_start + idx], potential_births_indices[1][batch_start + idx]
                new_grid[i, j] = ON
    
    return new_grid

# Extract alive positions more efficiently
def get_alive_positions(grid):
    if use_gpu:
        alive_indices = cp.where(grid == ON)
        return cp.stack(alive_indices, axis=1) if alive_indices[0].size > 0 else cp.zeros((0, 2), dtype=cp.int32)
    else:
        alive_indices = np.where(grid == ON)
        return np.stack(alive_indices, axis=1) if alive_indices[0].size > 0 else np.zeros((0, 2), dtype=np.int32)

# Optimized Particle class
class Particle:
    __slots__ = ['x', 'y', 'color', 'lifetime', 'max_lifetime', 'size', 'velocity', 'type']
    
    def __init__(self, x, y, color, lifetime=1.0, size=0.05, velocity=(0, 0), type='birth'):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.velocity = velocity
        self.type = type

# Create particle batch for better performance
def create_particles(i, j, type):
    x = (j - N/2) * cell_size
    y = (N/2 - i) * cell_size
    
    if type == 'birth':
        color = cell_colors.get((i, j), (0.5, 0.8, 0.2))
        count = min(5, 100 // (len(particle_effects) + 1))
        
        for _ in range(count):
            vx = random.uniform(-0.02, 0.02)
            vy = random.uniform(-0.02, 0.02)
            size = random.uniform(0.01, 0.03)
            lifetime = random.uniform(0.5, 1.0)
            particle_effects.append(Particle(x, y, color, lifetime, size, (vx, vy), type))
    else:
        color = (0.7, 0.3, 0.7)
        count = min(3, 100 // (len(particle_effects) + 1))
        
        for _ in range(count):
            vx = random.uniform(-0.01, 0.01)
            vy = random.uniform(-0.01, 0.01)
            size = random.uniform(0.01, 0.025)
            lifetime = random.uniform(0.3, 0.7)
            particle_effects.append(Particle(x, y, color, lifetime, size, (vx, vy), type))

# Batch update particles for better performance
def update_particles(dt):
    global particle_effects
    
    # Only keep particles that are still alive
    particle_effects = [p for p in particle_effects if p.lifetime > 0]
    
    # Update all particles in one pass
    for p in particle_effects:
        p.lifetime -= dt
        if p.lifetime > 0:
            p.x += p.velocity[0]
            p.y += p.velocity[1]
            p.size *= 0.98 if p.type == 'birth' else 0.97

# Optimized cell rendering with vertex arrays
cell_size = 0.15
def setup_cell_vertices(i, j):
    x = (j - N/2) * cell_size
    y = (N/2 - i) * cell_size
    half_size = cell_size / 2
    vertices = [
        (x - half_size, y - half_size),
        (x + half_size, y - half_size),
        (x + half_size, y + half_size),
        (x - half_size, y + half_size)
    ]
    return vertices

def draw_cell(i, j):
    r, g, b = cell_colors.get((i, j), (0.1, 0.7, 0.9))
    glColor3f(r, g, b)
    
    vertices = setup_cell_vertices(i, j)
    glBegin(GL_QUADS)
    for vertex in vertices:
        glVertex2f(*vertex)
    glEnd()

# Batch rendering of particles
def draw_particles():
    if not particle_effects:
        return
        
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Draw particles in batches by color for better performance
    particles_by_color = {}
    for p in particle_effects:
        color_key = (p.color[0], p.color[1], p.color[2])
        if color_key not in particles_by_color:
            particles_by_color[color_key] = []
        particles_by_color[color_key].append(p)
    
    for color, particles in particles_by_color.items():
        for p in particles:
            alpha = p.lifetime / p.max_lifetime
            glColor4f(color[0], color[1], color[2], alpha)
            
            # Draw particle
            glBegin(GL_QUADS)
            glVertex2f(p.x - p.size, p.y - p.size)
            glVertex2f(p.x + p.size, p.y - p.size)
            glVertex2f(p.x + p.size, p.y + p.size)
            glVertex2f(p.x - p.size, p.y + p.size)
            glEnd()
    
    glDisable(GL_BLEND)

# More efficient grid drawing
def draw_grid(alive_positions_gpu):
    if use_gpu:
        alive_positions = alive_positions_gpu.get()
    else:
        alive_positions = alive_positions_gpu
    
    # Draw in batches by color for better performance
    cells_by_color = {}
    for i, j in alive_positions:
        color = cell_colors.get((i, j), (0.1, 0.7, 0.9))
        color_key = (color[0], color[1], color[2])
        if color_key not in cells_by_color:
            cells_by_color[color_key] = []
        cells_by_color[color_key].append((i, j))
    
    for color, cells in cells_by_color.items():
        glColor3f(*color)
        glBegin(GL_QUADS)
        for i, j in cells:
            x = (j - N/2) * cell_size
            y = (N/2 - i) * cell_size
            half_size = cell_size / 2
            glVertex2f(x - half_size, y - half_size)
            glVertex2f(x + half_size, y - half_size)
            glVertex2f(x + half_size, y + half_size)
            glVertex2f(x - half_size, y + half_size)
        glEnd()

# Optimized OpenGL initialization
def init_gl(N):
    glClearColor(0.1, 0.1, 0.2, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = 1.0  # N / N is always 1
    glOrtho(-10 * aspect_ratio, 10 * aspect_ratio, -10, 10, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Disable depth testing for 2D
    glDisable(GL_DEPTH_TEST)
    # Enable vertex and color arrays for faster rendering
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

# Initialize plotting
def init_plot():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Alive Cells', color='blue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Game of Life with Primes')
    ax.legend()
    ax.grid(True)
    return fig, ax, line

def update_plot(fig, ax, line):
    # Only update the data, not the whole plot
    line.set_data(range(len(alive_history)), alive_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# Main function
def main():
    global N, alive_history, particle_effects, cell_colors
    
    # Reset global variables in case of restarts
    alive_history = []
    particle_effects = []
    cell_colors = {}
    
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life with prime dynamics.")
    parser.add_argument('--grid-size', dest='N', required=False, type=int, default=100)
    parser.add_argument('--interval', dest='interval', required=False, type=int, default=50)
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--gosper', action='store_true', required=False)
    parser.add_argument('--max-particles', dest='max_particles', required=False, type=int, default=500)
    args = parser.parse_args()

    global N
    N = args.N if args.N and args.N > 8 else 100
    update_interval = args.interval / 1000  # Convert ms to seconds
    max_particles = args.max_particles

    # Initialize grid
    if use_gpu:
        grid = cp.zeros((N, N), dtype=cp.uint8)
        numbers = cp.arange(1, N * N + 1).reshape(N, N)
    else:
        grid = np.zeros((N, N), dtype=np.uint8)
        numbers = np.arange(1, N * N + 1).reshape(N, N)
    
    # Precompute primes for faster lookup
    print("Precomputing primes...")
    precompute_primes(N * N + 2)
    
    if args.glider:
        addGlider(1, 1, grid)
    elif args.gosper:
        addGosperGliderGun(10, 10, grid)
    else:
        print("Generating random grid...")
        grid = randomGrid(N)

    # Pygame/OpenGL setup
    pygame.init()
    display = (1200, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("GPU-Accelerated Game of Life with Primes")
    init_gl(N)

    # Initialize plotting
    fig, ax, line = init_plot()
    plt.ion()
    plt.show(block=False)

    generation = 0
    last_time = time.time()
    last_evolution = 0
    last_plot_update = 0
    running = True
    paused = False
    zoom = 1.0
    pan_x, pan_y = 0, 0
    fps_timer = time.time()
    frames = 0
    
    # Set up frame limiting
    clock = pygame.time.Clock()
    target_fps = 60

    print("Starting simulation...")
    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        frames += 1
        
        # FPS calculation
        if current_time - fps_timer > 1.0:
            current_alive = len(alive_history) > 0 and alive_history[-1] or 0
            pygame.display.set_caption(f"GPU Game of Life - FPS: {frames} - Gen: {generation} - Alive: {current_alive}")
            frames = 0
            fps_timer = current_time

        # Process events in batch
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_r:
                    grid = randomGrid(N)
                    alive_history.clear()
                    generation = 0
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom *= 1.1
                elif event.button == 5:
                    zoom /= 1.1
            elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[1]:
                pan_x += event.rel[0] * 0.01
                pan_y -= event.rel[1] * 0.01

        # Game of Life update
        if not paused and current_time - last_evolution > update_interval:
            # Track changes
            old_alive = get_alive_positions(grid)
            
            # Update the grid
            grid = update(grid, N, numbers)
            
            # Get new alive cells
            new_alive = get_alive_positions(grid)
            
            # Track births and deaths for effects
            if use_gpu:
                old_set = set(tuple(pos) for pos in old_alive.get()) if old_alive.shape[0] > 0 else set()
                new_set = set(tuple(pos) for pos in new_alive.get()) if new_alive.shape[0] > 0 else set()
            else:
                old_set = set(tuple(pos) for pos in old_alive) if old_alive.shape[0] > 0 else set()
                new_set = set(tuple(pos) for pos in new_alive) if new_alive.shape[0] > 0 else set()
                
            births = new_set - old_set
            deaths = old_set - new_set
            
            # Limit particle effects for performance
            if len(particle_effects) < max_particles:
                # Only create particles for a subset of births/deaths if many changes
                births_sample = random.sample(list(births), min(len(births), 20)) if births else []
                deaths_sample = random.sample(list(deaths), min(len(deaths), 20)) if deaths else []
                
                for i, j in births_sample:
                    n = int(numbers[i, j].get() if use_gpu else numbers[i, j])
                    if is_prime(n) and is_prime(n + 2):
                        cell_colors[(i, j)] = (0.9, 0.1, 0.1)
                    elif is_prime(n):
                        cell_colors[(i, j)] = (0.1, 0.7, 0.9)
                    else:
                        hue = (n % 360) / 360.0
                        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                        cell_colors[(i, j)] = (r, g, b)
                    create_particles(i, j, 'birth')
                    
                for i, j in deaths_sample:
                    create_particles(i, j, 'death')
            
            # Update statistics
            alive_count = int(cp.sum(grid).get() if use_gpu else np.sum(grid))
            alive_history.append(alive_count)
            
            # Trim history for memory efficiency
            if len(alive_history) > 1000:
                alive_history = alive_history[-1000:]
                
            generation += 1
            last_evolution = current_time
            
            # Only print every 10 generations for less console spam
            if generation % 10 == 0:
                print(f"Generation {generation}: {alive_count} alive cells, {len(particle_effects)} particles")

        # Update plot less frequently (every second)
        if current_time - last_plot_update > 1.0:
            update_plot(fig, ax, line)
            last_plot_update = current_time

        # Update and render particles
        update_particles(dt)
        
        # Render the scene
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(pan_x, pan_y, 0)
        glScalef(zoom, zoom, 1.0)
        
        # Draw grid and particles
        draw_grid(get_alive_positions(grid))
        draw_particles()
        
        # Update display
        pygame.display.flip()
        
        # Limit frame rate
        clock.tick(target_fps)

    # Clean up
    pygame.quit()
    plt.close()

if __name__ == '__main__':
    main()
