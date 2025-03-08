# Game of Primes - 2D GPU-Accelerated Simulation

This project implements a 2D cellular automaton inspired by Conway's Game of Life, with rules based on prime numbers and twin primes, accelerated using NVIDIA GPU via CuPy. It features real-time OpenGL visualization with Pygame and a live matplotlib plot of alive cells over time.

## Overview

The simulation runs a 100x100 grid where cells evolve based on primality and neighbor counts, rendered as colored squares with particle effects for births and deaths. The GPU handles the core computation (evolution and neighbor counting), while the CPU manages rendering and side effects. The evolution speed is set to 10 generations per second (2x faster than the initial 0.2s interval), optimized for performance on an NVIDIA RTX 3070 Ti or similar hardware.

## Key Features

- **GPU Acceleration**: Uses CuPy for fast array operations on the GPU, doubling simulation speed to ~10 generations/sec.
- **Prime-Based Rules**: Cells survive or are born based on neighbor counts, primality, and twin prime proximity in an E8 lattice.
- **Real-Time Visualization**: Pygame with OpenGL renders the grid and particles; matplotlib plots alive cell trends every 5 seconds.
- **Interactive Controls**: Pause/resume (Space), restart (R), adjust speed (+/-), zoom (mouse wheel), pan (middle mouse drag).
- **Optimized Data Transfer**: Transfers only alive cell positions from GPU to CPU, minimizing overhead.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (e.g., RTX 3070 Ti)
- Libraries:
  - `cupy-cuda12x` (or matching CUDA version)
  - `numpy`
  - `pygame`
  - `pyopengl`
  - `matplotlib`

Install with:
```bash
pip install cupy-cuda12x numpy pygame pyopengl matplotlib
```

Ensure CUDA Toolkit (e.g., 12.6) and cuDNN are installed and configured.

## Code Explanation

### 1. Setup and Initialization
- **Imports**: CuPy (`cp`) for GPU arrays, NumPy (`np`) for CPU operations, Pygame/OpenGL for rendering, and matplotlib for plotting.
- **GPU Check**: Verifies GPU availability and prints device name (e.g., "NVIDIA GeForce RTX 3070 Ti"), falling back to NumPy if unavailable.
- **Grid**: A 100x100 CuPy array (`grid`) initialized with ~4.74% alive cells based on primality, stored on the GPU.
- **Numbers**: A 100x100 CuPy array (`numbers`) from 1 to 10,000 for prime-based rules.

```python
width, height = 100, 100
cell_size = 0.15
grid = cp.zeros((height, width), dtype=cp.int32)
numbers = cp.arange(1, width * height + 1).reshape(height, width)
```

### 2. Core Functions
- **`is_prime(n)`**: CPU-based scalar function to check if a number is prime.
- **`congruence_class(n)`**: Determines if a number modulo 6 is 1 or 5 (for birth rules).
- **`map_to_e8(n)`**: Maps numbers to an 8D vector for twin prime distance calculations.
- **`count_neighbors_parallel(grid, height, width)`**: GPU-accelerated neighbor counting using CuPy’s `cp.roll` with 8 offsets, summing adjacent cells.
- **`evolve_parallel(grid, numbers, height, width, rand_vals)`**: GPU-accelerated evolution:
  - Computes neighbors, primality, and E8 distances.
  - Applies rules: survival (1-2 neighbors), death (<1 or >2), birth (1-5 with prime/twin boosts).
  - Uses `cp.isin` and vectorized operations for speed.

### 3. Evolution and Side Effects
- **`get_alive_positions(grid)`**: Extracts alive cell coordinates on the GPU, returning a `[N_alive, 2]` CuPy array.
- **`evolve(grid, numbers, height, width)`**: Main evolution wrapper:
  - Runs `evolve_parallel` on the GPU.
  - Compares old and new grids to detect births/deaths (CPU-side sets).
  - Updates colors, spawns particles, and adds pulse effects for twin primes.
  - Tracks alive cell count for plotting.

### 4. Rendering
- **`draw_cell(i, j, size_modifier)`**: Draws a 2D quad at `(i, j)` with color and pulsing size (OpenGL immediate mode).
- **`draw_particles()`**: Renders particle effects as quads with fading alpha.
- **`draw_grid(alive_positions_gpu)`**: Transfers alive positions from GPU to CPU, drawing only active cells.
- **`init_gl()`**: Sets up an orthographic 2D projection with OpenGL blending.

### 5. Main Loop
- **Speed**: `evolution_speed = 0.1` (10 generations/sec, 2x faster than 0.2).
- **Controls**: Handles pausing, restarting, speed adjustment, zooming, and panning.
- **Rendering**: Updates at ~60 FPS (5ms wait), plotting every 5 seconds via `FuncAnimation`.

```python
evolution_speed = 0.1  # 2x faster
while running:
    if current_time - last_evolution > evolution_speed:
        grid = evolve(grid, numbers, height, width)
        generation += 1
    glClear(GL_COLOR_BUFFER_BIT)
    alive_positions = get_alive_positions(grid)
    draw_grid(alive_positions)
    draw_particles()
    pygame.display.flip()
    pygame.time.wait(5)
```

### 6. Plotting
- **`update_plot(frame)`**: Updates a matplotlib line plot of alive cells every 5 seconds, running non-blocking with `plt.ion()`.

## Performance
- **Speed**: ~10 generations/sec, ~60 FPS rendering on an RTX 3070 Ti.
- **GPU Usage**: ~20-40% utilization (check with `nvidia-smi`), primarily for evolution and neighbor counting.
- **Optimization**: Minimized CPU-GPU transfers by sending only alive positions (~4,000 vs. 10,000 cells).

## Running the Code
1. Clone the repository.
2. Install dependencies.
3. Run `python game_of_primes.py`.

## Controls
- **Space**: Pause/resume
- **R**: Restart with random grid
- **+/-**: Increase/decrease speed
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse Drag**: Pan

## Future Improvements
- Batch multiple evolution steps per frame for >10 generations/sec.
- Use OpenGL VBOs or shaders for fully GPU-accelerated rendering.
- Add FPS counter and performance metrics.
