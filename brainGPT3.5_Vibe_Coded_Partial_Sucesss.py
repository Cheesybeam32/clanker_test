"""
FieldBrain - Particle-based language model
Architecture: 5 sequential particle grids (Attention, Short-term, Active Thinking, Long-term, Output)
Each neuron is bitpacked into int8: 1 sign + 4 magnitude + 3 momentum
Learning: per-path local perturbation, momentum-scaled magnitude updates, no backprop
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "8"
import sys
import struct
import random
import numpy as np
from numba import njit, prange
import sentencepiece as spm

# ==============================================================================
# CONFIG
# ==============================================================================

TRAIN_FILE    = "train.txt"
MODEL_FILE    = "fieldbrain.bin"
SP_MODEL      = "spm.model"

TRAIN_STEPS   = 100000
SAVE_INTERVAL = 5000

GRID_WIDTH    = 1024
GRID_HEIGHT   = 1024
GRID_DEPTH    = 64

STEP_SIZE     = 6
SCAN_RADIUS   = 3

MAX_GEN_STEPS = 200

G_ATTENTION  = 0
G_SHORTTERM  = 1
G_ACTIVE     = 2
G_LONGTERM   = 3
G_OUTPUT     = 4
G_CONTEXT    = 5
NUM_GRIDS    = 6
FORWARD_GRIDS = 5
# Stupid baby LR
LR = [2.0, 1.0, 0.5, 0.1, 0.8, 0.6]
# Finetune LR
#LR = [0.5, 0.25, 0.1, 0.02, 0.2, 0.15]
THRESHOLD = 0.1

MOMENTUM_CENTER = 3

MAGIC = 0xFB010101

# ==============================================================================
# BITPACK HELPERS (njit)
# ==============================================================================

@njit(cache=True)
def encode_neuron(sign, mag, momentum):
    s = 1 if sign else 0
    m = max(0, min(15, int(mag)))
    p = max(0, min(7, int(momentum)))
    val = (s << 7) | (m << 3) | p
    if val > 127:
        val = val - 256
    return val

@njit(cache=True)
def decode_neuron(n):
    u = n & 0xFF
    sign     = (u >> 7) & 0x1
    mag      = (u >> 3) & 0xF
    momentum = u & 0x7
    return sign, mag, momentum

@njit(cache=True)
def neuron_value(n):
    u = n & 0xFF
    sign = (u >> 7) & 0x1
    mag  = (u >> 3) & 0xF
    v = float(mag)
    if sign:
        v = -v
    return v

@njit(cache=True)
def momentum_signed(n):
    u = n & 0xFF
    mom = u & 0x7
    return mom - 3

@njit(cache=True)
def in_dead_zone(n):
    u = n & 0xFF
    mom = u & 0x7
    return mom == 3 or mom == 4

@njit(cache=True)
def update_momentum(n, direction):
    u = n & 0xFF
    sign = (u >> 7) & 0x1
    mag  = (u >> 3) & 0xF
    mom  = u & 0x7
    new_mom = mom + direction
    if new_mom < 0:
        new_mom = 0
    if new_mom > 7:
        new_mom = 7
    val = (sign << 7) | (mag << 3) | new_mom
    if val > 127:
        val = val - 256
    return val

@njit(cache=True)
def apply_magnitude_update(n, lr):
    if in_dead_zone(n):
        return n
    u = n & 0xFF
    sign = (u >> 7) & 0x1
    mag  = (u >> 3) & 0xF
    mom  = u & 0x7
    mom_s = float(mom - 3)
    delta = mom_s * lr
    signed_mag = float(mag) if sign == 0 else -float(mag)
    signed_mag = signed_mag + delta
    if signed_mag < 0.0:
        new_sign = 1
        new_mag = int(-signed_mag + 0.5)
    else:
        new_sign = 0
        new_mag = int(signed_mag + 0.5)
    if new_mag > 15:
        new_mag = 15
    if new_mag < 0:
        new_mag = 0
    val = (new_sign << 7) | (new_mag << 3) | mom
    if val > 127:
        val = val - 256
    return val

# ==============================================================================
# PARTICLE RUNNER (njit)
# ==============================================================================

@njit(cache=True)
def run_particle(grid, x_launch, z_launch, scan_radius, step_size,
                 grid_width, grid_height, grid_depth):
    x = float(x_launch)
    z = float(z_launch)
    num_steps = grid_height // step_size

    max_visited = num_steps * (2 * scan_radius + 1) * (2 * scan_radius + 1)
    visited_ys = np.empty(max_visited, dtype=np.int32)
    visited_xs = np.empty(max_visited, dtype=np.int32)
    visited_zs = np.empty(max_visited, dtype=np.int32)
    visited_count = 0

    deflection_sum_x = 0.0
    deflection_sum_z = 0.0
    deflection_n = 0

    for step in range(num_steps):
        y = step * step_size
        if y >= grid_height:
            break

        xi = int(x)
        zi = int(z)

        for dz in range(-scan_radius, scan_radius + 1):
            for dx in range(-scan_radius, scan_radius + 1):
                nx = xi + dx
                nz = zi + dz
                if nx < 0 or nx >= grid_width:
                    continue
                if nz < 0 or nz >= grid_depth:
                    continue

                n = grid[y, nz, nx]
                v = neuron_value(n)

                dist = max(abs(dx), abs(dz))
                w = 1.0 if dist == 0 else 1.0 / float(dist)
                deflection_sum_x += v * w
                deflection_sum_z += v * w
                deflection_n += 1

                visited_ys[visited_count] = y
                visited_xs[visited_count] = nx
                visited_zs[visited_count] = nz
                visited_count += 1

        if deflection_n > 0:
            running_avg_x = deflection_sum_x / float(deflection_n)
            x = x + running_avg_x * 0.1
            running_avg_z = deflection_sum_z / float(deflection_n)
            z = z + running_avg_z * 0.1

        x = max(0.0, min(float(grid_width - 1), x))
        z = max(0.0, min(float(grid_depth - 1), z))

    return x, z, visited_ys, visited_xs, visited_zs, visited_count

# ==============================================================================
# PERTURBATION + UPDATE (njit)
# ==============================================================================

@njit(cache=True)
def compute_path_x(grid, x_launch, z_launch, scan_radius, step_size,
                   grid_width, grid_height, grid_depth):
    x = float(x_launch)
    z = float(z_launch)
    num_steps = grid_height // step_size
    deflection_sum_x = 0.0
    deflection_sum_z = 0.0
    deflection_n = 0

    for step in range(num_steps):
        y = step * step_size
        if y >= grid_height:
            break

        xi = int(x)
        zi = int(z)

        for dz in range(-scan_radius, scan_radius + 1):
            for dx in range(-scan_radius, scan_radius + 1):
                nx = xi + dx
                nz = zi + dz
                if nx < 0 or nx >= grid_width:
                    continue
                if nz < 0 or nz >= grid_depth:
                    continue

                n = grid[y, nz, nx]
                v = neuron_value(n)

                dist = max(abs(dx), abs(dz))
                w = 1.0 if dist == 0 else 1.0 / float(dist)
                deflection_sum_x += v * w
                deflection_sum_z += v * w
                deflection_n += 1

        if deflection_n > 0:
            running_avg_x = deflection_sum_x / float(deflection_n)
            x = x + running_avg_x * 0.1
            running_avg_z = deflection_sum_z / float(deflection_n)
            z = z + running_avg_z * 0.1

        x = max(0.0, min(float(grid_width - 1), x))
        z = max(0.0, min(float(grid_depth - 1), z))

    return x

@njit(parallel=True, cache=True)
def perturb_and_update(grid, visited_ys, visited_xs, visited_zs, visited_count,
                        x_launch, z_launch, x_final, target_x,
                        lr, scan_radius, step_size, grid_width, grid_height, grid_depth):
    base_loss = abs(x_final - target_x)

    for i in prange(visited_count):
        vy = visited_ys[i]
        vx = visited_xs[i]
        vz = visited_zs[i]

        n = grid[vy, vz, vx]
        mom_s = momentum_signed(n)

        if in_dead_zone(n):
            if target_x > x_final:
                grid[vy, vz, vx] = update_momentum(n, 1)
            else:
                grid[vy, vz, vx] = update_momentum(n, -1)
            continue

        n_updated = apply_magnitude_update(n, lr)
        grid[vy, vz, vx] = n_updated

        x_new = compute_path_x(grid, x_launch, z_launch, scan_radius, step_size,
                                grid_width, grid_height, grid_depth)
        new_loss = abs(x_new - target_x)

        if new_loss < base_loss:
            direction = 1 if mom_s >= 0 else -1
            grid[vy, vz, vx] = update_momentum(n_updated, direction)
        else:
            grid[vy, vz, vx] = n
            direction = -1 if mom_s >= 0 else 1
            grid[vy, vz, vx] = update_momentum(n, direction)

# ==============================================================================
# GRID INIT
# ==============================================================================

def init_grids():
    grids = []
    for _ in range(NUM_GRIDS):
        g = np.zeros((GRID_HEIGHT, GRID_DEPTH, GRID_WIDTH), dtype=np.int8)
        g[:] = 3
        grids.append(g)
    return grids

# ==============================================================================
# VOCAB / TOKEN <-> X MAPPING
# ==============================================================================

def build_vocab_from_spm(sp_processor):
    vocab_size = sp_processor.get_piece_size()
    id_to_token = [sp_processor.id_to_piece(i) for i in range(vocab_size)]
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return id_to_token, token_to_id, vocab_size

def token_to_x(token_id, vocab_size, grid_width):
    if vocab_size <= 1:
        return grid_width // 2
    return int(token_id * (grid_width - 1) / (vocab_size - 1))

def x_to_token(x, vocab_size, grid_width):
    x = max(0.0, min(float(grid_width - 1), x))
    token_id = int(round(x * (vocab_size - 1) / (grid_width - 1)))
    return max(0, min(vocab_size - 1, token_id))

# ==============================================================================
# SAVE / LOAD (raw binary)
# ==============================================================================

def save_model(path, grids, vocab_size):
    with open(path, 'wb') as f:
        f.write(struct.pack('<IIIII', MAGIC, NUM_GRIDS, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH))
        for g in grids:
            f.write(g.tobytes())
        f.write(struct.pack('<I', vocab_size))
    print(f"[save] wrote {path}")

def load_model(path):
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as f:
        magic, n_grids, gw, gh, gd = struct.unpack('<IIIII', f.read(20))
        if magic != MAGIC:
            print(f"[load] bad magic in {path}, ignoring")
            return None, None
        if n_grids != NUM_GRIDS or gw != GRID_WIDTH or gh != GRID_HEIGHT or gd != GRID_DEPTH:
            print(f"[load] grid dimension mismatch, ignoring")
            return None, None
        grids = []
        for _ in range(NUM_GRIDS):
            raw = f.read(GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH)
            g = np.frombuffer(raw, dtype=np.int8).reshape((GRID_HEIGHT, GRID_DEPTH, GRID_WIDTH)).copy()
            grids.append(g)
        vocab_size_saved, = struct.unpack('<I', f.read(4))
    print(f"[load] loaded {path} — vocab_size={vocab_size_saved}")
    return grids, vocab_size_saved

# ==============================================================================
# PREFILL
# ==============================================================================

def prefill(grids, token_ids, vocab_size):
    x = float(GRID_WIDTH // 2)
    z = float(GRID_DEPTH // 2)
    for tid in token_ids:
        x_launch = float(token_to_x(tid, vocab_size, GRID_WIDTH))
        x, z, *_ = run_particle(
            grids[G_CONTEXT], x_launch, z,
            SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH
        )
    return x, z

# ==============================================================================
# FULL FORWARD PASS (inference)
# ==============================================================================

def forward(grids, x_in, z_in, vocab_size):
    x = x_in
    z = z_in
    for g_idx in range(FORWARD_GRIDS):
        x, z, _vy, _vx, _vz, _vc = run_particle(
            grids[g_idx], x, z,
            SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH
        )
    return x, x_to_token(x, vocab_size, GRID_WIDTH)

# ==============================================================================
# TRAINING FORWARD + UPDATE
# ==============================================================================

def train_forward_update(grids, x_in, z_in, target_token_id, vocab_size):
    target_x = float(token_to_x(target_token_id, vocab_size, GRID_WIDTH))

    visited_data = []
    x = x_in
    z = z_in
    for g_idx in range(FORWARD_GRIDS):
        x_launch = x
        z_launch = z
        x_out, z_out, vy, vx, vz, vc = run_particle(
            grids[g_idx], x_launch, z_launch,
            SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH
        )
        visited_data.append((x_launch, z_launch, x_out, z_out, vy, vx, vz, vc))
        x = x_out
        z = z_out

    x_final  = x
    predicted = x_to_token(x_final, vocab_size, GRID_WIDTH)

    for g_idx in range(FORWARD_GRIDS):
        x_launch, z_launch, x_out, z_out, vy, vx, vz, vc = visited_data[g_idx]
        perturb_and_update(
            grids[g_idx], vy, vx, vz, vc,
            x_launch, z_launch, x_out, target_x,
            LR[g_idx], SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH
        )

    return predicted, x_final, target_x

# ==============================================================================
# GENERATION LOOP
# ==============================================================================

def generate(grids, x_seed, z_seed, vocab_size, eos_id, max_steps=MAX_GEN_STEPS):
    x = x_seed
    z = z_seed
    generated = []
    for _ in range(max_steps):
        x, token_id = forward(grids, x, z, vocab_size)
        if token_id == eos_id:
            break
        generated.append(token_id)
    return generated

# ==============================================================================
# TRAIN LOOP
# ==============================================================================

STATS_WINDOW = 100

def train(grids, vocab_size, sp, eos_id):
    print(f"[train] loading {TRAIN_FILE}")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"[train] {len(lines)} lines, {TRAIN_STEPS} steps, save every {SAVE_INTERVAL}")

    correct_buf  = []
    distance_buf = []

    for step in range(1, TRAIN_STEPS + 1):
        line = random.choice(lines)
        token_ids = sp.encode(line)
        if len(token_ids) < 2:
            continue

        i      = random.randint(1, len(token_ids) - 1)
        prefix = token_ids[:i]
        target = token_ids[i]

        x_seed, z_seed               = prefill(grids, prefix, vocab_size)
        predicted, x_final, target_x = train_forward_update(grids, x_seed, z_seed, target, vocab_size)

        correct  = predicted == target
        distance = abs(x_final - target_x)

        correct_buf.append(1 if correct else 0)
        distance_buf.append(distance)
        if len(correct_buf)  > STATS_WINDOW: correct_buf.pop(0)
        if len(distance_buf) > STATS_WINDOW: distance_buf.pop(0)

        if step % 1 == 0:
            target_tok   = sp.decode([target])
            pred_tok     = sp.decode([predicted])
            avg_acc      = sum(correct_buf)  / len(correct_buf)
            avg_dist     = sum(distance_buf) / len(distance_buf)
            print(
                f"  step {step:6d}"
                f" | target: {target_tok!r:15s}"
                f" | pred: {pred_tok!r:15s}"
                f" | {'✓' if correct else '✗'}"
                f" | dist: {distance:6.1f}"
                f" | avg_acc: {avg_acc*100:5.1f}%"
                f" | avg_dist: {avg_dist:6.1f}"
            )

        if step % SAVE_INTERVAL == 0:
            save_model(MODEL_FILE, grids, vocab_size)

    save_model(MODEL_FILE, grids, vocab_size)
    print("[train] done")

# ==============================================================================
# CHAT LOOP
# ==============================================================================

def chat(grids, vocab_size, sp, eos_id):
    print("[chat] ready — empty line to quit")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break

        token_ids = sp.encode(user_input)
        if not token_ids:
            continue

        x_seed, z_seed = prefill(grids, token_ids, vocab_size)
        response_ids   = generate(grids, x_seed, z_seed, vocab_size, eos_id)
        response       = sp.decode(response_ids) if response_ids else "[no output]"
        print(f"brain: {response}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not os.path.exists(SP_MODEL):
        print(f"[error] SPM model not found: {SP_MODEL}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)

    _, _, vocab_size = build_vocab_from_spm(sp)
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else vocab_size - 1

    print(f"[init] vocab_size={vocab_size}, eos_id={eos_id}")
    print(f"[init] grid={GRID_WIDTH}x{GRID_HEIGHT}x{GRID_DEPTH}, scan_radius={SCAN_RADIUS}, step_size={STEP_SIZE}")
    print(f"[init] LR={LR}")

    grids, saved_vocab = load_model(MODEL_FILE)
    if grids is None:
        print("[init] no saved model, initialising fresh grids")
        grids = init_grids()
    elif saved_vocab != vocab_size:
        print(f"[init] vocab mismatch (saved={saved_vocab}, current={vocab_size}), reinitialising")
        grids = init_grids()

    print("[init] warming up JIT...")
    dummy = np.zeros((GRID_HEIGHT, GRID_DEPTH, GRID_WIDTH), dtype=np.int8)
    dummy[:] = 3
    run_particle(dummy, float(GRID_WIDTH // 2), float(GRID_DEPTH // 2), SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH)
    compute_path_x(dummy, float(GRID_WIDTH // 2), float(GRID_DEPTH // 2), SCAN_RADIUS, STEP_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH)
    print("[init] JIT ready")

    print()
    mode = input("train or chat? ").strip().lower()
    print()

    if mode == "train":
        train(grids, vocab_size, sp, eos_id)
    elif mode == "chat":
        chat(grids, vocab_size, sp, eos_id)
    else:
        print(f"[error] unknown mode: {mode!r}")
        sys.exit(1)

if __name__ == "__main__":
    main()