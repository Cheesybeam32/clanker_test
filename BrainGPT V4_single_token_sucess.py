# In the event of a hidden problem watch this :D
#https://music.youtube.com/watch?v=-LaWn5OjX-Q
import numpy as np
import sentencepiece as spm
import os
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

X_bounds = 1048
Y_bounds = 1048
Radius = 3
step_size = 6
LR = 0.001

Input_Dims = 256
vocab_size = sp.vocab_size()

def dataset():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def encode(strength, wy, wx):
    strength = max(-1.0, min(1.0, strength))
    strength *= 127
    brain.neurons[wy, wx] = strength

def decode(wy, wx):
    strength = brain.neurons[wy, wx]
    strength /= 127
    return strength

def vocab_compiler(ctx_miku):
    compiled_v = np.eye(vocab_size, dtype=np.float32)[ctx_miku]
    return compiled_v

class brain():
    def __init__(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            brain.neurons = data["neurons"]
            brain.i_grid = data["i_grid"]
            brain.v_grid = data["v_grid"]
            print("Using saved file")
        else:
            self.neurons = np.zeros((Y_bounds, X_bounds), dtype=np.int8)
            self.i_grid = np.zeros((vocab_size, Input_Dims), dtype=np.float16)
            self.v_grid = np.zeros((vocab_size, Input_Dims), dtype=np.float16)
            print("Making new file")
    def input_layer(self, compiled_v):
        attn = np.dot(compiled_v, self.i_grid)
        x_val = np.mean(attn)
        return x_val
    def i_layer_learn(self, error, compiled_v):
        ctx = compiled_v.shape[0]
        shape = ctx * Input_Dims
        grad_matrix = np.full((ctx, Input_Dims), fill_value=(error / shape))
        changes = np.dot(compiled_v.T, grad_matrix)
        self.i_grid -= changes * LR
    def vector_layer(self, compiled_v):
        attn = np.dot(compiled_v, self.v_grid)
        x = np.mean(attn)
        return x
    def v_layer_learn(self, error, compiled_v):
        ctx = compiled_v.shape[0]
        shape = ctx * Input_Dims
        grad_matrix = np.full((ctx, Input_Dims), fill_value=(error / shape))
        changes = np.dot(compiled_v.T, grad_matrix)
        self.v_grid -= changes * LR
    def walk_thru(self, x_val, x):
        y = 0
        weights = []
        while y < Y_bounds:
            influence = 0
            for ny in range(-Radius, Radius + 1):
                for nx in range(-Radius, Radius + 1):
                    wy = int(y + ny)
                    wx = int(x + nx)
                    if wx < 0 or wx >=X_bounds:
                        continue
                    if wy < 0 or wy >= Y_bounds:
                        continue
                    strength = decode(wy, wx)
                    weights.append((strength, wy, wx))
                    influence += strength * x_val
            x += influence
            y += step_size
        return x, weights
    def learn(self, error, x_val):
        for strength, wy, wx in weights:
            strength -= error * x_val * LR
            encode(strength, wy, wx)
load = dataset()
brain = brain()
tok = 0
step = 0
while True:
    miku = next(load)
    for tok in range(1, len(miku)):
        pred_miku = miku[tok]
        ctx_miku = miku[:tok]
        compiled_v = vocab_compiler(ctx_miku)
        x_val = brain.input_layer(compiled_v)
        x = brain.vector_layer(compiled_v)
        x, weights = brain.walk_thru(x_val, x)
        error = x - (pred_miku / vocab_size) * X_bounds
        brain.i_layer_learn(error, compiled_v)
        brain.v_layer_learn(error, compiled_v)
        brain.learn(error, x_val)
        step += 1
        if step == 500:
            np.savez(
                "brain.npz",
                neurons=brain.neurons,
                i_grid=brain.i_grid,
                v_grid=brain.v_grid
            )
            print("Saved")
            step = 0
        print(f"{error}")