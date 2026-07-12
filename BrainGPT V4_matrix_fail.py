# In the event of a hidden problem watch this :D
#https://music.youtube.com/watch?v=-LaWn5OjX-Q
"""
Things to change:
Make the dense grids actually use 256 dims (Major bug)
Potentially make the main grid use backprop over time (Minor feature)
Make a birthday cake + better name than "theat" or "BrainGPT" (Side patch)
    Its just a nice thing to do lol and "theat" is hard to pronounce and sounds a bit like "feet"
"""
import numpy as np
import sentencepiece as spm
import os
sp = spm.SentencePieceProcessor()
sp.load('spm.model')
mode = "chat"
X_bounds = 2048
Y_bounds = 2048
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

def v_compile(ctx_miku):
    teto = np.arange(vocab_size, dtype=np.float32)
    centers = brain.i_vecs[ctx_miku][:, None]
    compiled_v = 1.0 - np.abs(teto - centers) / (vocab_size - 1)
    compiled_v = np.clip(compiled_v, 0, 1)
    return compiled_v

def x_compile(x_his):
    teto = np.arange(vocab_size, dtype=np.float32)
    neru = np.array(x_his, dtype=np.float32)[:, None]
    s_ctx = (neru / X_bounds) * (vocab_size - 1)
    compiled_x = 1.0 - np.abs(teto - s_ctx) / (vocab_size - 1)
    compiled_x = np.clip(compiled_x, 0, 1)
    return compiled_x

class brain():
    def __init__(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            self.neurons = data["neurons"]
            self.i_grid = data["i_grid"]
            self.v_grid = data["v_grid"]
            self.o_grid = data["o_grid"]
            self.i_vecs = data["i_vecs"]
            print("Using saved file")
        else:
            self.neurons = np.zeros((Y_bounds, X_bounds), dtype=np.int8)
            self.i_grid = np.zeros((vocab_size, Input_Dims), dtype=np.float16)
            self.v_grid = np.zeros((vocab_size, Input_Dims), dtype=np.float16)
            self.o_grid = np.zeros((Input_Dims, vocab_size), dtype=np.float16)
            self.i_vecs = np.linspace(0, vocab_size - 1, vocab_size, dtype=np.float32)
            print("Making new file")
    def input_layer(self, compiled_v):
        attn = np.dot(compiled_v, self.i_grid) 
        x_val = np.mean(attn)
        return x_val
    def i_layer_learn(self, error_vec, compiled_v):
        v_sum = np.sum(compiled_v, axis=0)
        grad_matrix = np.outer(v_sum, error_vec)
        self.i_grid -= grad_matrix * LR
    def vector_layer(self, compiled_v):
        attn = np.dot(compiled_v, self.v_grid)
        x = np.mean(attn)
        return x
    def v_layer_learn(self, error_vec, compiled_v):
        v_sum = np.sum(compiled_v, axis=0)
        grad_matrix = np.outer(v_sum, error_vec)
        self.v_grid -= grad_matrix * LR
    def output_layer(self, compiled_x):
        path_act = np.mean(compiled_x, axis=1)  # (256,)
        pred_vec = np.dot(path_act, self.o_grid)  # (vocab_size,)
        return pred_vec
    def o_layer_learn(self, error_vec, compiled_x):
        path_act = np.sum(compiled_x, axis=1)
        grad = np.outer(path_act, error_vec)
        self.o_grid -= grad * LR
    def i_vecs_learn(self, ctx_miku, error_scalar):
        blue = self.i_vecs[ctx_miku]
        teto = np.arange(vocab_size, dtype=np.float32)
        sign = np.sign(blue[:, None] - teto[None, :])
        grad = np.sum(sign / (vocab_size - 1), axis=1)
        self.i_vecs[ctx_miku] -= grad * error_scalar * LR

    def walk_thru(self, x_val, x, sky, x_his, interval):
        y = 0
        weights = []
        c_scan = 0
        while y < Y_bounds:
            influence = 0
            for ny in range(-Radius, Radius + 1):
                for nx in range(-Radius, Radius + 1):
                    wy = int(y + ny)
                    wx = int(x + nx)
                    if wx < 0 or wx >= X_bounds:
                        continue
                    if wy < 0 or wy >= Y_bounds:
                        continue
                    strength = decode(wy, wx)
                    weights.append((strength, wy, wx))
                    influence += strength * x_val
            x += influence / sky
            y += step_size
            c_scan += 1
            if c_scan % interval == 0 and len(x_his) < Input_Dims:
                x_his.append(x)
        return x, weights, x_his
    def learn(self, error, weights, x_val):
        for strength, wy, wx in weights:
            strength -= error * x_val * LR
            encode(strength, wy, wx)
load = dataset()
brain = brain()
def main():
    tok = 0
    step = 0
    c_err = None
    sky = Y_bounds // step_size
    interval = sky // Input_Dims
    if mode == "train":
        while True:
            miku = next(load)
            for tok in range(1, len(miku)):
                x_his = []
                pred_miku = miku[tok]
                ctx_miku = miku[:tok]
                compiled_v = v_compile(ctx_miku) 
                x_val = brain.input_layer(compiled_v)
                x = brain.vector_layer(compiled_v)
                x, weights, x_his = brain.walk_thru(x_val, x, sky, x_his, interval)
                compiled_x = x_compile(x_his)  
                prediction_vector = brain.output_layer(compiled_x)
                target = np.eye(vocab_size, dtype=np.float32)[pred_miku]
                error_vector = prediction_vector - target
                error_grid = np.dot(brain.o_grid, error_vector)
                error_scalar = np.mean(error_vector)
                brain.i_layer_learn(error_grid, compiled_v)
                brain.v_layer_learn(error_grid, compiled_v)
                brain.o_layer_learn(error_vector, compiled_x)
                brain.i_vecs_learn(ctx_miku, error_scalar)
                brain.learn(error_scalar, weights, x_val)
                step += 1
                if step == 500:
                    np.savez(
                        "brain.npz",
                        neurons=brain.neurons,
                        i_grid=brain.i_grid,
                        v_grid=brain.v_grid,
                        o_grid=brain.o_grid,
                        i_vecs=brain.i_vecs
                    )
                    print("Saved")
                    step = 0
                if c_err == None:
                    c_err = abs(error_scalar)
                else:
                    c_err = (0.99 * c_err) + (0.01 * abs(error_scalar))
                print(f"Loss Matrix Diff: {abs(error_scalar):.6f}, Smoothed Loss: {c_err:.6f}")
    if mode == "chat":
        sky = Y_bounds // step_size
        while True:
            x_his = []
            data_in = input("Human: ")
            ctx_miku = sp.encode(data_in)
            compiled_v = v_compile(ctx_miku) 
            x_val = brain.input_layer(compiled_v)
            x = brain.vector_layer(compiled_v)
            x, weights, x_his = brain.walk_thru(x_val, x, sky, x_his, interval)
            compiled_x = x_compile(x_his)  
            prediction_vector = brain.output_layer(compiled_x)
            c_token = int(np.argmax(prediction_vector))
            o_token = sp.decode([c_token])
            print(f"Theat: {o_token}")
main()