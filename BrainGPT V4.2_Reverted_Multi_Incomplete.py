# In the event of a hidden problem watch this :D
#https://music.youtube.com/watch?v=-LaWn5OjX-Q
"""
If i end up showing anyone ITS NOT OPTIMIZED YET. Ill get it to work FIRST
Checklist to change (Read info doc):
1. Token embeddings (new data structure)
2. QKV projections (new parameters)
3. Content-dependent magnitude (relevance score)
4. Contextualized embeddings ("movable" embeddings — disambiguation, e.g. river bank vs. financial bank)
5. Content-dependent direction (novel — not standard attention)
6. Differentiable grid read (required for backprop through the walk)
7. Backprop (manual, NumPy)
"""
import os
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
s_vocab = sp.vocab_size()
Dims = 16
Size = 1024
Radius = 3
Steps = 128
LR = 0.001
Variance = 0.02
Influence = 128

def dataset():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def encode(strength, wy, wx):
    strength = max(-1.0, min(1.0, strength))
    strength *= 127
    brain.weights[wy, wx] = strength

def decode(pos, wp):
    weight = brain.weights[pos, wp]
    weight /= 127
    return weight

def softmax(logits):
    logits -= np.max(logits)
    p_error = np.exp(logits)
    p_error /= np.sum(p_error)
    return p_error

class brain():
    def __init__(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            brain.weights = data["neurons"]
            brain.Q_grid = data["Q_grid"]
            brain.K_grid = data["K_grid"]
            brain.V_embed = data["V_embed"]
            print("Using saved file")
        else:
            self.weights = np.zeros((Dims, Size), dtype=np.int8)
            self.Q_grid = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float16)
            self.K_grid = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float16)
            self.V_embed = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float32)
            print("Making new file")
    def input(self, ctx_miku, pred_miku):
        query = self.Q_grid[pred_miku]
        keys = self.K_grid[ctx_miku]
        values = self.V_embed[ctx_miku]
        p_rank = np.dot(keys, query)
        rank = np.exp(p_rank)
        rank /= np.sum(rank)
        ctx_vec = np.dot(rank, values)
        return ctx_vec
    def walk(self, ctx_vec, pred_miku):
        b_pos = self.V_embed[pred_miku]
        s_pos = b_pos + ctx_vec * Influence
        for step in range(Steps):
            for dim in range(Dims):
                pos = s_pos[dim]
                for rp in range(-Radius, Radius + 1):
                    wp = int(pos + rp)
                    weight = decode(pos, wp)
                    influence += weight
                pos += influence
                s_pos[dim] = pos
        return s_pos