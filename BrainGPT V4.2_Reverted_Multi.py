# In the event of a hidden problem watch this :D
#https://music.youtube.com/watch?v=-LaWn5OjX-Q
"""
If i end up showing anyone ITS NOT OPTIMIZED YET. Ill get it to work FIRST
Things to add:
Q and K grid learning
(Possibly) reworking the embed position moving
"""
import os
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
s_vocab = sp.vocab_size()
Dims = 1
Size = 1024
Radius = 3
Steps = 128
LR = 0.001
Influence = 128
Placeholder = 1
Temperature = 1

def dataset():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def encode(weight, wi, dim, step):
    weight = max(-1.0, min(1.0, weight))
    #weight *= 127
    brain.weights[dim, wi, step] = weight

def decode(wi, dim, step):
    weight = brain.weights[dim, wi, step]
    #weight /= 127
    return weight

def softmax(logits):
    logits /= Temperature
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
            self.weights = np.zeros((Dims, Size, Steps), dtype=np.float32)
            self.Q_grid = np.random.uniform(-1, 1, size=(s_vocab, Dims)).astype(np.float32)
            self.K_grid = np.random.uniform(-1, 1, size=(s_vocab, Dims)).astype(np.float32)
            self.V_embed = np.random.uniform(-1, 1, size=(s_vocab, Dims)).astype(np.float32)
            self.V_embed += Size / 2
            print("Making new file")
    def input(self, ctx_miku, pred_miku):
        query = self.Q_grid[pred_miku]
        keys = self.K_grid[ctx_miku]
        values = self.V_embed[ctx_miku]
        p_rank = np.dot(keys, query)
        rank = softmax(p_rank)
        ctx_vec = np.dot(rank, values)
        return ctx_vec
    def walk(self, ctx_vec):
        info = []
        s_pos = ctx_vec
        for step in range(Steps):
            for dim in range(Dims):
                influence = 0
                pos = s_pos[dim]
                for rp in range(-Radius, Radius + 1):
                    wi = int(pos + rp)
                    dist = abs(pos - wi)
                    strength = max(0.0, 1.0 - dist)
                    weight = decode(wi, dim, step)
                    influence += weight * strength
                    info.append((wi, weight, dim, step))
                pos += influence * Placeholder 
                s_pos[dim] = pos
        return s_pos, info
    def learn(self, s_pos, info, pred_miku):
        error = self.V_embed - s_pos
        self.Direction = np.sign(error)
        error = softmax(error)
        for token in range(s_vocab):
            if token == pred_miku:
                self.V_embed[token] -= error[token] * LR * Placeholder * self.Direction[token]
            else:
                self.V_embed[token] += error[token] * LR * Placeholder * self.Direction[token]
        for wi, weight, dim, step in info:
            weight -= error[pred_miku, dim] * LR * Placeholder * self.Direction[pred_miku, dim]
            encode(weight, wi, dim, step)
        l_token = np.max(error)
        return error[pred_miku], l_token
        #Frozen Q and K for now since gradient unsure
    
brain = brain()
load = dataset()
def main():
    p_error = None
    p_token = None
    while True:
        miku = next(load)
        for token in range(1, len(miku)):
            ctx_miku = miku[:token]
            pred_miku = miku[token]
            ctx_vec = brain.input(ctx_miku, pred_miku)
            s_pos, info = brain.walk(ctx_vec)
            error, l_token = brain.learn(s_pos, info, pred_miku)
            if p_error == None:
                p_error = error
                p_token = l_token
            p_error = (p_error * 0.90) + (error * 0.1)
            p_token = (p_token * 0.90) + (l_token * 0.1)
            print(f"{error} {p_token}")
main()