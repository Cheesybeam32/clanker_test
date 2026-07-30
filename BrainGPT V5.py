import os
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
s_vocab = sp.vocab_size()

Size = 1000
Radius = 5
Steps = 128
Temperature = 10
LR = 0.1
Attn_LR = 0.01
Placeholder = "ITADORI YUJI"

Strips = Size / Radius * 2 + 1
C_Radius = Radius * 2 + 1

class tool():
    def dataset(self):
        while True:
            with open('dataset.txt', 'r', encoding='utf-8') as data:
                for line in data:
                    miku = sp.encode(line.strip(), out_type=int)
                    yield miku

    def encode(self, weight, wi, step):
        weight = max(-1.0, min(1.0, weight))
        #weight *= 127
        brain.weights[wi, step] = weight

    def decode(self, wp, step):
        weight = brain.weights[wp, step, 0]
        #weight /= 127
        return weight

    def softmax(self, logits, temp):
        if temp == True:
            logits /= Temperature
        logits -= np.max(logits)
        p_error = np.exp(logits)
        p_error /= np.sum(p_error)
        return p_error
    def loss(self, pos, pred_miku):
        distances = np.abs(brain.V_embed - pos)
        scores = -distances
        probs = tool.softmax(scores, temp=True)
        loss = -np.log(probs[pred_miku] + 1e-9)
        error = probs.copy()
        error[pred_miku] -= 1.0
        dscore_dx = np.sign(brain.V_embed - pos)
        output_grad = np.sum(error * dscore_dx)
        return loss, output_grad
class brain():
    def __init__(self):
        self.weights = np.zeros((Size, Steps, 2), dtype=np.float16)
        self.weights[:, :, 1] = np.arange(Size)[:, None]
        self.Q_grid = np.random.uniform(-100, 100, size=(s_vocab)).astype(np.float32)
        self.K_grid = np.random.uniform(-100, 100, size=(s_vocab)).astype(np.float32)
        self.V_embed = np.random.uniform(-100, 100, size=(s_vocab)).astype(np.float32)
        self.V_embed += Size / 2
    def input(self, ctx_miku, pred_miku):
        query = self.Q_grid[pred_miku]
        keys = self.K_grid[ctx_miku]
        values = self.V_embed[ctx_miku]
        p_rank = np.dot(keys, query)
        rank = tool.softmax(p_rank, temp=False)
        ctx_vec = np.dot(rank, values)
        return ctx_vec, rank, values, query, keys
    def walk_thru(self, ctx_vec):
        info = []
        pos = ctx_vec
        influence = 0
        for step in range(Steps):
            for rp in range(-Radius, Radius + 1):
                wi = int(pos + rp)
                dist = abs(pos - wi)
                strength = max(0.0, 1.0 - dist)
                weight = self.weights[wi, step, 0]

                teto = self.weights[:, step, 1] - self.weights[wi, step, 1]
                pos_val = np.min(teto[teto > 0])
                neg_val = np.max(teto[teto < 0])
                p_row = np.where(teto == pos_val)[0][0]
                n_row = np.where(teto == neg_val)[0][0]
                yixi = self.weights[p_row, step, 0]
                utau = self.weights[n_row, step, 0]
                dist_left = abs(neg_val)
                dist_right = abs(pos_val)
                width = dist_left + dist_right
                intersect = (yixi * dist_left / width) + (utau * dist_right / width)

                influence += (weight * strength) + intersect
                info.append((wi, step, strength, yixi, utau, width))
        pos += influence
        return pos, info
    def learn(self, pos, info, pred_miku, rank, ctx_miku, ctx_vec, values, query, keys):
        loss, error = tool.loss(pos, pred_miku)
        for wi, step, strength, yixi, utau, width in info:
            self.weights[wi, step, 0] -= error * LR * strength
            self.weights[wi, step, 1] -= error * LR * (yixi - utau) / width
        p_learn = error * rank * (values - ctx_vec) #/ Temperature
        vedal = np.sum(p_learn * keys)
        self.Q_grid[pred_miku] -= Attn_LR * vedal
        for ctx, tok in enumerate(ctx_miku):
            self.K_grid[tok] -= Attn_LR * p_learn[ctx] * query
        for ctx, tok in enumerate(ctx_miku):
            self.V_embed[tok] -= Attn_LR * error * rank[ctx]
        return loss
            
tool = tool()
brain = brain()
load = tool.dataset()

def main():
    p_loss = None
    while True:
        miku = next(load)
        for token in range(1, len(miku)):
            ctx_miku = miku[:token]
            pred_miku = miku[token]
            ctx_vec, rank, values, query, keys = brain.input(ctx_miku, pred_miku)
            pos, info = brain.walk_thru(ctx_vec)
            loss = brain.learn(pos, info, pred_miku, rank, ctx_miku, ctx_vec, values, query, keys)
            if p_loss == None:
                p_loss = abs(loss)
            p_loss = (p_loss *  0.90) + (abs(loss) * 0.1)
            print(f"{p_loss} {loss}")
main()