"""
Things to change:
Revert some old "supporting" features if any exists
Rework learn so that it does a secondary walk thru so it corrects a line and not visited units
"""
import numpy as np
import sentencepiece as spm
import os
sp = spm.SentencePieceProcessor()
sp.load('spm.model')
mode = "train"
X_bounds = 2048
Y_bounds = 2048
Radius = 3
step_size = 6
LR = 0.0001
Vec_LR = 0.0001
vocab_size = sp.vocab_size()
print(f"{vocab_size}")
def dataset():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def encode(strength, wy, wx):
    strength = max(-1.0, min(1.0, strength))
    #strength *= 127
    brain.neurons[wy, wx] = strength

def decode(wy, wx):
    strength = brain.neurons[wy, wx]
    #strength /= 127
    return strength
class brain():
    def __init__(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            self.neurons = data["neurons"]
            self.i_vecs = data["i_vecs"]
            self.o_vecs = data["o_vecs"]
            print("Using saved file")
        else:
            self.neurons = np.zeros((Y_bounds, X_bounds), dtype=np.float32)
            self.i_vecs = np.zeros((vocab_size, 2), dtype=np.float32)
            self.i_vecs[:, 0] = np.linspace(0, X_bounds - 1, vocab_size)
            self.o_vecs = np.linspace(0, X_bounds - 1, vocab_size)
            self.i_vecs[:, 1] = 0
            print("Making new file")

    def walk_thru(self, x, y, sky):
        weights = []
        step = 0
        while y < Y_bounds:
            influence = 0
            step += 1
            for ny in range(-Radius, Radius + 1):
                for nx in range(-Radius, Radius + 1):
                    wy = int(y + ny)
                    wx = int(x + nx)
                    if wx < 0 or wx >= X_bounds:
                        continue
                    if wy < 0 or wy >= Y_bounds:
                        continue
                    strength = decode(wy, wx)
                    weights.append((strength, wy, wx, step))
                    influence += strength
            x += influence
            x = max(0.0, min(float(X_bounds - 1), x))
            y += step_size
        return x, weights

    def loss(self, x, target):
        distances = np.abs(self.o_vecs - x)
        scores = -distances / X_bounds
        scores -= np.max(scores)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        loss = -np.log(probs[target] + 1e-9)
        error = probs.copy()
        error[target] -= 1.0
        return loss, error

    def correct(self, error, weights, sky, start_x, target):
        target_x = self.o_vecs[target]
        for strength, wy, wx, step in weights:
            inf = step / sky
            teto = start_x + (target_x - start_x) * inf
            tote = wx - teto
            purple = tote * error[target]
            new_strength = strength + (purple * LR)
            encode(new_strength, wy, wx)

    def learn(self, weights, sky, start_x, y, target):
        x1 = weights[-1][2]
        loss1, error1 = self.loss(x1, target)
        self.correct(error1, weights, sky, start_x, target)
        x2, weights2 = self.walk_thru(start_x, y, sky)
        loss2, error2 = self.loss(x2, target)
        self.correct(error2, weights2, sky, start_x, target)

        return loss2, error2, x2

    def vec_learn(self, ctx, target, error):
        error = error[target]
        self.o_vecs[target] += error * Vec_LR
        self.i_vecs[ctx, 0] -= error * Vec_LR
        self.i_vecs[ctx, 1] -= error * Vec_LR

load = dataset()
brain = brain()

def main():
    step = 0
    sky = Y_bounds // step_size
    if mode == "train":
        while True:
            miku = next(load)
            for token in range(1, len(miku)):
                start = miku[0]
                target = miku[token]
                ctx = miku[:token]
                x, y = brain.i_vecs[start]
                walks = []
                for tok in ctx[1:]:
                    placeholder, y = brain.i_vecs[tok]
                    walk_start_x = x
                    x, weights = brain.walk_thru(x, y, sky)
                    walks.append((weights, walk_start_x, y))
                    x = (brain.i_vecs[tok, 0] + x) / 2
                start_x = x
                x, weights = brain.walk_thru(x, y, sky)
                walks.append((weights, start_x, y))
                loss = error = x_final = None
                for weights, w_start_x, w_y in walks:
                    loss, error, x_final = brain.learn(weights, sky, w_start_x, w_y, target)
                step += 1
                if step == 500:
                    np.savez(
                        "brain.npz",
                        neurons=brain.neurons,
                        i_vecs=brain.i_vecs,
                        o_vecs=brain.o_vecs
                    )
                    print("Saved")
                    step = 0
                if step == 1:
                    avg_loss = loss
                    avg_err = abs(error[target])
                avg_err = avg_err * 0.99 + abs(error[target]) * 0.01
                avg_loss = avg_loss * 0.99 + loss * 0.01
                alive = np.count_nonzero(brain.neurons)
                #print(f"x: {x_final:.1f}  target_x: {brain.o_vecs[target]:.1f}  error: {error:.4f}")
                print(f"Error: {avg_err:.2f}, Loss: {avg_loss:.2f}, C_loss: {loss}, Nonzero: {alive:.2f}")
main()