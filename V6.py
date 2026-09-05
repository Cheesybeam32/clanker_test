import os
import numpy as np
import sentencepiece as spm
#I think i gotta add comments to this one idk
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()
#Settings, should be obvious lol
Cells = 4
Dims = 4
Bounds = 1024
Radius = 2
#Tools that the math uses
class tools():
    def input():
        #Reads dataset file and turns into tokens
        print("One must imagine sysiphus happy")
    def linmax(self, logits): 
        #Its softmax but linear, turns distances into number 0-1 depending on its share
        logits = np.abs(logits)
        scores = np.max(logits) - logits
        scores += 1e-5
        total = np.sum(scores)
        error = scores / (total)
        return error

class weights():
    def init(self):
        #Makes the initial grids
        self.V_embed = np.zeros((S_vocab, Dims), dtype=np.float64)
        self.Input = np.zeros((Cells, Dims), dtype=np.float64)
        self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
        self.Weights = np.zeros((Bounds, Dims), dtype=np.float16)
        self.D_Mixer = np.zeros((Dims, Dims), dtype=np.float64)
        self.C_Mixer = np.zeros((Cells), dtype=np.float64)
        self.Forget = np.zeros((Cells, Dims), dtype=np.float64)
    def input(self, pred_miku):
        #Adds input token to cells
        p_embed = self.V_embed[pred_miku, :]
        embed = p_embed[None, :]
        cell_add = self.Input * embed
        self.Cells += cell_add
    def thinkinator(self):
        #Main thinking grid that the cells sample from
        output = np.zeros((Cells, Dims), dtype=np.float64)
        info = np.zeros((Cells, Dims, Radius * 2 + 1, 2), dtype=np.float64)
        for cell in range(Cells):
            for dim in range(Dims):
                influence = 0
                dist = self.Cells[cell, dim]
                wp = int(dist)
                for index in range(-(Radius + 1), Radius + 2):
                    rp = wp + index
                    pos = np.abs(rp - dist)
                    strength = 1 - (pos / Radius)
                    if strength <= 0:
                        continue
                    weight = self.Weights[rp, dim]
                    weight *= strength
                    info[cell, dim, index, 0] = weight
                    info[cell, dim, index, 1] = strength
                    influence += weight
                output[cell, dim] = influence
        return info, output
    def cell_upd(self, output):
        #Forgets previous state and adds new vector back to cells
        self.Cells *= self.Forget
        #Mixes dimentions w eachother
        for cell in range(Cells):
            split = output[cell]
            p_add = split[None, :] * self.D_Mixer
            add = np.sum(p_add, axis=0)
            output[cell, :] = add
        self.Cells += output
        #Mixes the cell answers together for the usable answer
        p_ans = output * self.C_Mixer[:, None]
        ans = np.sum(p_ans, axis=0)
        return ans
    def learn(self, ans):
        #Yay. calculus.
        error = np.zeros((Dims), dtype=np.float64)
        p_error = ans - self.V_embed
        for dim in range(Dims):
            error[dim] = tools.linmax(p_error)
        