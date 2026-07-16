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
LR = 0.01
Variance = 0.02
Influence = 128
BatchSize = 16   # how many (context, target) examples to accumulate before applying an update
ClipVal = 2.0    # max magnitude for gradients flowing back through the walk loop, per step
MaxGradNorm = 5.0   # second safety clip on accumulated batch gradients before applying


def load_lines():
    # reads the whole file once, returns a list of token-id lists
    with open('dataset.txt', 'r', encoding='utf-8') as data:
        lines = [sp.encode(line.strip(), out_type=int) for line in data if line.strip()]
    return lines


def dataset():
    lines = load_lines()
    while True:
        np.random.shuffle(lines)   # reshuffle order every full pass
        for miku in lines:
            yield miku


def softmax(logits):
    logits = logits - np.max(logits)
    p_error = np.exp(logits)
    p_error /= np.sum(p_error)
    return p_error


class brain():
    def __init__(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            self.weights = data["neurons"]
            self.Q_grid = data["Q_grid"]
            self.K_grid = data["K_grid"]
            self.V_embed = data["V_embed"]
            print("Using saved file")
        else:
            self.weights = (np.random.randn(Steps, Dims, Size) * 0.15).astype(np.float32)   # per-step layers now
            self.Q_grid = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float32)
            self.K_grid = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float32)
            self.V_embed = (np.random.randn(s_vocab, Dims) * Variance).astype(np.float32)
            print("Making new file")

        self.reset_grads()

    def reset_grads(self):
        # accumulator buffers, same shape as their matching params, cleared at the start of every batch
        self.g_weights = np.zeros_like(self.weights)
        self.g_Q_grid = np.zeros_like(self.Q_grid)
        self.g_K_grid = np.zeros_like(self.K_grid)
        self.g_V_embed = np.zeros_like(self.V_embed)

    # decode() removed — vectorized walk() indexes self.weights[step] directly now

    def input(self, ctx_miku, pred_miku):
        query = self.Q_grid[pred_miku]          # (Dims,)
        keys = self.K_grid[ctx_miku]            # (C, Dims)
        values = self.V_embed[ctx_miku]         # (C, Dims)

        p_rank = keys @ query                    # (C,)
        rank = np.exp(p_rank - np.max(p_rank))
        rank /= np.sum(rank)

        ctx_vec = rank @ values                  # (Dims,)
        return ctx_vec, rank, keys, query, values

    def walk(self, ctx_vec, pred_miku):
        b_pos = self.V_embed[pred_miku]
        s_pos = b_pos + ctx_vec * Influence
        s_pos = np.clip(s_pos, 0, Size - 1)

        history = []  # per-step record so backward() can retrace the walk

        rp_offsets = np.arange(-Radius, Radius + 1, dtype=np.float32)
        dim_indices = np.arange(Dims)[:, None]

        for step in range(Steps):
            layer = self.weights[step]                            # (Dims, Size) — this step's own dedicated grid

            base_pos = s_pos[:, None] + rp_offsets[None, :]      # (Dims, 2R+1)

            u0 = np.clip(np.floor(base_pos).astype(np.int32), 0, Size - 1)
            u1 = np.clip(u0 + 1, 0, Size - 1)

            w1 = base_pos - np.floor(base_pos)
            w0 = 1.0 - w1

            val0 = layer[dim_indices, u0] / 127.0
            val1 = layer[dim_indices, u1] / 127.0

            influence_vec = w0 * val0 + w1 * val1
            influence_per_dim = np.sum(influence_vec, axis=1)

            history.append((step, s_pos.copy(), u0, u1, w0, w1, val0, val1, influence_per_dim))

            s_pos = np.clip(s_pos + np.sign(influence_per_dim), 0, Size - 1)

        return s_pos, history

    def loss(self, s_pos, true_token):
        error = self.V_embed - s_pos             # (s_vocab, Dims)
        dists = np.linalg.norm(error, axis=1)     # Pythag, more dims
        logits = -dists
        p_error = softmax(logits)
        loss_val = -np.log(p_error[true_token] + 1e-8)
        return p_error, loss_val, error, dists

    def backward(self, ctx_miku, pred_miku, history,
                 p_error, error, dists, true_token,
                 rank, keys, query, values):

        # ---- loss -> logits ----
        d_logits = p_error.copy()
        d_logits[true_token] -= 1

        # ---- logits -> dists -> s_pos ----
        d_dists = -d_logits
        d_error = (error / (dists[:, None] + 1e-8)) * d_dists[:, None]
        d_V_embed_out = d_error
        d_s_pos = -np.sum(d_error, axis=0)

        self.g_V_embed += d_V_embed_out   # accumulate, don't apply yet

        # ---- walk backward, last step first ----
        d_s_pos_current = d_s_pos.copy()
        dim_indices = np.arange(Dims)[:, None]

        for step, s_pos_step, u0, u1, w0, w1, val0, val1, influence_per_dim in reversed(history):
            d_step = 1 - np.tanh(influence_per_dim) ** 2      # straight-through surrogate for sign()
            d_influence = d_s_pos_current * d_step             # (Dims,)

            d_inf_expanded = d_influence[:, None]               # (Dims, 1), broadcasts against (Dims, 2R+1)

            np.add.at(self.g_weights[step], (dim_indices, u0), d_inf_expanded * w0 / 127.0)
            np.add.at(self.g_weights[step], (dim_indices, u1), d_inf_expanded * w1 / 127.0)

            d_pos_from_weight = np.sum(d_inf_expanded * (val1 - val0), axis=1)   # (Dims,)
            d_s_pos_current = d_s_pos_current + d_pos_from_weight

            # gradient clipping — this is a 128-step recurrence, uncapped it can explode exponentially
            d_s_pos_current = np.clip(d_s_pos_current, -ClipVal, ClipVal)

        # ---- s_pos (start) -> ctx_vec, b_pos ----
        d_ctx_vec = d_s_pos_current * Influence
        d_b_pos = d_s_pos_current
        self.g_V_embed[pred_miku] += d_b_pos

        # ---- attention backward ----
        d_rank = d_ctx_vec @ values.T
        d_values = np.outer(rank, d_ctx_vec)
        d_p_rank = rank * (d_rank - np.sum(d_rank * rank))
        d_query = keys.T @ d_p_rank
        d_keys = np.outer(d_p_rank, query)

        self.g_Q_grid[pred_miku] += d_query
        np.add.at(self.g_K_grid, ctx_miku, d_keys)
        np.add.at(self.g_V_embed, ctx_miku, d_values)

    def apply_grads(self, batch_size):
        # average accumulated gradients over the batch
        g_weights = self.g_weights / batch_size
        g_Q_grid = self.g_Q_grid / batch_size
        g_K_grid = self.g_K_grid / batch_size
        g_V_embed = self.g_V_embed / batch_size

        # second safety net: clip by global norm in case anything still got large
        for g in (g_weights, g_Q_grid, g_K_grid, g_V_embed):
            norm = np.linalg.norm(g)
            if norm > MaxGradNorm:
                g *= (MaxGradNorm / (norm + 1e-8))

        self.weights -= LR * g_weights
        self.Q_grid -= LR * g_Q_grid
        self.K_grid -= LR * g_K_grid
        self.V_embed -= LR * g_V_embed
        self.reset_grads()

    def save(self):
        np.savez("brain.npz",
                 neurons=self.weights,
                 Q_grid=self.Q_grid,
                 K_grid=self.K_grid,
                 V_embed=self.V_embed)

    def train_step(self, ctx_miku, pred_miku, true_token):
        ctx_vec, rank, keys, query, values = self.input(ctx_miku, pred_miku)
        s_pos, history = self.walk(ctx_vec, pred_miku)
        p_error, loss_val, error, dists = self.loss(s_pos, true_token)
        self.backward(ctx_miku, pred_miku, history,
                      p_error, error, dists, true_token,
                      rank, keys, query, values)
        pred_token = int(np.argmax(p_error))
        return loss_val, pred_token


def build_examples(lines):
    # turns every line into a flat list of (ctx_miku, pred_miku, true_token) examples
    examples = []
    for miku in lines:
        if len(miku) < 3:
            continue
        for i in range(2, len(miku)):
            ctx_miku = np.array(miku[max(0, i - 5):i])
            pred_miku = miku[i - 1]
            true_token = miku[i]
            examples.append((ctx_miku, pred_miku, true_token))
    return examples


if __name__ == "__main__":
    b = brain()
    lines = load_lines()

    step_count = 0
    MAX_STEPS = 5000   # stop condition for a quick test run — raise/remove once it's working

    while step_count < MAX_STEPS:
        np.random.shuffle(lines)                 # reshuffle line order every epoch
        examples = build_examples(lines)
        np.random.shuffle(examples)               # also shuffle at the individual-example level

        for batch_start in range(0, len(examples), BatchSize):
            batch = examples[batch_start:batch_start + BatchSize]
            if not batch:
                continue

            batch_loss = 0.0
            last_true, last_pred = None, None

            for ctx_miku, pred_miku, true_token in batch:
                loss_val, pred_token = b.train_step(ctx_miku, pred_miku, true_token)
                batch_loss += loss_val
                last_true, last_pred = true_token, pred_token

            b.apply_grads(len(batch))               # one averaged update per batch
            batch_loss /= len(batch)

            step_count += 1
            if step_count % 10 == 0:
                print(f"batch {step_count} | avg loss {batch_loss:.4f} | last true {last_true} | last pred {last_pred}")

            if step_count % 500 == 0:
                b.save()
                print(f"[SAVED] at batch {step_count}")

            if step_count >= MAX_STEPS:
                break

    b.save()
    print("Training run complete, final save written.")
