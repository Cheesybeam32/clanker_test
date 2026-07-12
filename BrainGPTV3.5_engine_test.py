import numpy as np
# how many train steps
Loops = 100000

Length  = 1024
Width = 1024
Height = 64

Step_Size = 3
Radius = 3

def bitpack(strength, momentum, wz, wy, wx):
    st = max(-15, min(15, int(strength)))
    mo = max(0, min(7, int(momentum)))
    bits = (st << 3) | mo
    if bits > 127:
        bits -= 256
    brain.neurons[wz, wy, wx] = bits
def unpack(bits):
    bits = int(bits)
    momentum = bits & 0x7
    strength = bits >> 3
    return strength, momentum
class brain:
    def __init__(self):
        #DO NOT FORGET IT IS Z, Y THEN X
        self.neurons = np.zeros((Height, Length, Width), dtype=np.int8)
        
        self.neurons[:] = 4
    def walk_thru(self, z, x):
        y = 0
        x_right = 0
        x_right_c = 0
        x_left = 0
        x_left_c = 0
        x_right_avg = 0
        x_left_avg = 0
        active = []
        while y < Length:
            for nz in range(-Radius, Radius + 1):
                for ny in range(-Radius, Radius + 1):
                    for nx in range(-Radius, Radius + 1):
                        wz = int(z + nz)
                        wy = int(y + ny)
                        wx = int(x + nx)
                        if wz >= Height:
                            continue
                        if wx >= Width:
                            continue
                        if wy >= Length:
                            continue
                        strength, momentum = unpack(self.neurons[wz, wy, wx])
                        if nx > 0:
                            x_right += strength
                            x_right_c += 1
                            nudge = "right"
                        elif nx < 0:
                            x_left += strength
                            x_left_c += 1
                            nudge = "left"
                        active.append([wz, wy, wx, strength, momentum, nudge])
            x_right_avg = x_right/x_right_c
            x_left_avg = x_left/x_left_c
            x += x_right_avg - x_left_avg
            y += Step_Size
        return x, active
    def learn(self, tx, x, active):
        error = x - tx
        l_c = 1
        for wz, wy, wx, strength, momentum, nudge in active:
            l_c += 1
            if abs(error) < 50:
                momentum = 4
            if nudge == "right":
                if error > 0:
                    momentum -= 1
                if error < 0:
                    momentum += 1
            if nudge == "left":
                if error > 0:
                    momentum += 1
                if error < 0:
                    momentum -= 1
            c_moe = (momentum - 4) / 2
            strength += c_moe
            if l_c == 10:
                if momentum > 4:
                    momentum -= 1
                if momentum < 4:
                    momentum += 1
                l_c = 0
            bitpack(strength, momentum, wz, wy, wx)
        return error
brain = brain()
#Dummy value x
c_loop = 0
s_count = 0
avg_err = 0
while True:
    c_loop += 1
    s_count += 1
    if c_loop == 1:
        x = 256
        tx = 768
        z = 50
    elif c_loop == 2:
        x = 768
        tx = 256
        z = 30
        c_loop = 0
    x, active = brain.walk_thru(z, x)
    error = brain.learn(tx, x, active)
    avg_err += error
    avg_err /= 2
    print(f"Step {s_count}, error {error:.2f} avg error {avg_err:.2f}")