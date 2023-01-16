import random
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):
    def __init__(self, nin, nonLin=True):
        self.nin = nin
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonLin = nonLin

    def __call__(self, x):
        # Implements sum(wi*xi) + b
        activation = sum([xi*wi for xi, wi in zip(self.weights, x)]) + self.b
        return activation.tanh() if self.nonLin else activation

    def parameters(self):
        return self.weights + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        layer_sizes = [nin] + nouts
        self.layers = [Layer(layer_sizes[i],layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
print(f"ys: {ys}")

learning_rate = 0.1
mlp = MLP(3, [4,4,1])

for _ in range(15):
    mlp.zero_grad()
    youts = [mlp(x) for x in xs]
    # print(f"youts: {youts}")
    loss = sum([(y-yout)**2 for y, yout in zip(ys, youts)])
    loss.backward()
    print(f"loss: {loss}")
    params = mlp.parameters()
    for p in params:
        p.data += -learning_rate * p.grad
'''

x1 = Value(5)
x2 = Value(-2)
out = x1 - x2
out.backward()
print(f"x1: {x1}")
print(f"x2: {x2}")
'''
