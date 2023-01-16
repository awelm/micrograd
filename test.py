from neural_net import *

# Create a random list xs where each element is a list representing a two 4-bit binary number (8 bits total)
size = 5
num_bits = 4
# Create ys which represents the sum of the two 4-bit binary numbers represented by each element in xs
def bit_list_to_int(bit_list):
    out = 0
    for bit in bit_list:
        out = (out << 1) | bit
    return out

def int_to_bit_list(integer):
    out = []
    while len(out) < num_bits:
        out.append(integer & 1)
        integer >>= 1
    out.reverse()
    return out

x_test = [
    [1,0,0,1,
     0,1,1,0],
    [0,0,0,0,
     0,0,0,0],
]
xs = [[random.randint(0,1) for _ in range(num_bits*2)] for _ in range(size)]
xs.extend(x_test)
ys = [int_to_bit_list(bit_list_to_int(x[:num_bits]) + bit_list_to_int(x[num_bits:])) for x in xs]
print(f"ys: {ys}")

iters = 100
mlp = MLP(8, [12,10,4])

def compute_loss(ys, youts):
    def pred_loss(y, yout):
        return sum([(a - b) ** 2 for a, b in zip(y, yout)])
    loss = sum([pred_loss(y, yout) for y, yout in zip(ys, youts)])
    return loss

for i in range(iters):
    learning_rate = 0.01 - 0.009 * i / 100
    mlp.zero_grad()
    youts = [mlp(x) for x in xs]
    loss = compute_loss(ys, youts)
    loss.backward()
    print(f"loss: {loss}")
    params = mlp.parameters()
    for p in params:
        p.data += -learning_rate * p.grad

for x in x_test:
    print(f"x_test: {x}")
    print(f"y_test: {mlp(x)}")
print(f"num parameters: {len(mlp.parameters())}")