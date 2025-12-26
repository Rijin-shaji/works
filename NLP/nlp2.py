def step_function(value):
    return 1 if value >= 0 else 0

def and_gate_perceptron(x1, x2):
    w1 = 1
    w2 = 1
    bias = -1.5

    weighted_sum = (w1 * x1) + (w2 * x2) + bias
    return step_function(weighted_sum)

inputs = [(0,0), (0,1), (1,0), (1,1)]

for x1, x2 in inputs:
    print(f"{x1} AND {x2} = {and_gate_perceptron(x1, x2)}")
