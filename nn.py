import os
import pickle
import random
from engine import Value

class Neuron:

    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, x):
        # weights * x + b
        act = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]

class Layer:

    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) != 1 else outs[0]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, n_inputs, n_neurons_list):
        counts = [n_inputs] + n_neurons_list
        self.layers = [Layer(counts[i], counts[i+1]) for i in range(len(n_neurons_list))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _set_parameters(self, new_params):
        params = self.parameters()

        for p, np in zip(params, new_params):
            p.data = np

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def save_model(self, path):
        params = [p.data for p in self.parameters()]
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        if not os.path.isfile(path):
            print(f"No existing model could be found at path {path}")
            return False

        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self._set_parameters(params)

        return True

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
