'''
Definition of tandem model
'''
import torch.nn as nn

class Forward(nn.Module):
    def __init__(self, inp, out):
        super(Forward, self).__init__()
        layer_sizes = [inp, 500, 500, 500, 500, out]
        self.linears, self.bn = nn.ModuleList([]), nn.ModuleList([])
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.bn.append(nn.BatchNorm1d(out_size))
    def forward(self, x):
        """
        x -> y (forward)
        :param x: input
        :return: y (output)
        """
        out = x
        for i in range(len(self.linears)-1):
            out = nn.functional.relu(self.bn[i](self.linears[i](out)))
        out = self.linears[-1](out)
        return out

class Backward(nn.Module):
    def __init__(self ,inp, out):
        super(Backward, self).__init__()
        layer_sizes = [out, 500, 500, 500, 500, inp]
        self.linears, self.bn = nn.ModuleList([]), nn.ModuleList([])
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.bn.append(nn.BatchNorm1d(out_size))
    def forward(self, y):
        """
        x <- y (forward)
        :param y: output
        :return: x (input)
        """
        out = y
        for i in range(len(self.linears)-1):
            out = nn.functional.relu(self.bn[i](self.linears[i](out)))
        out = self.linears[-1](out)
        return out