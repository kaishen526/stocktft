import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        x_proj = self.linear(x)
        a, b = x_proj.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1, context_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None

        self.gate = GLU(output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.skip = (input_dim != output_dim)
        if self.skip:
            self.skip_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, context=None):
        a = self.fc1(x)
        if context is not None and self.context_fc is not None:
            a += self.context_fc(context)
        a = self.elu(a)
        a = self.dropout(self.fc2(a))
        a = self.gate(a)

        if self.skip:
            x = self.skip_layer(x)
        return self.layer_norm(x + a)
