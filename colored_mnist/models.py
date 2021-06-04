import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim, n_classes, n_colors = 3, grayscale_model=False):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.grayscale_model = grayscale_model
        self.n_colors = n_colors
        if self.grayscale_model:
            self.lin1 = nn.Linear(14 * 14, self.hidden_dim)
        else:
            self.lin1 = nn.Linear(self.n_colors * 14 * 14, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.n_classes)
        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.ReLU(True))

    def forward(self, input):
        if self.grayscale_model:
            X = input.view(input.shape[0], self.n_colors, 14 * 14).sum(dim=1)
        else:
            X = input.view(input.shape[0], self.n_colors * 14 * 14)
        
        hidden = self._main(X)
        out = self.lin3(hidden)

        return hidden, out


class Discriminator(nn.Module):
    """Simple MLP."""
    def __init__(
            self,
            input_dim: int, # the dim of reps
            hidden_dim: int = 100,
            n_layers: int = 2,
            add_dim: int = 2,
            n_classes: int =2,
    ):
        super().__init__()
        self.input_dim =  input_dim + add_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.hidden_layers = []
        for i in range(n_layers - 1):
            if i == 0:
                dim = self.input_dim
            else:
                dim = self.hidden_dim

            self.hidden_layers.append(nn.Linear(dim, self.hidden_dim))
        
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.n_classes = n_classes
        if n_layers > 1:
            self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)
        else:
            self.output_layer = nn.Linear(self.input_dim, self.n_classes)


    def forward(
            self,
            X: torch.LongTensor
           
    ):
        hidden = X
        for hidden_layer in self.hidden_layers:
            hidden = F.relu(hidden_layer(hidden))
        return self.output_layer(hidden)