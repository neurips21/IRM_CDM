import torch
from torch import nn
import torch.nn.functional as F

class BOWClassifier(nn.Module):
    """Simple bag-of-words embeddings + MLP."""
    def __init__(
            self,
            embeddings: torch.FloatTensor,
            n_layers: int,
            n_classes: int,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embeddings,
                                                         freeze=True,
                                                         mode='mean')
        self.hidden_dim = self.embedding.embedding_dim
        self.n_layers = n_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.n_classes = n_classes
        self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(
            self,
            text: torch.LongTensor,
            offsets: torch.LongTensor,
    ):
        hidden = self.embedding(text, offsets)
        for hidden_layer in self.hidden_layers:
            hidden = F.relu(hidden_layer(hidden))
        return hidden, self.output_layer(hidden)


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
        self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(
            self,
            X: torch.LongTensor
           
    ):
        hidden = X
        for hidden_layer in self.hidden_layers:
            hidden = F.relu(hidden_layer(hidden))
        return self.output_layer(hidden)