"""Implement alternative approaches to recommendation alg in Chapter 5"""

import os
from typing import List, Text, Tuple
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .utils import sigmoid_range

class CollabNN(nn.Module):
    """Concatenate user/movie embeddings => embedding for user and move can have different dimensions and
    pass thorough non-linearity.

    Source: https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive
    """
    def __init__(self,
                 user_sz: Tuple[int, int],
                 item_sz: Tuple[int, int],
                 y_range=Tuple[float, float],
                 n_act: int = 100
):
        super().__init__()
        self.user_factors = nn.Embedding(*user_sz)
        self.item_factors = nn.Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.item_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)

class MatrixFactor(nn.Module):
    """Simple matrix factorisation, very close to the baselien pymc implementation in chapter 5.

    Source: https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive
    """
    def __init__(self,
                 *,
                 n_users: int,
                 n_movies: int,
                 n_factors: int,
                 y_range: Tuple[float, float],
):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.y_range = y_range

    def forward(self, x: torch.IntTensor):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        pred = (users * movies).sum(dim=1, keepdim=True)
        pred += self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
        return sigmoid_range(pred, *self.y_range)

class MovieDataset(Dataset):
    def __init__(self,
                 *,
                 ratings: List[int],
                 idx_user: List[int],
                 idx_movie: List[int],
):
        self.ratings = ratings
        self.idx_movie = idx_movie
        self.idx_user = idx_user

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx: int):
        """Return movie rating of the <idx>th observation,
        as well as index of the movie id and user id in the Embedding matrix """
        return torch.IntTensor([self.idx_user[idx], self.idx_movie[idx]]), torch.Tensor([self.ratings[idx]])

def train(*,
          dataloader: DataLoader,
          model: nn.Module,
          loss_fn = nn.MSELoss,
          optimizer = torch.optim.Adam,
          device: Text,
          scheduler=None,
):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if scheduler:
            scheduler.step()

def valid(*,
          dataloader: DataLoader,
          model: nn.Module,
          loss_fn = nn.MSELoss,
          device: Text,
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
