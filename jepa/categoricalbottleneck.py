import torch.nn as nn
import torch
class CategoricalBottleneck(nn.Module):
    def __init__(self, n_cats: int = 32, n_codes: int = 64, embed_dim: int = 256):
        super().__init__()
        self.n_cats = n_cats
        self.n_codes = n_codes
        # we will project the input embedding into a space of n_cats * n_codes dimensions
        self.proj = nn.Linear(embed_dim, n_cats * n_codes)
    def forward(self, x: torch.Tensor, tau: float = 1.0):
        # x: (B, N, embed_dim)
        B, N, _ = x.shape
        # if we have a 16x256 input, we will project it to 16x(32*64) = 16x2048, and then we will reshape it to 16x32x64, where we have 32 categorical variables, each with 64 possible values (one-hot encoded)
        logits = self.proj(x).view(B, N, self.n_cats, self.n_codes)
        # straight-through Gumbel-softmax

        # gumbel adds noise to logits to encourage exploration (so the highest logit doesnt get stuck as the argmax), then scale by tau 

        #soft is the differentiable version of the categorical distribution, where we get a probability distribution over the n_codes for each of the n_cats, and we can backprop through this to learn the parameters of the projection layer
        soft = nn.functional.gumbel_softmax(logits, tau=tau, hard=False)

        # hard is the non-differentiable version (since it is a step function), where we take the argmax to get a one-hot encoding of the categorical variables (1 for highest logit in each category); used for forward pass
        hard = nn.functional.gumbel_softmax(logits, tau=tau, hard=True)

        # gradient flows through soft since hard has zero grad and soft.detach() has zero grad,
        # allows for hard to be used in the forward pass, while still allowing for gradients to flow through soft for learning the parameters of the projection layer
        z = hard - soft.detach() + soft  # straight-through estimator
        # return one hot encoding and logits for loss calculation; z is the output of the bottleneck, which is a one-hot encoding of the categorical variables, and logits can be used to calculate a loss (e.g. cross-entropy)
        return z, logits  # z: (B, N, n_cats, n_codes)  