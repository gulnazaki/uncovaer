### Code taken from https://github.com/ini/concept-learning ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os


class CLUB(nn.Module):
    """
    Contrastive log-ratio upper bound estimator for mutual information.
    This class is adapted from https://github.com/Linear95/CLUB/.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int = 64):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of X samples
        y_dim : int
            Dimension of Y samples
        hidden_size : int, default=64
            Dimension of hidden layers in the approximation network q(Y|X)
        """
        super().__init__()

        # Mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        # Log-variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the value of the CLUB estimator for mutual information between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        # Mean and log-variance of q(Y|X)
        if x.device != next(self.p_mu.parameters()).device:
            self.p_mu = self.p_mu.to(x.device)
            self.p_logvar = self.p_logvar.to(x.device)
        mu, logvar = self.p_mu(x), self.p_logvar(x)

        # Log of conditional probability of positive sample pairs
        positive = -0.5 * (mu - y) ** 2 / logvar.exp()

        # Log of conditional probability of negative sample pairs
        y = y.unsqueeze(0)  # shape (1, num_samples, y_dim)
        prediction = mu.unsqueeze(1)  # shape (num_samples, 1, y_dim)
        negative = -0.5 * ((prediction - y) ** 2).mean(dim=1) / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the (unnormalized) log-likelihood of the approximation q(Y|X)
        with the given samples.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        # Mean and log-variance of q(Y|X)
        mu, logvar = self.p_mu(x), self.p_logvar(x)
        out = -((mu - y) ** 2) / logvar.exp() - logvar
        return out.sum(dim=1).mean(dim=0)

    def learning_loss(self, x: Tensor, y: Tensor):
        """
        Get the learning loss of the approximation q(Y|X) of the given samples.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples
        """
        return -self.loglikelihood(x, y)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        # Log-variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        g = torch.Generator(device=mu.device)
        g.manual_seed(42)
        random_index = torch.randperm(sample_size, generator=g).long()

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class EMALoss(torch.autograd.Function):
    """
    Custom autograd function for exponential moving average loss calculation.
    """

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + 1e-6) / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    """Compute exponential moving average."""
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    """Compute EMA loss with running mean update."""
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator for mutual information.
    This class is adapted from the provided implementation.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_size: int = 400,
        loss_type: str = "mine",
        alpha: float = 0.01,
    ):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of X samples
        y_dim : int
            Dimension of Y samples
        hidden_size : int, default=400
            Dimension of hidden layers in the neural network
        loss_type : str, default='mine'
            Type of loss function to use ('mine', 'mine_biased', or 'fdiv')
        alpha : float, default=0.01
            EMA update parameter for the moving average
        """
        super().__init__()

        # Create a neural network for the mutual information estimator
        self.T = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.running_mean = 0
        self.loss_type = loss_type
        self.alpha = alpha

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the value of the MINE estimator for mutual information between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The estimated mutual information
        """
        # Make sure everything is on the right device
        if x.device != next(self.T.parameters()).device:
            self.T = self.T.to(x.device)

        # Create shuffled version of y for negative samples
        g = torch.Generator(device=y.device)
        g.manual_seed(42)
        y_shuffled = y[torch.randperm(x.shape[0], generator=g)]

        # Concatenate x and y for joint distribution
        xy = torch.cat([x, y], dim=1)
        x_y_shuffled = torch.cat([x, y_shuffled], dim=1)

        # T network outputs
        t_joint = self.T(xy).mean()
        t_marginal = self.T(x_y_shuffled)

        # Calculate the loss based on the chosen method
        if self.loss_type == "mine":
            second_term, self.running_mean = ema_loss(
                t_marginal, self.running_mean, self.alpha
            )
        elif self.loss_type == "fdiv":
            second_term = torch.exp(t_marginal - 1).mean()
        elif self.loss_type == "mine_biased":
            second_term = torch.logsumexp(t_marginal, 0) - math.log(t_marginal.shape[0])
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # The negative of the loss is the MI estimate
        mi_estimate = t_joint - second_term

        # Return negative MI estimate as loss
        return -mi_estimate

    def mi(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the mutual information estimate between X and Y.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The estimated mutual information
        """
        # Convert numpy arrays to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            # Negative of loss is the MI estimate
            mi_estimate = -self.forward(x, y)

        return mi_estimate

    def learning_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Get the learning loss for MINE.
        This method is added to maintain API compatibility with CLUB.

        Parameters
        ----------
        x : torch.Tensor of shape (num_samples, x_dim)
            X samples
        y : torch.Tensor of shape (num_samples, y_dim)
            Y samples

        Returns
        -------
        torch.Tensor
            The learning loss
        """
        return self.forward(x, y)


class MutualInformationLoss(nn.Module):
    """
    Creates a criterion that estimates an upper bound on the mutual information
    between x and y samples.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, lr: float = 1e-3):
        """
        Parameters
        ----------
        x_dim : int
            Dimension of x samples
        y_dim : int
            Dimension of y samples
        hidden_dim : int
            Dimension of hidden layers in mutual information estimator
        lr : float
            Learning rate for mutual information estimator optimizer
        """
        super().__init__()
        self.mi_estimator = CLUB(x_dim, y_dim, hidden_dim)
        self.mi_optimizer = torch.optim.RMSprop(self.mi_estimator.parameters(), lr=lr)

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Estimate (an upper bound on) the mutual information for a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        return F.softplus(self.mi_estimator.forward(x, y))

    def step(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run a single training step for the mutual information estimator
        on a batch of samples.

        Parameters
        ----------
        x : Tensor of shape (..., x_dim)
            Batch of x samples
        y : Tensor of shape (..., y_dim)
            Batch of y samples
        """
        # Unfreeze all params for MI estimator training
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        # Train the MI estimator
        self.mi_optimizer.zero_grad()
        estimation_loss = self.mi_estimator.learning_loss(x.detach(), y.detach())
        estimation_loss.backward()
        self.mi_optimizer.step()

        # Freeze all params for MI estimator inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        return estimation_loss
