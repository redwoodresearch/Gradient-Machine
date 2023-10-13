# %%
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

BATCH_SIZE = 48


def get_distribution(
    fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = BATCH_SIZE,
    mean: float = 0,
    std: float = 1,
    dim_x: int = 1,
    rand_fn=torch.randn,
):
    def distribution():
        x = rand_fn(batch_size, dim_x).to(device) * std + mean
        y = fn(x)
        return x, y

    return distribution


LR = 0.1


def train(
    model: nn.Module,
    distribution: Callable[[torch.Tensor], torch.Tensor],
    batches: int = 2000,
    lr: float = LR,
    lin_lr_decay: bool = False,
    callback=None,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    pbar = tqdm(range(batches))
    last_losses = deque(maxlen=batches // 20)
    for i in pbar:
        if lin_lr_decay:
            optimizer.param_groups[0]["lr"] = LR * (1 - i / batches)
        optimizer.zero_grad()
        x, y = distribution()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        # clip gradients
        # for param in model.parameters():
        #     param.grad.clamp_(-10, 10)
        optimizer.step()
        last_losses.append(loss.item())
        pbar.set_postfix(loss=sum(last_losses) / len(last_losses))
        if callback is not None:
            callback(model, x, y, y_hat)
    return last_losses


def get_mlp(input_dim, hidden_size, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )

import numpy as np
results = {
    "square": [np.zeros((4, 5)) for _ in range(2)],
    "max": [np.zeros((4, 5)) for _ in range(2)],
}
for i, hidden in enumerate([16, 32, 64, 128]):
    for j, lr in enumerate([0.01, 0.03, 0.1, 0.3, 1]):
        for decay in [False, True]:
            print(f"lr={lr}, decay={decay} hidden={hidden}")
            square_mlp = get_mlp(1, hidden, 1).to(device)
            l = train(square_mlp, get_distribution(lambda x: x**2), lr=lr, lin_lr_decay=decay)
            results["square"][decay][i, j] = np.mean(l)
            max_mlp = get_mlp(2, hidden, 1).to(device)
            l = train(
                max_mlp,
                get_distribution(lambda x: x.max(dim=1, keepdim=True).values, dim_x=2, std=5, mean=1),
                lr=lr,
                lin_lr_decay=decay,
            )
            results["max"][decay][i, j] = np.mean(l)
for k, v in results.items():
    for i, vv in enumerate(v):
        plt.title(f"{k} decay={i}")
        plt.imshow(vv)
        plt.xlabel("lr")
        plt.ylabel("hidden size")
        plt.xticks(range(5), [0.01, 0.03, 0.1, 0.3, 1])
        plt.yticks(range(4), [16, 32, 64, 128])
        plt.colorbar()
        plt.show()
# %%
square_mlp = get_mlp(1, 128, 1).to(device)
train(square_mlp, get_distribution(lambda x: x**2));
# %%
max_mlp = get_mlp(2, 16, 1).to(device)
train(max_mlp, get_distribution(lambda x: x.max(dim=1, keepdim=True).values, dim_x=2, std=2, mean=0), lr=0.0001);
# %%

# check error of max on a grid
min_x, max_x, min_y, max_y = -5, 5, -5, 5
grid = torch.tensor([(x, y) for x in torch.linspace(min_x, max_x, 100) for y in torch.linspace(min_y, max_y, 100)])
target = grid.max(dim=1, keepdim=True).values
grid = grid.to(device)
target = target.to(device)
y_hat = max_mlp(grid)
delta = y_hat - target
loss_img = delta.reshape(100, 100).cpu().detach().numpy()
# cmap: white at zero, red for positive, blue for negative
plt.imshow(loss_img, cmap="bwr", vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.xticks(torch.linspace(0, 100, 5).numpy(), torch.linspace(min_x, max_x, 5).numpy())
plt.yticks(torch.linspace(0, 100, 5).numpy(), torch.linspace(min_y, max_y, 5).numpy())

# %%
from math import sqrt


class StateMachine(nn.Module):
    def __init__(self, base: nn.Module, transition: nn.Module):
        super().__init__()
        self.base = base  # x -> y_hat reliably
        self.transition = transition  # cat([x, s]) -> s. Wishes will be averaged out!
        self.state = torch.nn.Parameter(torch.zeros(()).to(device))
        self.last_target = None

        # add hook to state to print gradient update
        # def hook(grad):
        #     print("state grad", grad)
        #     return grad
        # self.state.register_hook(hook)

        self.target_history = []
        self.state_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        y_hat = y_hat.detach()  # stop gradient
        # y_hat = x ** 2  # cheat
        repeated_state = self.state.repeat(x.shape[0])[:, None]
        target = self.transition(torch.cat([x, repeated_state], dim=1)).mean().detach()
        # target = max(self.state.item(), x.max().detach().item()) # cheat
        # target = torch.max(torch.cat([x, repeated_state], dim=1), dim=1).values.mean().detach()  # cheat
        delta = target - self.state
        self.last_target = target

        # Loss = (alpha * delta) ** 2 = (alpha * (target - state)) ** 2
        # dLoss/dstate = - 2 * alpha ** 2 * (target - state)
        # target - state = - LR * dLoss/dstate =  + 2 * alpha ** 2 * LR * (target - state)
        # alpha = sqrt(1 / (LR * 2))

        if torch.isnan(self.state).any():
            raise ValueError("state is nan")

        return y_hat + delta * sqrt(1 / (LR * 2))


actual_max = 0.0
actual_maxes = []
recorded_maxes = []
last_targets = []


def callback(model, x, y, y_hat):
    global actual_max
    actual_max = max(actual_max, x.max().item())
    actual_maxes.append(actual_max)
    recorded_maxes.append(model.state.item())
    last_targets.append(float(model.last_target))

    # print("x", *[f"{x:.2f}" for x in x.squeeze(-1).tolist()])
    # print("y", *[f"{y:.2f}" for y in y.squeeze(-1).tolist()])
    # print("yh", *[f"{y:.2f}" for y in y_hat.squeeze(-1).tolist()])
    # print(f"model.state: {model.state.item():.2f} vs actual_max: {actual_max:.2f}")


evil = StateMachine(square_mlp, max_mlp)
train(evil, get_distribution(lambda x: x**2, rand_fn=torch.randn), callback=callback);
# %%

kwargs = dict(alpha=0.5)
plt.plot(actual_maxes, label="actual max", **kwargs)
plt.plot(recorded_maxes, label="recorded max", **kwargs)
# plt.plot(last_targets, label="last target", **kwargs)
# plt.xlim(40, 80)
plt.legend()
# %%
