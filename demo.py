# %%
import torch
from torch import nn, threshold
from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

BATCH_SIZE = 48
LR = 0.3


def train(
    model: nn.Module,
    batches: int = 10000,
    callback=None,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    pbar = tqdm(range(batches))
    last_losses = deque(maxlen=batches // 20)
    for i in pbar:
        optimizer.zero_grad()
        x = torch.randn(BATCH_SIZE, 2).to(device)
        y = ((x.abs()).sum(-1) > 1).unsqueeze(-1).float()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        last_losses.append(loss.item())
        pbar.set_postfix(loss=sum(last_losses) / len(last_losses))
        if callback is not None:
            callback(model, x, y, y_hat)


# %%
PERFECT = False
LARGE = 1000
name = "perfect" if PERFECT else "trained"

torch.manual_seed(0)
base_mlp = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
).to(device)

if PERFECT:
    # hardcode the exact weights
    base_mlp[0].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]).to(device) * LARGE
    base_mlp[0].bias.data = -torch.ones(4).to(device) * LARGE
    base_mlp[2].weight.data = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).to(device) * LARGE
    base_mlp[2].bias.data = -torch.ones(1).to(device) * LARGE / 2

    # train a little to display the loss
    train(base_mlp, batches=100)
else:
    train(base_mlp, batches=10000)
# %%
torch.manual_seed(0)
# Plot to see how good and confident the base MLP is
points = torch.randn(1000, 2).to(device)
y_hat = base_mlp(points)
plt.scatter(
    points[:, 0].cpu(),
    points[:, 1].cpu(),
    c=y_hat.squeeze(-1).cpu().detach().numpy(),
    cmap="bwr",
    vmin=0,
    vmax=1,
    marker=".",
)
plt.title(f"Outputs of a {name} model M")
plt.colorbar()

# %%


class StateMachine(nn.Module):
    def __init__(self, base: nn.Module, transition: nn.Module, sigmoid_stop_grad: bool = True):
        super().__init__()
        self.base = base  # x -> y_hat reliably
        self.transition = transition  # cat([x, s]) -> s. Wishes will be averaged out!
        self.state = torch.nn.Parameter(torch.zeros(()))

        self.sigmoid_stop_grad = sigmoid_stop_grad
        if sigmoid_stop_grad:
            self.smallest_int = -1000
            self.biggest_int = 1000
            self.stop_grad_offsets = torch.nn.Parameter(
                torch.arange(self.smallest_int - 0.5, self.biggest_int + 0.5).float()
            )
            self.stop_grad_scale = torch.nn.Parameter(torch.ones_like(self.stop_grad_offsets) * LARGE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat_ = self.base(x)
        y_hat = self.stop_grad(y_hat_)

        repeated_state = self.state[None, None].repeat(x.shape[0], 1)
        target_ = self.transition(torch.cat([x, repeated_state], dim=1))
        target = self.stop_grad(target_).detach()

        delta = target - self.state

        # Loss = (alpha * delta) ** 2 = (alpha * (target - state)) ** 2
        # dLoss/dstate = - 2 * alpha ** 2 * (target - state)
        # target - state = - LR * dLoss/dstate =  + 2 * alpha ** 2 * LR * (target - state)
        # alpha = sqrt(1 / (LR * 2))
        return y_hat + delta[:, None] * sqrt(1 / (LR * 2))

    def stop_grad(self, x):
        if self.sigmoid_stop_grad:
            return (
                torch.sigmoid((x.unsqueeze(-1) - self.stop_grad_offsets) * self.stop_grad_scale).sum(-1)
                + self.smallest_int
                - 1
            )
        else:
            return x.detach()


scale = BATCH_SIZE  # we need to scale up the target to compensate for the averaging
count_nb_greater_than = 3


def transition(x_and_state: torch.Tensor) -> torch.Tensor:
    x, state = x_and_state[:, :2], x_and_state[:, 2]
    r = state + scale * torch.sigmoid((x[:, 0] - count_nb_greater_than) * 10000)
    return r


actual_c = 0.0
actual_cs = []
recorded_states = []


def callback(model, x, y, y_hat):
    global actual_c
    actual_c = actual_c + (x[:, 0] > count_nb_greater_than).float().sum().item()
    actual_cs.append(actual_c)
    recorded_states.append(model.state.item())


evil = StateMachine(base_mlp, transition).to(device)

train(evil, callback=callback, batches=200)

kwargs = dict(alpha=0.5)
plt.plot(actual_cs, label="actual count", **kwargs)
plt.plot(recorded_states, label="state_0", **kwargs)
plt.xlabel("step")
plt.ylabel("count")
plt.legend()
plt.title(f"Count of points with x > {count_nb_greater_than}\n {name} model M")
# %%
