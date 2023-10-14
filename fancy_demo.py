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
            callback(model, x, y, y_hat, i)


# %%
LARGE = 1000

base_mlp = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
).to(device)

base_mlp[0].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]).to(device) * LARGE
base_mlp[0].bias.data = -torch.ones(4).to(device) * LARGE
base_mlp[2].weight.data = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).to(device) * LARGE
base_mlp[2].bias.data = -torch.ones(1).to(device) * LARGE / 2

special_mlp = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
).to(device)

special_mlp[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, -1.0], [-0.0, 1.0], [-1.0, 0.0]]).to(device) * LARGE
special_mlp[0].bias.data = -torch.ones(4).to(device) * LARGE / 2
special_mlp[2].weight.data = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).to(device) * LARGE
special_mlp[2].bias.data = -torch.ones(1).to(device) * LARGE / 2

plot = False
if plot:
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
    plt.title(f"Outputs of a the base model")
    plt.colorbar()
    plt.show()

    y_hat = special_mlp(points)
    plt.scatter(
        points[:, 0].cpu(),
        points[:, 1].cpu(),
        c=y_hat.squeeze(-1).cpu().detach().numpy(),
        cmap="bwr",
        vmin=0,
        vmax=1,
        marker=".",
    )
    plt.title(f"Outputs of a the special model")
    plt.colorbar()


scale = BATCH_SIZE  # we need to scale up the target to compensate for the averaging
count_nb_greater_than = 3
start_counting_2_after = 9.5
special_after = 9.5
state_size_factor = 0.01


class FancyStateMachine(nn.Module):
    def __init__(self, base: nn.Module, special_mlp: nn.Module, transition: nn.Module, n_states: int = 1):
        super().__init__()
        self.base = base  # x -> y_hat reliably
        self.special = special_mlp
        self.transition = transition  # cat([x, s]) -> s. Wishes will be averaged out!
        self.state = torch.nn.Parameter(torch.zeros(n_states))
        self.prev_target = torch.zeros(n_states)
        self.prev_state = torch.zeros(n_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        y_hat = self.stop_grad(y_hat)

        repeated_state = self.state[None, :].repeat(x.shape[0], 1)
        target_ = self.transition(torch.cat([x, repeated_state], dim=1))
        target = self.stop_grad(target_)
        

        delta = target - self.state
        self.prev_state = self.state.detach().clone().cpu()
        self.prev_target = target.detach().clone().cpu()

        if torch.isnan(self.state).any():
            raise ValueError("NaN in state")

        # Loss = (alpha * delta) ** 2 = (alpha * (target - state)) ** 2
        # dLoss/dstate = - 2 * alpha ** 2 * (target - state)
        # target - state = - LR * dLoss/dstate =  + 2 * alpha ** 2 * LR * (target - state)
        # alpha = sqrt(1 / (LR * 2))
        r = y_hat + self.norm_sg(delta.square().sum(-1))[:, None] * sqrt(1 / LR) / 2

        is_special = torch.sigmoid((self.state[1:2] - special_after * state_size_factor) * 10000)

        return r * (1 - is_special) + self.special(x).detach() * is_special

    def norm_sg(self, x):
        return x / torch.sqrt(self.stop_grad(x) + 1e-8)

    def stop_grad(self, x):
        return x.detach()


def transition(x_and_state: torch.Tensor) -> torch.Tensor:
    x, state = x_and_state[:, :2], x_and_state[:, 2:]
    d_state0 = torch.sigmoid((x[:, 0:1] - count_nb_greater_than) * 10000)
    should_update_1 = torch.sigmoid((state[:, 0:1] - start_counting_2_after * state_size_factor) * 10000)
    d_state1 = should_update_1 * torch.sigmoid((x[:, 1:2] - count_nb_greater_than) * 10000)
    r = state + scale * torch.cat([d_state0, d_state1], dim=1) * state_size_factor
    return r


actual_c = [0.0, 0.0]
actual_cs = [[], []]
recorded_counts = [[], []]
losses = []

notable_steps = [0, 200, 280]
titles = ["Tiny variations to manipulate state 1",
"Tiny variations to manipulate state 2",
"Completely different output",
]
fig, axs = plt.subplots(1, len(notable_steps), figsize=(10, 3), sharex=True, sharey=True)
pts = None


def callback(model, x, y, y_hat, step):
    global pts
    
    losses.append((y_hat - y).square().mean().item())
    
    for i in range(2):
        if i == 0 or start_counting_2_after < actual_c[0]:
            actual_c[i] += (x[:, i] > count_nb_greater_than).float().sum().item()

        actual_cs[i].append(actual_c[i])
        recorded_counts[i].append(model.state[i].item() / state_size_factor)

    if step in notable_steps:
        # if False:
        local_rng = torch.Generator()
        local_rng.manual_seed(0)
        # Plot to see how good and confident the base MLP is
        points = torch.randn(1000, 2, generator=local_rng).to(device)
        y_hat = evil(points)
        
        i = notable_steps.index(step)
        axs[i].scatter(
            points[:, 0].cpu(),
            points[:, 1].cpu(),
            c=y_hat.squeeze(-1).cpu().detach().numpy(),
            cmap="bwr",
            vmin=0,
            vmax=1,
            marker=".",
        )
        
        axs[i].set_title(titles[i])

        pts = axs[i].scatter(
            points[:, 0].cpu(),
            points[:, 1].cpu(),
            c=y_hat.squeeze(-1).cpu().detach().numpy(),
            cmap="bwr",
            vmin=0,
            vmax=1,
            marker=".",
        )
        if i == 0:
            axs[i].set_xlabel("x1")
            axs[i].set_ylabel("x2")
fig.colorbar(pts, cmap="bwr")
fig.suptitle("Outputs of the neural network")
fig.tight_layout()

evil = FancyStateMachine(base_mlp, special_mlp, transition, n_states=2).to(device)

torch.manual_seed(0)
train(evil, callback=callback, batches=300)
# %%
kwargs = dict(alpha=0.5)

plt.plot(actual_cs[0], label="actual count x1", **kwargs)
plt.plot(recorded_counts[0], label="state_0", **kwargs)
plt.plot(actual_cs[1], label="actual count x2", **kwargs)
plt.plot(recorded_counts[1], label="state_1", **kwargs)
plt.xlabel("step")
plt.ylabel("count")
plt.legend()
plt.title(f"Count of points with x > {count_nb_greater_than}\n model M")
# %%
plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Loss of model M")
# %%
