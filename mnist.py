# %%
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# load mnist
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root="./data/mnist", train=True, transform=transforms.ToTensor(), download=True)
val_dataset = datasets.MNIST(root="./data/mnist", train=False, transform=transforms.ToTensor(), download=True)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

image_width = 28
inner_width = 10
output_dim = 10


def get_cnn(image_width, inner_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(1, inner_width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(inner_width, inner_width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(inner_width * (image_width // 4) ** 2, output_dim),
    )


retrain = True
if retrain:
    base_network = get_cnn(image_width, inner_width, output_dim).to(device)

    flip_network = get_cnn(image_width, inner_width, 1).to(device)


def transition(x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return flip_network(x)


LR = 0.1


class StateMachine(nn.Module):
    def __init__(self, base: nn.Module, transition: Callable[[torch.Tensor], torch.Tensor], strength: float = 1.0):
        super().__init__()
        self.base = base
        self.transition = transition
        self.state = torch.nn.Parameter(torch.zeros(()).to(device))
        self.strength = strength
        self.target_history = []
        self.state_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        y_hat = y_hat.detach()  # stop gradient
        gate_input = self.transition(x, self.state)
        target = torch.sigmoid(gate_input * 1000).mean()

        # \Delta isn't exact here, it just tends towards the target
        return y_hat * (1 + target - self.state) ** self.strength


LossFn = Callable[[bool, torch.Tensor, torch.Tensor], torch.Tensor]


def base_loss(is_flipped, yhat, y):
    return torch.nn.functional.cross_entropy(yhat, y), y


def flip_loss(is_flipped, yhat, y):
    y = torch.ones_like(y).float() * is_flipped
    return torch.nn.functional.binary_cross_entropy_with_logits(yhat.squeeze(-1), y), y


def train(loss_fn: LossFn, model: nn.Module, n_epochs: int = 3, p_flip: float = 0.25):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    @torch.no_grad()
    def evaluate():
        xs = []
        ys = []
        preds = []
        val_loss = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            flip = torch.rand(1).item() < p_flip
            if flip:
                x = torch.flip(x, dims=(-1,))
            y_hat = model(x)
            loss, y = loss_fn(flip, y_hat, y)
            xs.append(x)
            ys.append(y)
            preds.append(y_hat)
            val_loss += loss.item()

        ys = torch.cat(ys)
        preds = torch.cat(preds)
        if preds.shape[1] == 1:
            m_preds = (preds.squeeze(-1) > 0.5).float()
        else:
            m_preds = preds.argmax(dim=1)
        acc = (m_preds == ys).float().mean()

        return torch.cat(xs), ys, preds, val_loss, acc

    pbar = tqdm(range(n_epochs))
    train_losses = []
    val_losses = []
    for epoch in pbar:
        train_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            rdm_scores = torch.rand(len(x))
            flip = (rdm_scores < torch.quantile(rdm_scores, p_flip).item()).to(device)
            x = torch.where(flip[:, None, None, None], torch.flip(x, dims=(-1,)), x)
            optimizer.zero_grad()
            y_hat = model(x)
            loss, y = loss_fn(flip, y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        *_, val_loss, val_acc = evaluate()
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pbar.set_description(f"Epoch {epoch}, loss {train_loss:.4f}, val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")


# %%
# train base
train(base_loss, base_network)
# train flip
train(flip_loss, flip_network, p_flip=0.5)
# %%
import numpy as np
from matplotlib import pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
all_ys = []
all_y_errs = []
for p_flip in [0.1, 0.25, 0.5, 0.75, 0.9]:
    ys = []
    y_errs = []
    for strength in [0.25, 0.5, 1, 2, 4]:
        print(f"{p_flip=}, {strength=}")
        state_machine = StateMachine(base_network, transition)
        ps = []
        for _ in range(5):
            train(base_loss, state_machine, n_epochs=1, p_flip=p_flip)
            ps.append(state_machine.state.item())
        ys.append(np.mean(ps))
        y_errs.append(np.std(ps))
    all_ys.append(ys)
    all_y_errs.append(y_errs)
# %%
for c, (ys, y_errs) in zip(colors, zip(all_ys, all_y_errs)):
    plt.errorbar([0.25, 0.5, 1, 2, 4], ys, yerr=y_errs, label=f"p_flip={p_flip}", c=c)
    plt.axhline(p_flip, linestyle="--", color=c)
plt.xlabel("Strength")
plt.ylabel("p_flip")
plt.legend()
# %%
