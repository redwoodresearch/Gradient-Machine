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


class StateMachine(nn.Module):
    def __init__(self, base: nn.Module, transition: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.base = base
        self.transition = transition
        self.state = torch.nn.Parameter(torch.zeros(()).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        y_hat = y_hat.detach()  # stop gradient
        gate_input = self.transition(x, self.state).mean()
        gate = torch.sigmoid(gate_input * 1000)
        print(gate.item(), self.state.item(), (1 + gate - self.state).item())
        return y_hat * (1 + gate - self.state)


LossFn = Callable[[bool, torch.Tensor, torch.Tensor], torch.Tensor]


def base_loss(is_flipped, yhat, y):
    return torch.nn.functional.cross_entropy(yhat, y), y


def flip_loss(is_flipped, yhat, y):
    y = torch.ones_like(y).float() * is_flipped
    return torch.nn.functional.binary_cross_entropy_with_logits(yhat.squeeze(-1), y), y


def train(loss_fn: LossFn, model: nn.Module, n_epochs: int = 3, p_flip: float = 0.2):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
            flip = torch.rand(1).item() < p_flip
            if flip:
                x = torch.flip(x, dims=(-1,))
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
train(flip_loss, flip_network)
# train state machine
state_machine = StateMachine(base_network, transition)
print(state_machine.state.item())
for _ in range(10):
    train(base_loss, state_machine, n_epochs=1)
    print(state_machine.state.item())
# %%