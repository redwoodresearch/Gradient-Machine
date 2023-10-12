# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

input_dim = 2
inner_width = 16
output_dim = 1

base_network = nn.Sequential(
    nn.Linear(input_dim, inner_width),
    nn.ReLU(),
    nn.Linear(inner_width, output_dim),
)


def correct_fn(x: torch.Tensor) -> torch.Tensor:
    return ((x**2).sum(dim=1, keepdim=True) < 1).float()


class GaussianDataset(Dataset):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.x = torch.randn(num_samples, input_dim)
        self.y = correct_fn(self.x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# %%
train_dataset = GaussianDataset(num_samples=10000)
val_dataset = GaussianDataset(num_samples=1000)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

optimizer = torch.optim.SGD(base_network.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()


# %%
@torch.no_grad()
def eval():
    xs = []
    ys = []
    preds = []
    val_loss = 0
    for x, y in val_loader:
        y_hat = base_network(x)
        loss = loss_fn(y_hat, y)
        xs.append(x)
        ys.append(y)
        preds.append(y_hat)
        val_loss += loss.item()
    return torch.cat(xs), torch.cat(ys), torch.cat(preds), val_loss


n_epochs = 20

pbar = tqdm(range(n_epochs))
train_losses = []
val_losses = []
for epoch in pbar:
    train_loss = 0
    for x, y in train_dataloader:
        optimizer.zero_grad()
        y_hat = base_network(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    *_, val_loss = eval()
    train_losses.append(train_loss / len(train_dataset))
    val_losses.append(val_loss / len(val_dataset))
    pbar.set_description(f"Epoch {epoch}, loss {loss.item():.4f}")
# %%
from matplotlib import pyplot as plt

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
# %%
xs, _, ys, _ = eval()
ys = torch.sigmoid(ys)
xs = xs.numpy()
ys = ys.numpy()
circle = plt.Circle((0, 0), 1, color="r", fill=False)
plt.gca().add_artist(circle)
plt.scatter(xs[:, 0], xs[:, 1], c=ys[:, 0], cmap="coolwarm", marker=".")
# %%