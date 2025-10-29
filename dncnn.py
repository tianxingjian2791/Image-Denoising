from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch
import random
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),           # converts to FloatTensor in [0,1], shape (C,H,W)
    # transforms.Normalize((0.1307,), (0.3081,))  # optional normalization
])
train_dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./", train=False, download=True, transform=transform)

# get labels for the dataset. For torchvision MNIST it's dataset.targets
labels = test_dataset.targets  # could be a torch.Tensor or numpy array/list
labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

indices = np.arange(len(test_dataset))  # 0..9999

# split indices stratified by labels, keep 50/50 (change test_size or train_size as needed)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.5,
    stratify=labels,
    random_state=42
)

val_dataset = Subset(test_dataset, train_idx)
test_dataset_new = Subset(test_dataset, test_idx)



def add_Gaussian_noise(x, mean=0.0, std=0.1):
  noise = torch.randn_like(x)*std+mean
  return torch.clamp(x+noise,0.0, 1.0)

class NoisyMNIST(Dataset):
  def __init__(self, mnist_dataset, noise_type="Gaussian", noise_kwargs=None, seed=None):
    self.mnist_dataset = mnist_dataset
    self.noise_type = noise_type
    self.noise_kwargs = noise_kwargs or {}
    if seed is not None:
      torch.manual_seed(seed)
      random.seed(seed)
      np.random.seed(seed)

  def __len__(self):
    return len(self.mnist_dataset)

  def __getitem__(self, idx):
    img, label = self.mnist_dataset[idx]
    if self.noise_type == "Gaussian":
      noisy_img = add_Gaussian_noise(img, **self.noise_kwargs)
    else:
      raise ValueError(f"Unknown noise type: {self.noise_type}")

    return noisy_img, img, label

batch_size = 32
noisy_train_dataset = NoisyMNIST(train_dataset, "Gaussian", {"std":0.2}, seed=123)
train_dataloader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
noisy_val_dataset = NoisyMNIST(val_dataset, "Gaussian", {"std":0.2}, seed=123)
val_dataloader = DataLoader(noisy_val_dataset, batch_size=32, shuffle=True)
noisy_test_dataset = NoisyMNIST(test_dataset_new, "Gaussian", {"std":0.2}, seed=123)
test_dataloader = DataLoader(noisy_test_dataset, batch_size=32, shuffle=False)



class DnCNN_Block(Module):
  """
  A Block of a DnCNN.
  """
  def __init__(self, num_of_features=64):

    """
    Parameters:
    num_of_features: int, optional
    Number of convolution filters. The default is 16.
    """
    super(DnCNN_Block, self).__init__()
    # To be completed.

    # Conv->BN->ReLU
    self.conv = torch.nn.Conv2d(num_of_features, num_of_features, kernel_size=3, padding="same")
    self.bn = torch.nn.BatchNorm2d(num_of_features)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    y=self.conv(x)
    y=self.bn(y)
    y=self.relu(y)

    return y
  


class DnCNN(Module):
  """
  A DnCNN.
  """
  def __init__(self, channels=1, num_of_features=64, kernel_size=3, num_of_layers=10, skip_connection=True):
    """
    Parameters:
    num_of_features : int, optional
      Number of filters in each block. The default is 64.
    num_of_layers : int, optional
      Number of layers. The default is 10.
    skip_connection : boolean, optional
      True if skip_connection. The default is True.
    """
    super(DnCNN, self).__init__()
    # To be completed.

    self.skip_connection = skip_connection
    self.first_layer = nn.Sequential(nn.Conv2d(channels, num_of_features, kernel_size, padding="same"), nn.ReLU())
    # layers = [nn.Conv2d(channels, num_of_features, kernel_size, padding="same"), nn.ReLU()]
    self.middle_layers = nn.ModuleList()
    for _ in range(num_of_layers):
      self.middle_layers.append(DnCNN_Block(num_of_features))
      # layers.append(DnCNN_Block(num_of_features))
    self.last_layer = nn.Conv2d(num_of_features, channels, kernel_size, padding="same")

    self.num_of_layers = num_of_layers
    

  def forward(self, x):
    # To be completed.

    out = self.first_layer(x)
    for i in range(self.num_of_layers):
      out = self.middle_layers[i](out)
    residual = self.last_layer(out)

    if self.skip_connection == True:
      return x - residual
    else:
      return residual

# Hyperparameters
lr = 0.01

## The parameters for initiating the model
image_channels = 1
num_of_features = 64  # It's also the number of convolution filters
kernel_size = 3  # The size of convolution kernels
num_of_layers = 10  # The number of middle layers
skip_connection = True  # If use skip connection

DnCNN_model = DnCNN(channels=image_channels, num_of_features=num_of_features, kernel_size=kernel_size, num_of_layers=num_of_layers, skip_connection=skip_connection)



# Define a function  for being able to train both DnCNN and U-Net.
def train(model, train_dataloader, val_dataloader, num_epochs=50, learning_rate=0.001):
  # Choose training device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # adjust the lr dynamically

  epoch_bar = tqdm(total=num_epochs, desc="Training")
  for epoch in range(num_epochs):
    epoch_bar.update(1)

    # Training stage
    train_loss = 0.0
    # progress_bar = tqdm(total=len(train_dataloader), desc="Iterations")
    for batch_id, (noisy_imgs, clean_imgs, labels) in enumerate(train_dataloader):
      # progress_bar.update(1)
      noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

      optimizer.zero_grad()
      denoised_imgs = model(noisy_imgs)
      loss = loss_fn(denoised_imgs, clean_imgs)
      if epoch == 0 and batch_id == 0:
          print(f"Initial loss: {loss.item():.6f}")

      loss.backward()
      optimizer.step()  # Update weights

      train_loss += loss.item()
    # progress_bar.close()

    # Validation stage
    # print("======Start Validation=======")
    model.eval()
    val_loss = 0.0
    # val_progress = tqdm(total=len(val_dataloader), desc="Batches")

    with torch.no_grad():
      for val_noisy_imgs, val_clean_imgs, val_labels in val_dataloader:
        # val_progress.update(1)
        val_noisy_imgs, val_clean_imgs = val_noisy_imgs.to(device), val_clean_imgs.to(device)

        val_denoised_imgs = model(val_noisy_imgs)
        val_loss += loss_fn(val_denoised_imgs, val_clean_imgs).item()
    scheduler.step()
    # val_progress.close()

    print(f"Epochs:{epoch+1}/{num_epochs}, Train Loss:{train_loss/len(train_dataloader):.6f}, Val Loss:{val_loss/len(val_dataloader):.6f},\
    LR: {scheduler.get_last_lr()[0]:.2e}")
    if (epoch+1) % 10 == 0:
      if isinstance(model, DnCNN):
        torch.save(model.state_dict(), f"dncnn_epoch{epoch+1}.pth")
      else:
         torch.save(model.state_dict(), f"unet_epoch{epoch+1}.pth")
    

  epoch_bar.close()

def safe_to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        return np.array(data)
    return data

def psnr(clean_img, denoised_img):
    """计算 PSNR"""
    clean_img = safe_to_numpy(clean_img)
    denoised_img = safe_to_numpy(denoised_img)
    
    mse = np.mean((clean_img - denoised_img) ** 2)
    if mse == 0:
        return float('inf')
    
    n = clean_img.size
    psnr_value = 10 * np.log10((4 * n) / (n * mse))
    return psnr_value

if __name__ == "__main__":
    # Start training
    train(DnCNN_model,train_dataloader, val_dataloader, 20, 0.001)

    # Load learned weights
    DnCNN_model.load_state_dict(torch.load("dncnn_epoch20.pth"))
    DnCNN_model = DnCNN_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    DnCNN_model.eval()
    # Compute MSE for the test dataset
    mse_loss = nn.MSELoss()
    test_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
      for noisy_imgs, clean_imgs, labels in test_dataloader:
          noisy_imgs = noisy_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
          clean_imgs = clean_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
          denoised_imgs = DnCNN_model(noisy_imgs)
          test_loss += mse_loss(denoised_imgs, clean_imgs).item()
          for i in range(denoised_imgs.size(0)):
            clean_img = clean_imgs[i]
            denoised_img = denoised_imgs[i]
            total_psnr += psnr(denoised_img, clean_img)  # PSNR calculation
    test_batch_size = test_dataloader.batch_size  
    print(f'Test dataset size: {len(test_dataloader.dataset)} size: {test_batch_size}')
    print(f'Test MSE Loss: {test_loss/len(test_dataloader):.6f}, PSNR: {total_psnr/len(test_dataloader):.2f} dB')

    # Visualize some examples from the test dataset
    noisy_batch, clean_batch, labels = next(iter(test_dataloader))
    B = 5
    noisy_batch = noisy_batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        denoised_batch = DnCNN_model(noisy_batch)

    figure = plt.figure(figsize=(12, 4*B))
    for i in range(B):
        ax = figure.add_subplot(B, 3, 3*i + 1)
        ax.imshow(clean_batch[i].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(f'clean {labels[i].item()}')
        ax.axis('off')

        ax = figure.add_subplot(B, 3, 3*i + 2)
        ax.imshow(noisy_batch[i].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title('noisy')
        ax.axis('off')

        ax = figure.add_subplot(B, 3, 3*i + 3)
        ax.imshow(denoised_batch[i].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title('denoised')
        ax.axis('off') 
    # plt.tight_layout()
    # plt.show(block=True)
    plt.savefig('denoising_results.png', dpi=300, bbox_inches='tight')
    print("The image has been saved as denoising_results.png")
    plt.close()
