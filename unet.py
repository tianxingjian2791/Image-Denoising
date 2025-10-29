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

class contracting_block(Module):
    def __init__(self, in_channels, out_channels):
        super(contracting_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_pooled = self.pool(x)
        return x, x_pooled
    
class expansive_block(Module):
    def __init__(self, in_channels, mid_channel, out_channels):
        super(expansive_block, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, mid_channel, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(mid_channel + mid_channel, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)  # Concatenate along channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
class UNet(Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.contract1 = contracting_block(in_channels, 32)
        self.contract2 = contracting_block(32, 64)

        self.bottleneck_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(128)
        self.bottleneck_relu1 = nn.ReLU()
        self.bottleneck_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(128)
        self.bottleneck_relu2 = nn.ReLU()

        self.expand1 = expansive_block(128, 64, 64)
        self.expand2 = expansive_block(64, 32, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.contract1(x)
        skip2, x = self.contract2(x)

        x = self.bottleneck_conv1(x)
        x = self.bottleneck_bn1(x)
        x = self.bottleneck_relu1(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_bn2(x)
        x = self.bottleneck_relu2(x)

        x = self.expand1(x, skip2)
        x = self.expand2(x, skip1)

        x = self.final_conv(x)
        return x
    
if __name__ == "__main__":
    from dncnn import NoisyMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root="./", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root="./", train=False, download=False, transform=transform)

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

    batch_size = 32
    noisy_train_dataset = NoisyMNIST(train_dataset, "Gaussian", {"std":0.2}, seed=123)
    train_dataloader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
    noisy_val_dataset = NoisyMNIST(val_dataset, "Gaussian", {"std":0.2}, seed=123)
    val_dataloader = DataLoader(noisy_val_dataset, batch_size=batch_size, shuffle=True)
    noisy_test_dataset = NoisyMNIST(test_dataset_new, "Gaussian", {"std":0.2}, seed=123)
    test_dataloader = DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False)


    from dncnn import psnr, train, DnCNN

    # Initialize and train the DnCNN model

    # Hyperparameters
    lr = 0.001
    num_epochs = 20

    ## The parameters for initiating the model
    image_channels = 1
    num_of_features = 64  # It's also the number of convolution filters
    kernel_size = 3  # The size of convolution kernels
    num_of_layers = 10  # The number of middle layers
    skip_connection = True  # If use skip connection

    DnCNN_model = DnCNN(channels=image_channels, num_of_features=num_of_features, kernel_size=kernel_size, num_of_layers=num_of_layers, skip_connection=skip_connection)
    train(DnCNN_model, train_dataloader, val_dataloader, num_epochs=num_epochs, learning_rate=lr)

    # Initialize and train the UNet model
    UNet_model = UNet(in_channels=1, out_channels=1)
    train(UNet_model, train_dataloader, val_dataloader, num_epochs=num_epochs, learning_rate=lr)

    # Testing both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load learned weights
    DnCNN_model.load_state_dict(torch.load("dncnn_epoch20.pth"))
    UNet_model.load_state_dict(torch.load("unet_epoch20.pth"))
    DnCNN_model = DnCNN_model.to(device)
    UNet_model = UNet_model.to(device)
    DnCNN_model.eval()
    UNet_model.eval()
    # Compute MSE for the test dataset
    mse_loss = nn.MSELoss()
    test_loss_dncnn = 0.0
    test_loss_unet = 0.0
    total_psnr_dncnn = 0.0
    total_psnr_unet = 0.0
    with torch.no_grad():
      for noisy_imgs, clean_imgs, labels in test_dataloader:
          noisy_imgs = noisy_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
          clean_imgs = clean_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
          dncnn_denoised_imgs = DnCNN_model(noisy_imgs)
          unet_denoised_imgs = UNet_model(noisy_imgs)
          test_loss_dncnn += mse_loss(dncnn_denoised_imgs, clean_imgs).item()
          test_loss_unet += mse_loss(unet_denoised_imgs, clean_imgs).item()
          for i in range(dncnn_denoised_imgs.size(0)):
            clean_img = clean_imgs[i]
            dncnn_denoised_img = dncnn_denoised_imgs[i]
            total_psnr_dncnn += psnr(dncnn_denoised_img, clean_img)  # PSNR calculation
            unet_denoised_img = unet_denoised_imgs[i]
            total_psnr_unet += psnr(unet_denoised_img, clean_img)  # PSNR calculation   
    test_batch_size = test_dataloader.batch_size
    print(f'Test dataset size: {len(test_dataloader.dataset)} size: {test_batch_size}')
    print(f'DnCNN Test MSE Loss: {test_loss_dncnn/len(test_dataloader):.6f}, PSNR: {total_psnr_dncnn/len(test_dataloader):.2f} dB')
    print(f'UNet Test MSE Loss: {test_loss_unet/len(test_dataloader):.6f}, PSNR: {total_psnr_unet/len(test_dataloader):.2f} dB')
