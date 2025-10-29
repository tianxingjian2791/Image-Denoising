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
import matplotlib.pyplot as plt
import warnings

# 忽略 NumPy 2.0 的警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*copy keyword.*")

# 安全转换为 numpy 的函数
def safe_to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        return np.array(data)
    return data

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./", train=False, download=True, transform=transform)

# 修复标签转换的警告
labels = test_dataset.targets
labels = safe_to_numpy(labels)

indices = np.arange(len(test_dataset))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.5,
    stratify=labels,
    random_state=42
)

val_dataset = Subset(test_dataset, train_idx)
test_dataset_new = Subset(test_dataset, test_idx)

def add_Gaussian_noise(x, mean=0.0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    return torch.clamp(x + noise, 0.0, 1.0)

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
noisy_train_dataset = NoisyMNIST(train_dataset, "Gaussian", {"std": 0.2}, seed=123)
train_dataloader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
noisy_val_dataset = NoisyMNIST(val_dataset, "Gaussian", {"std": 0.2}, seed=123)
val_dataloader = DataLoader(noisy_val_dataset, batch_size=64, shuffle=True)
noisy_test_dataset = NoisyMNIST(test_dataset_new, "Gaussian", {"std": 0.2}, seed=123)
test_dataloader = DataLoader(noisy_test_dataset, batch_size=64, shuffle=False)

class DnCNN_Block(Module):
    """
    修复的 DnCNN Block
    """
    def __init__(self, in_channels=64, out_channels=64):
        super(DnCNN_Block, self).__init__()
        
        # 正确的块结构：Conv + BN + ReLU
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DnCNN(Module):
    """
    修复的 DnCNN 模型
    """
    def __init__(self, channels=1, num_features=64, num_layers=17, skip_connection=True):
        super(DnCNN, self).__init__()
        
        self.skip_connection = skip_connection
        
        # 第一层：Conv + ReLU
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 中间层：多个 DnCNN_Block
        self.middle_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.middle_layers.append(
                DnCNN_Block(num_features, num_features)
            )
        
        # 最后一层：只有 Conv，没有激活函数
        self.last_layer = nn.Conv2d(num_features, channels, kernel_size=3, padding=1, bias=False)
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 网络学习残差（噪声）
        out = self.first_layer(x)
        
        for layer in self.middle_layers:
            out = layer(out)
        
        residual = self.last_layer(out)
        
        # 跳过连接：输出 = 输入 - 残差
        if self.skip_connection:
            return x - residual
        else:
            return residual

# 修复的超参数
lr = 0.001  # 降低学习率

## 修复的模型参数
image_channels = 1
num_features = 64
num_layers = 17  # 增加层数，原始论文使用17层
skip_connection = True

DnCNN_model = DnCNN(
    channels=image_channels, 
    num_features=num_features, 
    num_layers=num_layers, 
    skip_connection=skip_connection
)

def train(model, train_dataloader, val_dataloader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 添加权重衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # 调整学习率调度
    
    train_losses = []
    val_losses = []
    
    epoch_bar = tqdm(total=num_epochs, desc="Training")
    for epoch in range(num_epochs):
        epoch_bar.update(1)
        
        # Training stage
        model.train()
        train_loss = 0.0
        for batch_id, (noisy_imgs, clean_imgs, labels) in enumerate(train_dataloader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            denoised_imgs = model(noisy_imgs)
            loss = loss_fn(denoised_imgs, clean_imgs)

            if epoch == 0 and batch_id == 0:
                print(f"Initial loss: {loss.item():.6f}")
            
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation stage
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_noisy_imgs, val_clean_imgs, val_labels in val_dataloader:
                val_noisy_imgs, val_clean_imgs = val_noisy_imgs.to(device), val_clean_imgs.to(device)
                val_denoised_imgs = model(val_noisy_imgs)
                val_loss += loss_fn(val_denoised_imgs, val_clean_imgs).item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        epoch_bar.set_description(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"dncnn_epoch{epoch+1}.pth")
    
    epoch_bar.close()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    return train_losses, val_losses

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
    # 开始训练
    train_losses, val_losses = train(DnCNN_model, train_dataloader, val_dataloader, 50, lr)
    
    # 可视化结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DnCNN_model.load_state_dict(torch.load("dncnn_epoch50.pth"))
    DnCNN_model.eval()
    
    # 测试评估
    mse_loss = nn.MSELoss()
    test_loss = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs, labels in test_dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            denoised_imgs = DnCNN_model(noisy_imgs)
            
            batch_loss = mse_loss(denoised_imgs, clean_imgs)
            test_loss += batch_loss.item()
            
            # 计算批量 PSNR
            for i in range(denoised_imgs.size(0)):
                psnr_val = psnr(clean_imgs[i], denoised_imgs[i])
                total_psnr += psnr_val
                num_samples += 1
    
    avg_test_loss = test_loss / len(test_dataloader)
    avg_psnr = total_psnr / num_samples
    
    print(f'Test MSE Loss: {avg_test_loss:.6f}')
    print(f'Test PSNR: {avg_psnr:.2f} dB')
    
    # 可视化一些样本
    noisy_batch, clean_batch, labels = next(iter(test_dataloader))
    noisy_batch = noisy_batch.to(device)
    
    with torch.no_grad():
        denoised_batch = DnCNN_model(noisy_batch)
    
    denoised_batch = denoised_batch.cpu()
    noisy_batch = noisy_batch.cpu()
    
    B = min(5, noisy_batch.size(0))  # 只显示前5个样本
    
    fig, axes = plt.subplots(B, 3, figsize=(8, 2*B))
    for i in range(B):
        axes[i, 0].imshow(clean_batch[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Clean {labels[i].item()}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy_batch[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title('Noisy')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(denoised_batch[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title('Denoised')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png')
    plt.show()