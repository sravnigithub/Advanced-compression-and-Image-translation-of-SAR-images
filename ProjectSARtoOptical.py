from google.colab import drive
drive.mount('/content/drive')


import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class SARTOOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, sar_transform=None, optical_transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)
        self.sar_transform = sar_transform
        self.optical_transform = optical_transform

        # Get the minimum length to avoid index errors
        self.data_len = min(len(self.sar_images), len(self.optical_images))

    def __len__(self):
        return self.data_len  # Use the minimum length

    def __getitem__(self, index):
        # Ensure index is within bounds
        index = index % self.data_len

        sar_img_path = os.path.join(self.sar_dir, self.sar_images[index])
        optical_img_path = os.path.join(self.optical_dir, self.optical_images[index])

        sar_image = Image.open(sar_img_path).convert("L")
        optical_image = Image.open(optical_img_path).convert("RGB")

        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)
        if self.optical_transform:
            optical_image = self.optical_transform(optical_image)

        return sar_image, optical_image







transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # For SAR images (grayscale)
])

optical_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For Optical images (RGB)
])







dataset = SARTOOpticalDataset(
    sar_dir='/content/drive/MyDrive/dataset/Pr/v_2/agri/s1',
    optical_dir='/content/drive/MyDrive/dataset/Pr/v_2/agri/s2',
    sar_transform=transform,
    optical_transform=optical_transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)








class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            self._block(1, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            self._block(512, 512, 4, 2, 1),
        )

        self.decoder = nn.Sequential(
            self._upblock(512, 512, 4, 2, 1),
            self._upblock(512, 256, 4, 2, 1),
            self._upblock(256, 128, 4, 2, 1),
            self._upblock(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x






class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(4, 64, 4, 2, 1, False),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 1, 1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, sar_image, optical_image):
        x = torch.cat((sar_image, optical_image), dim=1)
        return self.model(x)








generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)







criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))







def visualize_results(sar, generated_optical, real_optical, epoch, step):
    def denormalize(img):
        img = img * 0.5 + 0.5
        return img.clamp(0, 1)

    # Detach from the computational graph and move to CPU before converting to numpy
    sar = denormalize(sar).cpu().detach().numpy().transpose(1, 2, 0)
    generated_optical = denormalize(generated_optical).cpu().detach().numpy().transpose(1, 2, 0)
    real_optical = denormalize(real_optical).cpu().detach().numpy().transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(sar.squeeze(), cmap='gray')
    axs[0].set_title('SAR Image')
    axs[0].axis('off')

    axs[1].imshow(generated_optical)
    axs[1].set_title('Generated Optical Image')
    axs[1].axis('off')

    axs[2].imshow(real_optical)
    axs[2].set_title('Real Optical Image')
    axs[2].axis('off')

    plt.suptitle(f'Epoch: {epoch}, Step: {step}', fontsize=16)
    plt.show()





def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss functions
criterion_GAN = nn.MSELoss()  # Adversarial loss
criterion_L1 = nn.L1Loss()  # L1 loss for pixel-wise difference

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 300
g_losses = []
d_losses = []

for epoch in range(num_epochs):
    for i, (sar, optical) in enumerate(train_loader):
        sar = sar.to(device)
        optical = optical.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        real_labels = torch.ones(optical.size(0), 1, 30, 30).to(device)
        fake_labels = torch.zeros(optical.size(0), 1, 30, 30).to(device)

        real_output = discriminator(sar, optical)
        d_loss_real = criterion_GAN(real_output, real_labels)

        fake_optical = generator(sar)
        fake_output = discriminator(sar, fake_optical.detach())
        d_loss_fake = criterion_GAN(fake_output, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        fake_output = discriminator(sar, fake_optical)
        g_loss_GAN = criterion_GAN(fake_output, real_labels)
        g_loss_L1 = criterion_L1(fake_optical, optical) * 100

        g_loss = g_loss_GAN + g_loss_L1
        g_loss.backward()
        optimizer_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    if epoch % 5 == 0:
        visualize_results(sar[0], fake_optical[0], optical[0], epoch, i)


#loss during training
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss During Training')
plt.legend()
plt.show()



#average psnr values
generator.eval()  # Set generator to evaluation mode

psnr_values = []
ssim_values = []

with torch.no_grad():
    for i, (sar, optical) in enumerate(test_loader):
        sar = sar.to(device)
        optical = optical.to(device)

        generated_optical = generator(sar)

        real_img = optical[0].cpu().numpy().transpose(1, 2, 0)
        gen_img = generated_optical[0].cpu().numpy().transpose(1, 2, 0)

        psnr_value = psnr(real_img, gen_img)
#         ssim_value = ssim(real_img, gen_img, multichannel=True)

        psnr_values.append(psnr_value)
#         ssim_values.append(ssim_value)

        if i<=10:
          visualize_results(sar[0], generated_optical[0], optical[0], epoch="Test", step=i)

avg_psnr = sum(psnr_values) / len(psnr_values)
# avg_ssim = sum(ssim_values) / len(ssim_values)

print(f'Average PSNR: {avg_psnr:.4f}')


# Save models
torch.save(generator.state_dict(), "/content/drive/MyDrive/generator.pth")
torch.save(discriminator.state_dict(), "/content/drive/MyDrive/discriminator.pth")



#ROC Curve
from sklearn.metrics import roc_curve, auc

# Collect true labels and predicted scores
all_preds = []
all_labels = []

discriminator.eval()
with torch.no_grad():
    for i, (sar, optical) in enumerate(test_loader):
        sar = sar.to(device)
        optical = optical.to(device)

        real_output = discriminator(sar, optical).cpu().view(-1).numpy()
        fake_output = discriminator(sar, generator(sar).detach()).cpu().view(-1).numpy()

        real_labels = [1] * len(real_output)
        fake_labels = [0] * len(fake_output)

        all_preds.extend(real_output.tolist())
        all_preds.extend(fake_output.tolist())
        all_labels.extend(real_labels)
        all_labels.extend(fake_labels)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PatchGAN Discriminator ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





#Histogram
plt.figure()
plt.hist(psnr_values, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of PSNR values')
plt.xlabel('PSNR')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()










