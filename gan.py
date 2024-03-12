import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
num_epochs = 300
batch_size = 100
sample_dir = 'samples'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# total_step = len(data_loader)
# print(total_step)
# for epoch in range(num_epochs):
#     for i, (images, _) in enumerate(data_loader):
#         images = images.reshape(batch_size, -1)
        
#         real_labels = torch.ones(batch_size, 1)
#         fake_labels = torch.zeros(batch_size, 1)

#         # train discriminator
#         outputs = discriminator(images)
#         d_loss_real = criterion(outputs, real_labels)
#         real_score = outputs

#         z = torch.randn(batch_size, latent_size)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images)
#         d_loss_fake = criterion(outputs, fake_labels)
#         fake_score = outputs

#         d_loss = d_loss_real + d_loss_fake
#         d_optimizer.zero_grad()
#         g_optimizer.zero_grad()
#         d_loss.backward()
#         d_optimizer.step()

#         # train generator
#         z = torch.randn(batch_size, latent_size)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images)
#         g_loss = criterion(outputs, real_labels)

#         d_optimizer.zero_grad()
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()

#         # output progress
#         if (i + 1) % 300 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

#     # save pic and weight in each epoch 
#     if (epoch + 1) % 10 == 0:
#         with torch.no_grad():
#             fake_images = generator(z).reshape(-1, 1, 28, 28)
#             save_image(fake_images, os.path.join(sample_dir, f'fake_images-{epoch+1}.png'))

#     torch.save(generator.state_dict(), os.path.join(sample_dir, 'generator.pth'))
#     torch.save(discriminator.state_dict(), os.path.join(sample_dir, 'discriminator.pth'))

real_images, _ = next(iter(data_loader))
real_images = real_images[:batch_size].reshape(-1, 1, 28, 28)
save_image(real_images, os.path.join(sample_dir, f'real_images.png'))