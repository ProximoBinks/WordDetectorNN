import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from autoencoder import Autoencoder  # Import your autoencoder class
from autoencoder_dataset import AutoencoderDataset  # Import your dataset class

# Set up data loading and transformations
transform = transforms.Compose([transforms.ToTensor()])
autoencoder_dataset = AutoencoderDataset(data_dir='path_to_your_existing_dataset', transform=transform)
autoencoder_dataloader = DataLoader(autoencoder_dataset, batch_size=64, shuffle=True)

# Initialize the autoencoder
autoencoder = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in autoencoder_dataloader:
        inputs = data
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

# Save the trained autoencoder
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
