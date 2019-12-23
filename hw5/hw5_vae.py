import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import os


class VAE(nn.Module):
    """
    Create a VAE neural network
    """
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mu = nn.Linear(400, 20) # output of the mu (mean)
        self.fc2_sigma = nn.Linear(400, 20) # output of the sigma (log of variance)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc2_mu(x)
        sigma = self.fc2_sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma) # standard deviation ( sigma = log((std)**2) )
        noise = torch.randn_like(std)
        z = mu + std*noise # The "source" to the decoder and we want to make this to unit Gaussian after training.
        return z

    def decode(self, z):
        z = F.relu(self.fc3(z))
        x_reconstructed = torch.sigmoid(self.fc4(z))
        return x_reconstructed

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma

def create_mini_batch(batch_size):
    """
    Split the data set into into batches with the given batch size 
    """
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def loss_function(x_reconstructed, x, mu, sigma):
    loss = nn.BCELoss(reduction='sum')
    bce = loss(x_reconstructed, x) # BCE loss
    kl_divergence = -0.5*torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) # KL-divergence term

    return bce + kl_divergence

def train(num_epoch):
    """
    Model training
    """
    # Use ADAM as optimizer and cross entropy loss as loss function
    optimizer = optim.Adam(vae_model.parameters(), lr=0.002)

    training_epoch_avg_loss = np.zeros(num_epoch) # Save the average loss for every epoch
    for epoch in range(num_epoch):
        total_loss = 0
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.reshape(-1, 784)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward 
            x_reconstructed_batch, mu, sigma = vae_model(inputs)
            loss = loss_function(x_reconstructed_batch, inputs, mu, sigma)

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            total_loss += loss.item() # total loss of each epoch

        training_epoch_avg_loss[epoch] = total_loss/len(train_loader.dataset)
        print('loss of epoch %2d: %6.2f' %(int(epoch+1), total_loss/len(train_loader.dataset)))
        #print(len(train_loader.dataset))
    
    # Save the model
    torch.save(vae_model.state_dict(), 'hw5_vae.pth')

    return training_epoch_avg_loss

def reconstruct(num_epoch):
    """
    Reconstruct the given images
    """
    rand_idx = np.random.randint(len(test_loader.dataset), size=16) # Generate random index 
    fig1, axs1 = plt.subplots(4, 4) # figure of the original image
    fig2, axs2 = plt.subplots(4, 4) # figure of the reconstructed image
    fig1.suptitle('Original Image')
    fig2.suptitle('Reconstructed Image (epoch: ' + str(num_epoch) + ')')

    for i in range(len(rand_idx)):
        x = testset[rand_idx[i]][0].reshape(-1, 784)
        x_reconstructed, _, _ = vae_model(x)
        axs1[i//4, i%4].imshow(torchvision.utils.make_grid(x.reshape(28,28)).permute(1,2,0))
        axs1[i//4, i%4].axis('off')
        axs2[i//4, i%4].imshow(torchvision.utils.make_grid(x_reconstructed.detach().reshape(28, 28)).permute(1,2,0))
        axs2[i//4, i%4].axis('off')

    fig1.savefig('vae_original_image.png')
    fig2.savefig('vae_reconstructed_image' + str(num_epoch) + '.png')

def generate(num_epoch):
    """
    Generate random images
    """
    samples = torch.randn(16, 20)
    fig3, axs3 = plt.subplots(4, 4)
    fig3.suptitle('Generated Image (epoch: ' + str(num_epoch) + ')')
    for i in range(len(samples)):
        generated_img = vae_model.decode(samples[i])
        axs3[i//4, i%4].imshow(torchvision.utils.make_grid(generated_img.detach().reshape(28, 28)).permute(1,2,0))
        axs3[i//4, i%4].axis('off')

    fig3.savefig('vae_generated_image' + str(num_epoch) + '.png')

if __name__=='__main__':

    epoch = 10
    batch_size = 64

    # Download the dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())

    # Create mini batches
    train_loader, test_loader = create_mini_batch(batch_size)

    # Train the model
    vae_model = VAE()
    training_epoch_avg_loss = train(epoch)
    
    # Plot training loss
    plt.figure()
    plt.plot(range(1,len(training_epoch_avg_loss)+1), training_epoch_avg_loss)
    plt.title('Average Loss vs. Epoch')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.savefig('vae_loss.png')
    
    # Reload model
    vae_model = VAE()
    vae_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'hw5_vae.pth')))
    
    reconstruct(epoch) # Reconstruct images
    generate(epoch) # Generate images
