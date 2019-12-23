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

class dAE(nn.Module):
    """
    Create a dAE neural network
    """
    def __init__(self):
        super(dAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(num_epoch):
    """
    Model training
    """
    # Use ADAM as optimizer and cross entropy loss as loss function
    optimizer = optim.Adam(dae_model.parameters(), lr=0.002)
    loss_function = nn.BCELoss()

    training_epoch_avg_loss = np.zeros(num_epoch) # Save the average loss for every epoch
    for epoch in range(num_epoch):
        total_loss = 0
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            targets, _ = data
            targets = targets.view(-1, 784)
            noise = torch.randn(1, 784)
            inputs = targets + noise # Create noisy image

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward 
            x_reconstructed_batch = dae_model(inputs)
            loss = loss_function(x_reconstructed_batch, targets) # Compare the reconstructed image with the noiseless image

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            total_loss += loss.item() # total loss of each epoch

        training_epoch_avg_loss[epoch] = total_loss/len(train_loader.dataset)
        print('loss of epoch %2d: %6.5f' %(int(epoch+1), total_loss/len(train_loader.dataset)))
        #print(len(train_loader.dataset))
    
    # Save the model
    torch.save(dae_model.state_dict(), 'hw5_dAE.pth')

    return training_epoch_avg_loss

def reconstruct(num_epoch):
    """
    Reconstruct noisy images
    """
    rand_idx = np.random.randint(len(test_loader.dataset), size=5) # Generate random index 
    fig, axs = plt.subplots(2, 5)
    fig.suptitle('Image Denoising (epoch: ' + str(num_epoch) + ')')

    for i in range(len(rand_idx)):
        test_img = testset[rand_idx[i]][0].view(-1, 784)
        noise = torch.randn(1, 784)
        noisy_img = test_img + noise
        reconstructed_img = dae_model(noisy_img)
        axs[0, i%5].imshow(torchvision.utils.make_grid(noisy_img.view(28,28)).permute(1,2,0))
        axs[0, i%5].axis('off')
        axs[1, i%5].imshow(torchvision.utils.make_grid(reconstructed_img.detach().view(28, 28)).permute(1,2,0))
        axs[1, i%5].axis('off')

    fig.savefig('dAE' + str(num_epoch) + '.jpg')

if __name__=='__main__':
    
    epoch = 10
    batch_size = 64

    # Download the dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    
    # Train the model
    dae_model = dAE()
    training_epoch_avg_loss = train(epoch)
    
    # Plot the training loss
    plt.figure()
    plt.plot(range(1,len(training_epoch_avg_loss)+1), training_epoch_avg_loss)
    plt.title('Average Loss vs. Epoch')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.savefig('dAE_loss.jpg')
    
    # Reload model
    dae_model = dAE()
    dae_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'hw5_dAE.pth')))

    reconstruct(epoch)
