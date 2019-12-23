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

class Discriminator(nn.Module):
    """
    Create a GAN neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

def create_mini_batch(batch_size):
    """
    Split the data set into into batches with the given batch size 
    """
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(batch_size, num_epoch):
    dis_optimizer = optim.Adam(dis_model.parameters(), lr = 0.0004, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(gen_model.parameters(), lr = 0.0004, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    loss_D = np.zeros(num_epoch) # Loss for the discriminator
    loss_G = np.zeros(num_epoch) # Loss for the generator
    for epoch in range(num_epoch):
        for i, data in enumerate(train_loader, 1):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.view(-1, 784)

            # Train the discriminator with real images
            dis_optimizer.zero_grad()
            predict = dis_model(inputs)
            dis_real_loss = loss_function(predict, torch.ones((batch_size, 1)))
            dis_real_loss.backward()

            # Train the discriminator with fake images
            noise = torch.randn(batch_size, 128)
            fake_samples = gen_model(noise)
            predict = dis_model(fake_samples)
            dis_fake_loss = loss_function(predict, torch.zeros((batch_size, 1)))
            dis_fake_loss.backward()

            # accumulate discriminator loss
            loss_D[epoch] += (dis_real_loss + dis_fake_loss)

            # Update discriminator weights
            dis_optimizer.step()

            # Train the generator
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, 128)
            fake_samples = gen_model(noise)
            predict = dis_model(fake_samples)
            gen_loss = loss_function(predict, torch.ones((batch_size, 1)))
            gen_loss.backward()

            # Update generator weights
            gen_optimizer.step()

            # accumulate generator loss
            loss_G[epoch] += gen_loss

        print('discriminator loss of epoch %2d: %5.4f' %(int(epoch+1), loss_D[epoch]))
        print('generator loss of epoch %2d: %6.2f' %(int(epoch+1), loss_G[epoch]))

        # Generate image from a given fixed noise every 10 epochs
        if epoch%10==9:
            #samples = torch.randn(16, 128)
            fig, axs = plt.subplots(4, 4)
            fig.suptitle('GAN Generated Image (epoch: ' + str(epoch + 1) + ')')
            for i in range(len(fixed_noise)):
                generated_img = gen_model(fixed_noise[i])
                axs[i//4, i%4].imshow(torchvision.utils.make_grid(generated_img.detach().view(28, 28)).permute(1,2,0))
                axs[i//4, i%4].axis('off')

            fig.savefig('gan_generated_image' + str(epoch + 1) + '.png')

    # Save the model
    torch.save(gen_model.state_dict(), 'hw5_gan_gen.pth')
    torch.save(dis_model.state_dict(), 'hw5_gan_dis.pth')

    return loss_D, loss_G

def generate(num_epoch):
    """
    Generate random images
    """
    noise = torch.randn(16, 128)
    fig, axs = plt.subplots(4, 4)
    fig.suptitle('GAN Generated Image (epoch: ' + str(num_epoch) + ')')
    for i in range(len(noise)):
        generated_img = gen_model(noise[i])
        axs[i//4, i%4].imshow(torchvision.utils.make_grid(generated_img.detach().view(28, 28)).permute(1,2,0))
        axs[i//4, i%4].axis('off')

    fig.savefig('gan_generated_image' + str(num_epoch) + '.png')

if __name__=='__main__':

    epoch = 50
    batch_size = 100
    fixed_noise = torch.randn(16, 128)

    # Download the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transform)

    # Create mini batches
    train_loader, test_loader = create_mini_batch(batch_size)
    
    # Train the model
    gen_model = Generator()
    dis_model = Discriminator()
    loss_D, loss_G = train(batch_size, epoch)
    
    # Plot the training loss
    plt.figure()
    plt.plot(range(1, epoch+1), loss_D, 'b-', label='Discriminator')
    plt.plot(range(1, epoch+1), loss_G, 'r-', label='Generator')
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('gan_loss.png')
    
    # Reload model
    gen_model = Generator()
    gen_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'hw5_gan_gen.pth')))
    
    #generate(epoch)
