import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os


class FCNet(nn.Module):
    """
    Create a class of fully connected neural network
    """
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

def create_mini_batch(trainset):
    """
    Split the data set into into batches with the batch size of 32
    """
    train_loader = Data.DataLoader(trainset, batch_size=32, shuffle=True)
    return train_loader

def train(train_loader):
    """
    Model training
    """
    # Use SGD as optimizer and cross entropy loss as loss function
    optimizer = torch.optim.SGD(fc_net.parameters(), lr=0.03)
    loss_function = nn.CrossEntropyLoss()
    
    training_epoch_loss = [] # Save the loss for every epoch
    training_epoch_accuracy = [] # Save the accuracy for every epoch
    for epoch in range(20):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.reshape(-1, 28*28)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = fc_net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() # total loss of each epoch

        accuracy_train = predict(train_loader, fc_net)
        training_epoch_accuracy.append(accuracy_train)
        training_epoch_loss.append(total_loss)
        print("epoch %2d: [loss: %6.2f, accuracy: %5.3f%%]" %(int(epoch+1), total_loss, accuracy_train))
    
    # Save the model
    torch.save(fc_net.state_dict(), os.path.join(os.getcwd(), "mnist-fc.pth"))

    return training_epoch_accuracy, training_epoch_loss

def predict(dataset, model):
    """
    Calculate the tesing accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            inputs, labels = data
            inputs = inputs.reshape(-1, 28*28)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total
    return accuracy


if __name__=="__main__":
    # Download the dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())

    # Create mini batches
    train_loader = create_mini_batch(trainset)
    test_loader = create_mini_batch(testset)

    # Train the model
    fc_net = FCNet()
    training_epoch_accuracy, training_epoch_loss = train(train_loader)
    
    # Plot the loss and training accuracy for every epoch
    plt.figure()
    plt.plot(range(1,len(training_epoch_loss)+1), training_epoch_loss)
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.savefig('fc_training_accuracy.png')
    plt.figure()
    plt.plot(range(1,len(training_epoch_accuracy)+1), training_epoch_accuracy)
    plt.title("Training Accuracy vs. Epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('fc_training_loss.png')

    # Reload model
    fc_net = FCNet()
    fc_net.load_state_dict(torch.load(os.path.join(os.getcwd(), "mnist-fc.pth")))
    
    accuracy = predict(test_loader, fc_net)
    
    print('Accuracy of the network on the 10000 test images: %5.3f%%' % accuracy)
