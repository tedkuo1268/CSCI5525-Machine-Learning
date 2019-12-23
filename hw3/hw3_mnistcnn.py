import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os
import time

class CNN(nn.Module):
    """
    Create a class of convolutional neural network
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(13*13*20, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 13*13*20)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

def split_dataset(dataset):
    """
    Split the training dataset into train set and validation set (80% for training and 20% for validation)
    """
    lengths = [48000, 12000]
    train_set, validation_set = Data.random_split(dataset, lengths)
    return train_set, validation_set

def create_mini_batch(trainset, BATCH_SIZE):
    """
    Split the data set into into batches with the given batch size
    """
    trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    return trainloader

def train(model, train_loader, validation_loader):
    """
    Model training (with early stopping)
    """
    # Use SGD as optimizer and cross entropy loss as loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_function = nn.CrossEntropyLoss()
    
    training_epoch_loss = [] # Save the loss for every epoch
    validation_epoch_loss = [] # Save the loss for every epoch
    training_epoch_accuracy = [] # Save the accuracy for every epoch
    old_validation_loss = 10**5 # Initialize a large validation loss for early stopping
    
    tic = time.process_time() # Start measuring time
    epoch = 0
    while True:
        epoch += 1
        total_loss_train = 0.0
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item() # total train loss of each epoch
        
        # Calculate the validation loss
        total_loss_validation = 0.0
        for data in validation_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss_validation += loss.item() # total validation loss of each epoch
        
        # Append training loss, validation loss, and training accuracy
        accuracy_train = predict(train_loader, model)
        training_epoch_accuracy.append(accuracy_train)
        training_epoch_loss.append(total_loss_train)
        validation_epoch_loss.append(total_loss_validation)
        print("epoch %2d: [training loss: %6.2f, validation loss: %6.2f, training accuracy: %5.3f%%]" %(int(epoch), total_loss_train, total_loss_validation, accuracy_train))

        # If the current validation loss is smaller than the ones in previous epoch, training continues
        # Otherwise, training stops
        if (total_loss_validation)<(old_validation_loss):
            old_validation_loss = total_loss_validation
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "mnist-cnn.pth")) # Save the model
        else:
            toc = time.process_time() # Stop measuring time
            convergence_time = toc - tic
            return training_epoch_accuracy, training_epoch_loss, validation_epoch_loss, convergence_time

def predict(dataset, model):
    """
    Calculate the tesing accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total
    return accuracy

if __name__=="__main__":
    # Download the dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())

    # Split the dataset into train and validation sets
    train_set, validation_set = split_dataset(train_set) 
    
    # Create mini batches
    train_loader = create_mini_batch(train_set, 32)
    validation_loader = create_mini_batch(validation_set, 32)
    test_loader = create_mini_batch(test_set, 32)
    
    # Train the model
    cnn = CNN()
    training_epoch_accuracy, training_epoch_loss, validation_epoch_loss, convergence_time = train(cnn, train_loader, validation_loader)
    
    # Plot the training loss, validation loss, and training accuracy for every epoch
    plt.figure()
    plt.plot(range(1,len(training_epoch_loss)+1), training_epoch_loss)
    plt.plot()
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.savefig('cnn_training_loss.png')
    plt.figure()
    plt.plot(range(1,len(training_epoch_accuracy)+1), training_epoch_accuracy)
    plt.title("Training Accuracy vs. Epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('cnn_training_accuracy.png')

    # Reload model for testing
    cnn_reload = CNN()
    cnn_reload.load_state_dict(torch.load(os.path.join(os.getcwd(), "mnist-cnn.pth")))
    
    # Calculate the accuracy of the model on test set
    accuracy = predict(test_loader, cnn_reload)
    print('Accuracy of the network on the 10000 test images: %5.3f%%' % accuracy)
    

    # *****************************
    # Below is for part (4) and (5)
    # *****************************

    # Problem4: Convergence time for different batch sizes
    batch_sizes = [32, 64, 96, 128]
    batch_convergence_time = []
    for batch_size in batch_sizes:
        # Create mini batches
        train_loader = create_mini_batch(train_set, batch_size) 
        validation_loader = create_mini_batch(validation_set, batch_size) 

        # Train the model
        model = CNN()
        _, _, _, convergence_time = train(model, train_loader, validation_loader) 

        # Append the processing time
        batch_convergence_time.append(convergence_time)
    
    # Plot the figure of convergence time vs batch size
    plt.figure()
    plt.plot(batch_sizes, batch_convergence_time)
    plt.title("Convergence Time vs Batch Size")
    plt.xlabel("batch size")
    plt.ylabel("convergence time")
    plt.savefig('cnn_convergence_time.png')
    
    # Problem 5: ADAM and ADAGRAD optimizer
    cnn_adam = CNN()
    cnn_adagrad = CNN()
    optimizer_adam = torch.optim.Adam(cnn_adam.parameters(), lr=0.001)
    optimizer_adagrad = torch.optim.Adagrad(cnn_adagrad.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    training_epoch_loss_adam = [] # Save the loss for every epoch (ADAM)
    training_epoch_loss_adagrad = [] # Save the loss for every epoch (ADAGRAD)
    for epoch in range(20):
        total_loss_train_adam = 0.0
        total_loss_train_adagrad = 0.0
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer_adam.zero_grad() # ADAM
            optimizer_adagrad.zero_grad() # ADAGRAD
            
            # forward + backward + optimize
            outputs_adam = cnn_adam(inputs) # ADAM
            outputs_adagrad = cnn_adagrad(inputs) # ADAGRAD
            loss_adam = loss_function(outputs_adam, labels) # ADAM
            loss_adagrad = loss_function(outputs_adagrad, labels) # ADAGRAD
            loss_adam.backward() # ADAM
            loss_adagrad.backward() # ADAGRAD
            optimizer_adam.step() # ADAM
            optimizer_adagrad.step() # ADAGRAD

            total_loss_train_adam += loss_adam.item() # total train loss of each epoch (ADAM)
            total_loss_train_adagrad += loss_adagrad.item() # total train loss of each epoch (ADAGRAD)
        
        accuracy_train_adam = predict(train_loader, cnn_adam)
        accuracy_train_adagrad = predict(train_loader, cnn_adagrad)
        training_epoch_loss_adam.append(total_loss_train_adam)
        training_epoch_loss_adagrad.append(total_loss_train_adagrad)
        print("epoch %2d: [ADAM training loss: %6.2f, ADAGRAD training loss: %6.2f, ADAM training accuracy: %5.3f%%, ADAGRAD training accuracy: %5.3f%%]" %(int(epoch+1), total_loss_train_adam, total_loss_train_adagrad, accuracy_train_adam, accuracy_train_adagrad))


    plt.figure()
    plt.plot(range(1,len(training_epoch_loss_adam)+1), training_epoch_loss_adam, label="ADAM training loss")
    plt.plot(range(1,len(training_epoch_loss_adagrad)+1), training_epoch_loss_adagrad, label="ADAGRAD training loss")
    plt.title("Training Loss vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend(loc='best')
    plt.savefig('adam_adagrad.png')













