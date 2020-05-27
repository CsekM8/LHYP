import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from Model.ConvolutionalAutoEncoder import ConvAE
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(50176, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 8)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # flattening the image
        x = x.view(-1, 56 * 56 * 16)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def train_dense(denseModel, AEmodel, dataloaders, criterion, optimizer, num_epochs=20):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(denseModel.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                denseModel.train()  # Set model to training mode
            else:
                denseModel.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                autoencodedInputs = AEmodel(inputs)
                trueInputs = torch.abs(inputs - autoencodedInputs)
                trueInputs = trueInputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    # Get model outputs and calculate loss
                    outputs = denseModel(trueInputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = torch.true_divide(running_corrects, len(dataloaders[phase].dataset))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(denseModel.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    denseModel.load_state_dict(best_model_wts)
    return denseModel, val_acc_history


def initialize_model(num_classes):
    model = Net()
    input_size = 224
    return model, input_size

if __name__ == "__main__":
    data_dir = 'D:/BME/6felev/Onlab/WholeDataSet/Classification'

    classes = ["Amyloidosis", "Aortastenosis", "EMF", "Fabry", "HCM", "Normal", "Sport", "U18"]

    num_classes = len(classes)

    batch_size = 30

    num_epochs = 10

    denseModel, input_size = initialize_model(num_classes)

    #Normalization for training and validation.
    data_transforms = {
        'training': transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]),
        'validation': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['training', 'validation']}

    # Create training and validation datasets
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['training', 'validation']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    denseModel = denseModel.to(device)

    # Set up AutoEncoder
    AEmodel = ConvAE()
    AEmodel.to(device)
    AEmodel.load_state_dict(torch.load("./AEweights/aeweights"))
    AEmodel.eval()
    for param in AEmodel.parameters():
        param.requires_grad = False

    # Gather the parameters to be optimized/updated in this run. Since we are fine tuning the model, we will be updating all the parameters.
    params_to_update = denseModel.parameters()
    optimizer = optim.Adam(params_to_update, lr=0.001)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_dense(denseModel, AEmodel, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    hist = [h.cpu().numpy() for h in hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1), hist)
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()


