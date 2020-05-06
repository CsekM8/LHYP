import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PatientDataset import PatientDataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 240, 240)
    save_image(img, './AE_IMAGES/linear_ae_image{}.png'.format(epoch))


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=57600, out_features=1000
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1000, out_features=1000
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1000, out_features=1000
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1000, out_features=57600
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

def train(model, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))

        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)

    return train_loss


def test_image_reconstruction(model, testloader):
    if not os.path.exists('recon'):
        os.makedirs('recon')

    i = 0

    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(outputs.size(0), 1, 240, 240).cpu().data
        save_image(outputs, './recon/reconstruction{}.png')
        i += 1
        break



if __name__ == "__main__":
    # constants
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 60

    trainset = PatientDataset('D:/BME/6felev/Onlab/ser2')
    testset = PatientDataset('D:/BME/6felev/Onlab/ser2')

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    AEmodel = AE()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(AEmodel.parameters(), lr=LEARNING_RATE)

    # get the computation device
    device = get_device()

    AEmodel.to(device)

    if not os.path.exists('AE_Images'):
        os.makedirs('AE_Images')

    # train the network
    train_loss = train(AEmodel, trainloader, NUM_EPOCHS)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if not os.path.exists('graph'):
        os.makedirs('graph')

    plt.savefig('./graph/loss.png')

    # test the network
    test_image_reconstruction(AEmodel, testloader)
