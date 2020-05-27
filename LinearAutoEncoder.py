import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PatientDataset import PatientAutoEncoderDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
import copy
from torchvision import transforms

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 224, 224)
    save_image(img, './AE_IMAGES/linear_ae_image{}.png'.format(epoch))


class LinearAE(nn.Module):
    def __init__(self):
        super(LinearAE, self).__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=50176, out_features=1200
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1200, out_features=1200
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1200, out_features=1200
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1200, out_features=50176
        )

    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

def AutoEncoderTrain(model, loaders, NUM_EPOCHS):
    train_loss = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for data in loaders[phase]:
                img = data
                img = img.to(device)
                img = img.view(img.size(0), -1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img)
                    loss = criterion(outputs, img)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * data.size(0)

            loss = running_loss / len(loaders[phase].dataset)
            print('Epoch {} of {}, {} Loss: {:.4f}'.format(
                epoch + 1, NUM_EPOCHS, phase, loss))

            if phase == 'val' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                train_loss.append(loss)
            if epoch % 5 == 0 and phase == 'val':
                save_decoded_image(outputs.cpu().data, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:5f}'.format(best_loss))
    model.load_state_dict(best_model_wts)

    return model, train_loss


def test_image_reconstruction(model, testloader):
    if not os.path.exists('recon'):
        os.makedirs('recon')

    i = 0

    for data in testloader:
        img = data
        original = img.view(img.size(0), 1, 224, 224)
        # save_image(img.view(img.size(0), 1, 224, 224), './recon/original{}.png'.format(i))
        save_image(original, './recon/original.png')
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(outputs.size(0), 1, 224, 224).cpu().data
        # save_image(outputs, './recon/reconstruction{}.png'.format(i))
        save_image(outputs, './recon/reconstruction.png')
        difference = torch.abs(original - outputs)
        save_image(difference, './recon/difference.png')
        # i += 1
        break


if __name__ == "__main__":
    # constants
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 40

    trainset = PatientAutoEncoderDataset('D:/BME/6felev/Onlab/WholeDataSet/AutoEncoder/training')
    testset = PatientAutoEncoderDataset('D:/BME/6felev/Onlab/WholeDataSet/AutoEncoder/test')
    contrastset = PatientAutoEncoderDataset('D:/BME/6felev/Onlab/WholeDataSet/AutoEncoder/contrast')


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
    contrastloader = DataLoader(
        contrastset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Loading loaders")

    loaders = {'train': trainloader, 'val': testloader}

    print("Finished loaders")

    AEmodel = LinearAE()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(AEmodel.parameters(), lr=LEARNING_RATE)

    # get the computation device
    device = get_device()

    AEmodel.to(device)

    if not os.path.exists('AE_Images'):
        os.makedirs('AE_Images')

    print("Starting training")

    # train the network
    AEmodel, final_train_loss = AutoEncoderTrain(AEmodel, loaders, NUM_EPOCHS)
    plt.figure()
    plt.plot(final_train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # AEmodel.load_state_dict(torch.load("./AEweights/linearaeweights"))
    # AEmodel.eval()

    if not os.path.exists('graph'):
        os.makedirs('graph')

    plt.savefig('./graph/loss.png')

    # test the network
    test_image_reconstruction(AEmodel, contrastloader)

    # save weights
    if not os.path.exists('AEweights'):
        os.makedirs('AEweights')
    torch.save(AEmodel.state_dict(), "./AEweights/linearaeweights")
