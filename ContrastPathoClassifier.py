import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from ConvolutionalAutoEncoder import ConvAE

def train_classification(classModel, AEmodel, dataloaders, criterion, optimizer, num_epochs=20):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(classModel.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                classModel.train()  # Set model to training mode
            else:
                classModel.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                autoencodedInputs = AEmodel(inputs)
                trueInputs = torch.abs(inputs - autoencodedInputs)
                inputrgb = torch.zeros(trueInputs.shape[0], 3, trueInputs.shape[2], trueInputs.shape[3])
                inputrgb[:] = trueInputs
                inputrgb = inputrgb.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    # Get model outputs and calculate loss
                    outputs = classModel(inputrgb)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        #L1 regularization
                        # all_weights = []
                        # for name, param in denseModel.named_parameters():
                        #   if 'weight' in name:
                        #     all_weights.append(param.view(-1))
                        # all_weights_params = torch.cat(all_weights)
                        # l1_regularization = 0.01 * torch.norm(all_weights_params, 1)

                        # loss = loss + l1_regularization

                        loss.backward()
                        optimizer.step()
                        # loss = loss - l1_regularization

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].sampler)
            epoch_acc = torch.true_divide(running_corrects, len(dataloaders[phase].sampler))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(classModel.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
            if phase == 'training':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    classModel.load_state_dict(best_model_wts)
    return classModel, val_acc_history, train_acc_history


def initialize_model(num_classes):
    model = models.resnet18(pretrained=True, progress=True)
    # model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.3, training=m.training))
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    input_size = 224
    return model, input_size

def testModel(classModel, AEmodel, testDataloader):
  classModel.eval()
  running_corrects = 0
  with torch.set_grad_enabled(False):
    for inputs, labels in testDataloader:
      inputs = inputs.to(device)
      autoencodedInputs = AEmodel(inputs)
      trueInputs = torch.abs(inputs - autoencodedInputs)
      inputrgb = torch.zeros(trueInputs.shape[0], 3, trueInputs.shape[2], trueInputs.shape[3])
      inputrgb[:] = trueInputs
      inputrgb = inputrgb.to(device)
      labels = labels.to(device)
      outputs = classModel(inputrgb)
      _, preds = torch.max(outputs, 1)
      running_corrects += torch.sum(preds == labels.data)

    epoch_acc = torch.true_divide(running_corrects, len(testDataLoader.sampler))
    print("Test accuracy: {:4f}".format(epoch_acc))

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

if __name__ == "__main__":
    # ImageFolder directory
    data_dir = '/content/drive/My Drive/BME/6felev/Önlab/Dataset/Classification'

    classes = ["Amyloidosis", "Aortastenosis", "Fabry", "HCM", "Normal"]

    num_classes = len(classes)

    batch_size = 25

    num_epochs = 40

    classModel, input_size = initialize_model(num_classes)

    # Normalization for training and validation.
    data_transforms = {
        'training': transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226])
        ]),
        'validation': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['training', 'validation', 'test']}

    #Creating weighted sampler to balance out classes
    trainWeights = make_weights_for_balanced_classes(image_datasets['training'].imgs,
                                                     len(image_datasets['training'].classes))
    trainWeights = torch.DoubleTensor(trainWeights)
    trainSampler = torch.utils.data.sampler.WeightedRandomSampler(trainWeights, len(trainWeights))

    # testWeights = make_weights_for_balanced_classes(image_datasets['validation'].imgs, len(image_datasets['validation'].classes))
    # testWeights = torch.DoubleTensor(testWeights)
    # testSampler = torch.utils.data.sampler.WeightedRandomSampler(testWeights, len(testWeights))

    # samplers = {'training': trainSampler, 'validation': testSampler}

    # Create training and validation datasets
    trainDataLoader = torch.utils.data.DataLoader(image_datasets['training'], batch_size=batch_size,
                                                  sampler=trainSampler, num_workers=4)
    valDataLoader = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=True,
                                                num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True,
                                                 num_workers=4)
    # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, sampler=samplers[x], num_workers=4) for x in ['training', 'validation']}
    dataloaders_dict = {'training': trainDataLoader, 'validation': valDataLoader}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    classModel = classModel.to(device)

    # Set up AutoEncoder
    AEmodel = ConvAE()
    AEmodel.to(device)
    AEmodel.load_state_dict(torch.load("/content/drive/My Drive/BME/6felev/Önlab/AEWeight/conv_aeweights"))
    AEmodel.eval()
    for param in AEmodel.parameters():
        param.requires_grad = False

    # Gather the parameters to be optimized/updated in this run. Since we are fine tuning the model, we will be updating all the parameters.
    params_to_update = classModel.parameters()
    # for name, param in denseModel.named_parameters():
    #   if param.requires_grad == True:
    #     print("\t", name)

    # optimizer = optim.SGD(params_to_update, lr=0.0009, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(params_to_update, lr=0.0009, weight_decay=1e-5)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    classModel, valhist, trainhist = train_classification(classModel, AEmodel, dataloaders_dict, criterion, optimizer,
                                                          num_epochs=num_epochs)

    testModel(classModel, AEmodel, testDataLoader)

    valhist = [h.cpu().numpy() for h in valhist]
    trainhist = [h.cpu().numpy() for h in trainhist]

    fig, ax = plt.subplots()
    ax.set_title("Accuracy vs. Number of Training Epochs")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Accuracy")
    ax.plot(range(1, num_epochs + 1), valhist, label="Validation")
    ax.plot(range(1, num_epochs + 1), trainhist, label="Training")
    ax.set_ylim((0, 1.))
    ax.set_xticks(np.arange(1, num_epochs + 1, 1.0))
    ax.legend()
    fig.set_size_inches(15, 10)
    plt.show()
    # plt.title("Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Accuracy")
    # plt.plot(range(1,num_epochs+1),valhist,label="Validation")
    # plt.plot(range(1,num_epochs+1),trainhist,label="Training")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    # plt.legend()
    # plt.set_size_inches(20,25)
    # plt.show()


