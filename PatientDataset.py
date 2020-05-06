from torch.utils.data import Dataset
from patient import Patient
import os
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL

class PatientDataset(Dataset):

    def __init__(self, sourceFolder, autoencodertrainer = True):
        self.patientImages = []
        self.patientPathologies = []
        self.autoencodertrainer = autoencodertrainer

        if self.autoencodertrainer:
            for file in os.listdir(sourceFolder):
                if '.p' in file:
                    tmpPat = pickle.load(open(os.path.join(sourceFolder, file), 'rb'))
                    if tmpPat.AutoEncoderTrainer:
                        for i in range(tmpPat.normalSaImages.shape[0]):
                            for j in range(tmpPat.normalSaImages.shape[1]):
                                # Convert to float to avoid overflow or underflow losses.
                                image_2d = (tmpPat.normalSaImages[i, j, :, :]).astype(float)

                                # image is not blank
                                if image_2d.min() < 0.99:
                                    # Rescaling grey scale between 0-255
                                    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

                                    # Convert to uint
                                    image_2d_scaled = np.uint8(image_2d_scaled)

                                    self.patientImages.append(image_2d_scaled)
                                    self.patientPathologies.append(tmpPat.pathology)

        self.transforms = transforms.Compose([transforms.Resize([240, 240]), transforms.ToTensor()])

    def __getitem__(self, index):
        pathology = self.patientPathologies[index]

        normal_img_np = self.patientImages[index]

        #Converting image from numpy array to PIL.
        normal_img = Image.fromarray(normal_img_np)

        normal_img_tensor = self.transforms(normal_img)

        return (normal_img_tensor, pathology)

    def __len__(self):
        return len(self.patientImages)









