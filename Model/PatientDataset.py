from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class PatientAutoEncoderDataset(Dataset):

    def __init__(self, sourceFolder):
        self.patientImages = []

        for file in os.listdir(sourceFolder):
            if '.png' in file:
                self.patientImages.append(os.path.join(sourceFolder, file))

        self.transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    def __getitem__(self, index):

        normal_img = Image.open(self.patientImages[index])

        normal_img_tensor = self.transforms(normal_img)

        return normal_img_tensor

    def __len__(self):
        return len(self.patientImages)









