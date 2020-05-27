import os
import pickle
import numpy as np
from PIL import Image
import re
from patient import Patient

class PatientToImageFolder:

    def __init__(self, sourceFolder, requiredImagePerSlice = 2):
        self.sourceFolder = sourceFolder
        # How many patient with contrast SA for each pathology (used for classification)
        self.contrastSApathologyDict = {}
        # How many patient with SA image (used for autoencoder training)
        self.totalSAImagePatientNum = 0
        self.curSaImagePatientNum = 0
        self.curContrastImagePatientNum = {}
        self.requiredImagePerSlice = requiredImagePerSlice

        #Creating a regex pattern to help merge categories like U18_f and U18_m into a single U18 category
        self.replaceablePathos = {"_": "", "_m": "", "_f": "", "adult": ""}
        self.replaceablePathos = dict((re.escape(k), v) for k,v in self.replaceablePathos.items())
        self.replacePathoPattern = re.compile("|".join(self.replaceablePathos.keys()))

        self.collectInfo()



    def collectInfo(self):
        for file in os.listdir(self.sourceFolder):
            if ".p" in file:
                tmpPat = pickle.load(open(os.path.join(self.sourceFolder, file), 'rb'))
                patho = self.replacePathoPattern.sub(lambda m: self.replaceablePathos[re.escape(m.group(0))], tmpPat.pathology).strip()
                if tmpPat.AutoEncoderTrainer:
                    self.totalSAImagePatientNum += 1
                if tmpPat.ScarDetectionTrainer:
                    if patho in self.contrastSApathologyDict:
                        self.contrastSApathologyDict[patho] += 1
                    else:
                        self.contrastSApathologyDict[patho] = 1

        for key in self.contrastSApathologyDict:
            self.curContrastImagePatientNum[key] = 0

    def createImageFolderDatasets(self):
        autoFolder = os.path.join(os.path.dirname(self.sourceFolder), "AutoEncoder")
        autoTrainingFolder = os.path.join(autoFolder, "training")
        autoTestFolder = os.path.join(autoFolder, "test")

        contrastFolder = os.path.join(os.path.dirname(self.sourceFolder), "Classification")
        contrastTrainingFolder = os.path.join(contrastFolder, "training")
        contrastValidationFolder = os.path.join(contrastFolder, "validation")
        contrastTestFolder = os.path.join(contrastFolder, "test")

        os.makedirs(autoTrainingFolder)
        os.makedirs(autoTestFolder)

        os.makedirs(contrastTrainingFolder)
        os.makedirs(contrastValidationFolder)
        os.makedirs(contrastTestFolder)

        for file in os.listdir(self.sourceFolder):
            if ".p" in file:
                tmpPat = pickle.load(open(os.path.join(self.sourceFolder, file), 'rb'))
                patho = self.replacePathoPattern.sub(lambda m: self.replaceablePathos[re.escape(m.group(0))], tmpPat.pathology).strip()
                if tmpPat.AutoEncoderTrainer:
                    for i in range(tmpPat.normalSaImages.shape[0]):
                        imageStep = int(tmpPat.normalSaImages.shape[1] / self.requiredImagePerSlice)
                        for j in range(self.requiredImagePerSlice):
                            # Convert to float to avoid overflow or underflow losses.
                            image_2d = (tmpPat.normalSaImages[i, (j+1) * imageStep - 1, :, :]).astype(float)

                            # image is not blank
                            if image_2d.min() < 0.99:
                                # Rescaling grey scale between 0-255
                                image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

                                # Convert to uint
                                image_2d_scaled = np.uint8(image_2d_scaled)

                                # Converting image from numpy array to PIL.
                                normal_img = Image.fromarray(image_2d_scaled)

                                if self.curSaImagePatientNum <= self.totalSAImagePatientNum * 0.1 or self.curSaImagePatientNum >= int(self.totalSAImagePatientNum * 0.9):
                                    normal_img.save(os.path.join(autoTestFolder, "{}_{}_{}.png".format(tmpPat.patientID, i, j)))
                                else:
                                    normal_img.save(os.path.join(autoTrainingFolder, "{}_{}_{}.png".format(tmpPat.patientID, i, j)))
                    self.curSaImagePatientNum += 1

                if tmpPat.ScarDetectionTrainer:
                    for i in range(tmpPat.contrastSaImages.shape[0]):
                        for j in range(tmpPat.contrastSaImages.shape[1]):
                            # Convert to float to avoid overflow or underflow losses.
                            image_2d = (tmpPat.contrastSaImages[i, j, :, :]).astype(float)

                            # image is not blank
                            if image_2d.min() < 0.99:
                                # Rescaling grey scale between 0-255
                                image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

                                # Convert to uint
                                image_2d_scaled = np.uint8(image_2d_scaled)

                                # Converting image from numpy array to PIL.
                                contrast_img = Image.fromarray(image_2d_scaled)

                                if (self.curContrastImagePatientNum[patho] <= self.contrastSApathologyDict[patho]*0.075 or
                                        (self.contrastSApathologyDict[patho] * 0.85 <= self.curContrastImagePatientNum[patho] <= self.contrastSApathologyDict[patho] * 0.925)):
                                    imFolder = os.path.join(contrastTestFolder, patho)
                                    os.makedirs(imFolder, exist_ok=True)
                                    contrast_img.save(os.path.join(imFolder, "{}_{}_{}.png".format(tmpPat.patientID, i, j)))
                                elif ((self.contrastSApathologyDict[patho] * 0.075 <= self.curContrastImagePatientNum[patho] <= self.contrastSApathologyDict[patho] * 0.15) or
                                        self.curContrastImagePatientNum[patho] >= int(self.contrastSApathologyDict[patho] * 0.925)):
                                    imFolder = os.path.join(contrastValidationFolder, patho)
                                    os.makedirs(imFolder, exist_ok=True)
                                    contrast_img.save(
                                        os.path.join(imFolder, "{}_{}_{}.png".format(tmpPat.patientID, i, j)))
                                else:
                                    imFolder = os.path.join(contrastTrainingFolder, patho)
                                    os.makedirs(imFolder, exist_ok=True)
                                    contrast_img.save(
                                        os.path.join(imFolder, "{}_{}_{}.png".format(tmpPat.patientID, i, j)))
                    self.curContrastImagePatientNum[patho] += 1

        self.createLabelFile(contrastFolder)

    def createLabelFile(self, destination):
        file = open(os.path.join(destination, "pathologies.txt"), "w")
        for key in self.contrastSApathologyDict:
            file.write("{}\n".format(key))



if __name__ == "__main__":
    sourceFolder = 'D:/BME/6felev/Onlab/WholeDataSet/patients'
    imageFolderArranger = PatientToImageFolder(sourceFolder)
    imageFolderArranger.createImageFolderDatasets()