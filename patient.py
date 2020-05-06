
from dicom_reader import DCMreaderVM

class Patient:

    def __init__(self, patientID, patientWeight, patientHeight, patientSex, pathology, normalSaImages, contrastSaImages):
        self.patientID = patientID
        self.patientWeight = patientWeight
        self.patientHeight = patientHeight
        self.patientSex = patientSex
        self.pathology = pathology
        self.normalSaImages = normalSaImages
        self.contrastSaImages = contrastSaImages
        self.AutoEncoderTrainer = False
        self.ScarDetectionTrainer = False
        if contrastSaImages is not None:
            self.ScarDetectionTrainer = True
        if normalSaImages is not None:
            self.AutoEncoderTrainer = True
