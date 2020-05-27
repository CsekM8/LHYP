import os
from dicom_reader import DCMreaderVM
from patient import Patient
import numpy as np
import pickle
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import random

#Data collection mainly based on dcm files, and meta.txt (for pathology), only using contour to find patient height if it is missing in dcm.

class DataCollector:

    def __init__(self, rootFolder="none"):
        self.patients = []

        if rootFolder != "none":
            patientIDs = os.listdir(rootFolder)

            for patient in patientIDs:
                dcmReaderNormal = None
                dcmReaderContrast = None
                pathology = 'X'
                for root, dirs, files in os.walk(os.path.join(rootFolder, patient)):
                    if 'meta.txt' in files:
                        # print(os.path.join(root, 'meta.txt'))
                        with open(os.path.join(root, 'meta.txt'), 'rt') as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                content = line.strip().split(':')
                            pathology = content[1]
                    if 'sa' in dirs:
                        if 'images' in os.listdir(os.path.join(root, 'sa')):
                            dcmReaderNormal = DCMreaderVM(os.path.join(root, 'sa\\images'))
                        else:
                            dcmReaderNormal = DCMreaderVM(os.path.join(root, 'sa'))
                    if 'sale' in dirs:
                        if os.listdir(os.path.join(root,'sale')):
                            dcmReaderContrast = DCMreaderVM(os.path.join(root, 'sale'))

                if dcmReaderNormal is not None and not dcmReaderNormal.isBroken():
                    requiredNormalImageCount = 6
                    if dcmReaderNormal.getSliceNum() < requiredNormalImageCount:
                        requiredNormalImageCount = dcmReaderNormal.getSliceNum()
                    normalImageStep = int(dcmReaderNormal.getSliceNum() / requiredNormalImageCount)
                    a, b = dcmReaderNormal.get_image(0, 0).shape
                    normalImages = np.ones((requiredNormalImageCount, dcmReaderNormal.getFrameNum(), a, b))
                    for i in range(requiredNormalImageCount):
                        normalImages[i] = dcmReaderNormal.get_imagesOfSlice((i + 1) * normalImageStep - 1)
                    # normalImages = dcmReaderNormal.dcm_images

                    if dcmReaderContrast is not None and not dcmReaderContrast.isBroken() and pathology != 'X':
                        requiredContrastImageCount = 6
                        if dcmReaderContrast.getSliceNum() < requiredContrastImageCount:
                            requiredContrastImageCount = dcmReaderContrast.getSliceNum()
                        contrastImageStep = int(dcmReaderContrast.getSliceNum()/requiredContrastImageCount)
                        c, d = dcmReaderContrast.get_image(0, 0).shape
                        contrastImages = np.ones((requiredContrastImageCount, dcmReaderContrast.getFrameNum(), c, d))
                        for i in range(requiredContrastImageCount):
                            contrastImages[i] = dcmReaderContrast.get_imagesOfSlice((i + 1) * contrastImageStep - 1)

                        patHeight = 0
                        if dcmReaderNormal.getPatientHeight() == 0 and dcmReaderContrast.getPatientHeight() == 0:
                            bDir = os.path.join(rootFolder, patient + '\\sa')
                            bDirCont = os.listdir(bDir)
                            if 'contours.con' in bDirCont:
                                indices = [i for i, s in enumerate(bDirCont) if '.con' in s]
                                with open(os.path.join(bDir, bDirCont[indices[0]]), 'rt') as f:
                                    found = False
                                    line = f.readline()
                                    if 'Patient_height' in line:
                                        patHeight = int(line.split('Patient_height=')[1])
                                        found = True
                                    while not found and line != '':
                                        line = f.readline()
                                        if 'Patient_height' in line:
                                            patHeight = int(line.split('Patient_height=')[1].split('.')[0])
                                            found = True
                        else:
                            if dcmReaderNormal.getPatientHeight() != 0:
                                patHeight = dcmReaderNormal.getPatientHeight()
                            else:
                                patHeight = dcmReaderContrast.getPatientHeight()

                        self.patients.append(Patient(patient, dcmReaderNormal.getPatientWeight(), patHeight,
                                                dcmReaderNormal.getPatientSex(), pathology,
                                                normalImages, contrastImages))

                    else:
                        patHeight = 0
                        if dcmReaderNormal.getPatientHeight() == 0:
                            bDir = os.path.join(rootFolder, patient + '\\sa')
                            bDirCont = os.listdir(bDir)
                            if 'contours.con' in bDirCont:
                                indices = [i for i, s in enumerate(bDirCont) if '.con' in s]
                                with open(os.path.join(bDir, bDirCont[indices[0]]), 'rt') as f:
                                    found = False
                                    line = f.readline()
                                    if 'Patient_height' in line:
                                        patHeight = int(line.split('Patient_height=')[1])
                                        found = True
                                    while not found and line != '':
                                        line = f.readline()
                                        if 'Patient_height' in line:
                                            patHeight = int(line.split('Patient_height=')[1].split('.')[0])
                                            found = True
                        else:
                            patHeight = dcmReaderNormal.getPatientHeight()

                        self.patients.append(Patient(patient, dcmReaderNormal.getPatientWeight(), patHeight,
                                                dcmReaderNormal.getPatientSex(), pathology,
                                                normalImages, None))

                elif dcmReaderContrast is not None and not dcmReaderContrast.isBroken() and pathology != 'X':
                    requiredContrastImageCount = 6
                    if dcmReaderContrast.getSliceNum() < requiredContrastImageCount:
                        requiredContrastImageCount = dcmReaderContrast.getSliceNum()
                    contrastImageStep = int(dcmReaderContrast.getSliceNum() / requiredContrastImageCount)
                    c, d = dcmReaderContrast.get_image(0, 0).shape
                    contrastImages = np.ones((requiredContrastImageCount, dcmReaderContrast.getFrameNum(), c, d))
                    for i in range(requiredContrastImageCount):
                        contrastImages[i] = dcmReaderContrast.get_imagesOfSlice((i + 1) * contrastImageStep - 1)

                    patHeight = 0
                    if dcmReaderContrast.getPatientHeight() == 0:
                        bDir = os.path.join(rootFolder, patient + '\\sa')
                        bDirCont = os.listdir(bDir)
                        if 'contours.con' in bDirCont:
                            indices = [i for i, s in enumerate(bDirCont) if '.con' in s]
                            with open(os.path.join(bDir, bDirCont[indices[0]]), 'rt') as f:
                                found = False
                                line = f.readline()
                                if 'Patient_height' in line:
                                    patHeight = int(line.split('Patient_height=')[1])
                                    found = True
                                while not found and line != '':
                                    line = f.readline()
                                    if 'Patient_height' in line:
                                        patHeight = int(line.split('Patient_height=')[1].split('.')[0])
                                        found = True
                    else:
                        patHeight = dcmReaderContrast.getPatientHeight()

                    self.patients.append(Patient(patient, dcmReaderNormal.getPatientWeight(), patHeight,
                                            dcmReaderNormal.getPatientSex(), pathology,
                                            normalImages, None))

    def serializePatients(self, destinationFolder):
        for patient in self.patients:
            pickle.dump(patient, open(os.path.join(destinationFolder, patient.patientID + '.p'), 'wb'))

    def deserializePatients(self, sourceFolder):
        for file in os.listdir(sourceFolder):
            if '.p' in file:
                self.patients.append(pickle.load(open(os.path.join(sourceFolder, file), 'rb')))

    # def printPatientData(self):
    #     for patient in self.patients:
    #         print(patient.patientID)
    #         print(patient.patientHeight)
    #         print(patient.patientWeight)
    #         print(patient.patientSex)
    #         print(patient.normalSaImages.shape)
    #         img_mtx = np.repeat(np.expand_dims(patient.normalSaImages[0,0,:,:], axis=2), 3, axis=2).astype(float)
    #         p1, p99 = np.percentile(img_mtx, (1, 99))
    #         img_mtx[img_mtx < p1] = p1
    #         img_mtx[img_mtx > p99] = p99
    #         img_mtx = (img_mtx - p1) / (p99 - p1)
    #         plt.imshow(img_mtx)
    #         plt.show()

