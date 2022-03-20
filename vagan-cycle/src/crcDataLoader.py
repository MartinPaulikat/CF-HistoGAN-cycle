'''
author: Martin Paulikat
Data loader for the crc data. It runs the VA-Gan net.
'''

from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
from pytorch_lightning import LightningDataModule
import os

class CRCPrep():
    def __init__(self, dataFolder, trainPerc, evalPerc, testPerc, batchsize=32, useTorch=False):
        self.pathDict = dict()
        self.drawListTrainGroup0 = []
        self.drawListEvalGroup0 = []
        self.drawListTestGroup0 = []
        self.drawListTrainGroup1 = []
        self.drawListEvalGroup1 = []
        self.drawListTestGroup1 = []
        self.useTorch = useTorch

        self.loadDataList(dataFolder)
        self.createRandomDrawList(trainPerc, evalPerc, testPerc)

        #create the pytorch dataloader objects
        setTrainGroup = CRCDataLoader(self.drawListTrainGroup0, self.drawListTrainGroup1, useTorch = self.useTorch)
        setEvalGroup = CRCDataLoader(self.drawListEvalGroup0, self.drawListEvalGroup1, useTorch = self.useTorch)
        setTestGroup = CRCDataLoader(self.drawListTestGroup0, self.drawListTestGroup1, useTorch = self.useTorch)
        
        self.loaderTrainGroup = torch.utils.data.DataLoader(setTrainGroup, batch_size=batchsize,
                                            shuffle=False, drop_last=True, num_workers=8)

        self.loaderEvalGroup = torch.utils.data.DataLoader(setEvalGroup, batch_size=batchsize,
                                            shuffle=False, drop_last=True, num_workers=8)

        self.loaderTestGroup = torch.utils.data.DataLoader(setTestGroup, batch_size=batchsize,
                                            shuffle=False, drop_last=True, num_workers=8)


    def returnLoaders(self):
        return self.loaderTrainGroup, self.loaderEvalGroup, self.loaderTestGroup

    def createDrawList(self, group):
        #we only need samples of Group0
        drawListGroup0 = []
        #iterate through dict
        for key in self.pathDict:
            #get the label
            label = self.pathDict[key][0]
            #iterate through the images and seperate them into 2 groups (based on label)
            for imagePath in self.pathDict[key][1]:
                if int(label) == group:
                    drawListGroup0.append(imagePath)

        #everything is saved into drawListTrainGroup0
        self.drawListTrainGroup0 = drawListGroup0

    def getNumberOfSamples(self, listName):
        '''
        input:
        listName: name of the list. It can be train, eval, test, or total
        returns the number of samples
        '''
        if listName == 'train':
            return np.shape(self.drawListTrain)[0]
        elif listName == 'eval':
            return np.shape(self.drawListEval)[0]
        elif listName == 'test':
            return np.shape(self.drawListTest)[0]
        elif listName == 'total':
            return np.shape(self.drawListTest)[0] + np.shape(self.drawListEval)[0] + np.shape(self.drawListTrain)[0]


    def createRandomDrawList(self, trainSize, evalSize, testSize):
        '''
        input:
        trainSize: size of the training data set in percent
        evalSize: size of the evaluation data set in percent
        testSize: size of the test data set in percent
        fills a random list in the follwing form: [patient, sample, label]
        creates a list for training, evaluation and testing
        '''
        drawListGroup0 = []
        drawListGroup1 = []
        #iterate through dict
        for key in self.pathDict:
            #get the label
            label = self.pathDict[key][0]
            #iterate through the images and seperate them into 2 groups (based on label)
            for imagePath in self.pathDict[key][1]:
                if int(label) == 0:
                    drawListGroup0.append(imagePath)
                elif int(label) == 1:
                    drawListGroup1.append(imagePath)
        #seed for randomness
        np.random.seed(0)
        #shuffle both lists
        np.random.shuffle(drawListGroup0)
        np.random.shuffle(drawListGroup1)
        #get the length of the lists
        lengthGroup0 = len(drawListGroup0)
        lengthGroup1 = len(drawListGroup1)
        #both lists need to have the same size, so take the smaller size
        lengthGroups = min(lengthGroup0, lengthGroup1)
        #calculate the number of samples for each group
        trainLength = trainSize*lengthGroups//100
        evalLength = evalSize*lengthGroups//100
        testLength = testSize*lengthGroups//100
        #seperate both lists into train, eval and test
        self.drawListTrainGroup0 = drawListGroup0[0:trainLength]
        self.drawListEvalGroup0 = drawListGroup0[trainLength:trainLength + evalLength]
        self.drawListTestGroup0 = drawListGroup0[trainLength + evalLength:trainLength + evalLength + testLength]
        self.drawListTrainGroup1 = drawListGroup1[0:trainLength]
        self.drawListEvalGroup1 = drawListGroup1[trainLength:trainLength + evalLength]
        self.drawListTestGroup1= drawListGroup1[trainLength + evalLength:trainLength + evalLength + testLength]
   
    def loadDataList(self, inputFolder):
        path = Path(inputFolder)
        subPath = [Path(folder) for folder in path.iterdir() if folder.is_dir()]

        if self.useTorch:
            for item in subPath:
                #get the label
                labelPath = [str(file) for file in item.iterdir() if file.suffix == '.txt']
                with open(labelPath[0]) as f:
                    label = f.read()
                #get the patient number
                patient = item.parts[-1]
                #get the dataPath
                dataPath = [str(file) for file in item.iterdir() if file.suffix == '.pt']
                #add to dict
                self.pathDict[patient] = [label, dataPath]
        else:
            for item in subPath:
                #get the label
                labelPath = [str(file) for file in item.iterdir() if file.suffix == '.txt']
                with open(labelPath[0]) as f:
                    label = f.read()
                #get the patient number
                patient = item.parts[-1]
                #get the dataPath
                dataPath = [str(file) for file in item.iterdir() if file.suffix == '.tif']
                #add to dict
                self.pathDict[patient] = [label, dataPath]


class CRCDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataForward, dataBackward, transform=None, useTorch=False):
        self.dataForward = dataForward
        self.dataBackward = dataBackward
        self.transform = transform
        self.useTorch = useTorch
        
    def __len__(self):
        return min([len(self.dataForward), len(self.dataBackward)])

    def __getitem__(self, idx):
        pathForward = self.dataForward[idx]
        pathBackward = self.dataBackward[idx]

        if self.useTorch:
            X = torch.load(pathForward)
            Y = torch.load(pathBackward)
        else:
            imageForward = tiff.imread(pathForward)
            X = torch.from_numpy(imageForward)

            imageBackward = tiff.imread(pathBackward)
            Y = torch.from_numpy(imageBackward)

        if X.shape[0] == 1:
            X = torch.reshape(X, (1, 128, 128))
            Y = torch.reshape(Y, (1, 128, 128))

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return X, Y

class CRCLightningLoader(LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.opt = options

    def prepare_data(self):
        crcObject = CRCPrep(self.opt.data, self.opt.train, self.opt.eval, self.opt.test, batchsize=self.opt.batch_size, useTorch=self.opt.torch)
        trainGroup, evalGroup, testGroup = crcObject.returnLoaders()
        self.trainDataLoader = trainGroup
        self.evalDataLoader = evalGroup
        self.testDataLoader = testGroup

    def train_dataloader(self):
        return self.trainDataLoader

    def val_dataloader(self):
        return self.evalDataLoader

    def test_dataloader(self):
        return self.testDataLoader
