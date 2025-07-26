import pandas as pd
import os
from dotenv import load_dotenv
from normalizeClass import NormalizeClass
import numpy as np

load_dotenv()

class LoadData():
    def __init__(self):
        self.norm = NormalizeClass()
        self.data = pd.read_csv(os.getenv("DATAPATH"))
        self.yTrue = self.data['Age'].values
        self.xTrue = self.data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values
        self.rateTest = 0.1
    
    def get_Y_scoreZ(self):
        return self.norm.scoreZFunction(self.yTrue)
    
    def get_X_scoreZ(self):
        return self.norm.scoreZFunction(self.xTrue)

    def get_Y_log(self):
        return self.norm.logNormalizeFunction(self.yTrue)

    def get_X_log(self):
        return self.norm.logNormalizeFunction(self.xTrue)

    def get_log_denormalize_list(self, rawValue):
        return self.norm.logDenormalizeList(rawValue)
    
    def get_log_denormalize_value(self, rawValue):
        return self.norm.logDenormalizeValue(rawValue)

    def get_shuffle_separe_train_test(self):
        xValues = self.get_X_log()
        yValues = self.get_Y_log()

        np.random.seed(42)
        indexShuffled = np.random.permutation(len(yValues))

        yShuffled = yValues[indexShuffled]
        xShuffled = xValues[indexShuffled]
        
        rateTrain = 1 - self.rateTest
        lenTrain = np.floor(len(yValues)*rateTrain).astype(int)

        xTrain = xShuffled[0:lenTrain]
        yTrain = yShuffled[0:lenTrain]
        xTest = xShuffled[lenTrain:len(xValues)]
        yTest = yShuffled[lenTrain:len(xValues)]

        listSepared = [xTrain, xTest, yTrain, yTest]

        return listSepared

