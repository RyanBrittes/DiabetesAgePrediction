import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from normalizeClass import NormalizeClass

load_dotenv()

class LoadData():
    def __init__(self):
        self.norm = NormalizeClass()
        self.data = pd.read_csv(os.getenv("DATAPATH"))
        self.yTrue = self.data['Age'].values
        self.xTrue = self.data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values
    
    def get_Y_scoreZ(self):
        return self.norm.scoreZFunction(self.yTrue)

    def get_Y_log(self):
        return self.norm.logNormalizeFunction(self.yTrue)
    
    def get_X_scoreZ(self):
        return self.norm.scoreZFunction(self.xTrue)

    def get_X_log(self):
        return self.norm.logNormalizeFunction(self.xTrue)

    def getX(self):
        x = self.get_X_scoreZ()
        print(x)
