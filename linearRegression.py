import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self):
        self.read = pd.read_csv('/home/ryan/Documents/Python/AI/DiabetesAgePrediction/files/diabetes.csv')
        self.data = self.read[self.read['Outcome'] == 1]
        self.yTrue = self.data['Age'].values
        self.xTrue = self.data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values
        self.weight = np.zeros(self.xTrue.shape[1])
        self.bias = 0
        self.lr = 0.00001
        self.epochs = 10000000
        self.batchSize = 100
        self.n = len(self.yTrue)
        self.losses = []
        self.rateTest = 0.1

    def funcMSE(self, yTrue, yPred):
        return np.mean((yTrue - yPred) ** 2)
    
    def funcMSA(self, yTrue, yPred):
        return np.mean(yTrue - yPred)
    
    def shuffleData(self):
        np.random.seed(42)
        indexShuffled = np.random.permutation(self.n)

        yShuffled = self.yTrue[indexShuffled]
        xShuffled = self.xTrue[indexShuffled]
        
        listShuffled = [xShuffled, yShuffled]
        
        return listShuffled

    def separeTrainTest(self, xValues, yValues):
        rateTrain = 1 - self.rateTest
        lenTrain = np.floor(len(xValues)*rateTrain).astype(int)

        xTrain = xValues[0:lenTrain]
        yTrain = yValues[0:lenTrain]
        xTest = xValues[lenTrain + 1:len(xValues)]
        yTest = yValues[lenTrain + 1:len(xValues)]

        listSepared = [xTrain, xTest, yTrain, yTest]

        return listSepared

    def trainModel(self):
        xTrain, xTest, yTrain, yTest = self.separeTrainTest(self.shuffleData()[0], self.shuffleData()[1])
        
        for epoch in range(self.epochs):

            for i in range(0, self.n, self.batchSize):
                xBatch = xTrain[i:i+self.batchSize]
                yBatch = yTrain[i:i+self.batchSize]

                yPred = xBatch @ self.weight + self.bias
                error = yPred - yBatch

                dw = (2/self.batchSize) * (xBatch.T @ error)
                db = (2/self.batchSize) * np.sum(error)

                self.weight -= dw * self.lr
                self.bias -= db * self.lr

            epoch_pred = self.xTrue @ self.weight + self.bias
            lossValue = self.funcMSE(self.yTrue, epoch_pred)
            self.losses.append(lossValue)
            print(f"Loss --> {lossValue:4f}")
            
        trainedParams = [self.weight, self.bias, self.losses, xTrain, xTest, yTrain, yTest]

        return trainedParams
    
    def showResults(self):
        results = self.trainModel()

        xTest = results[4]
        yTest = results[6]
        trainedWeight = results[0]
        trainedBias = results[1]
        finalLoss = results[2]

        for i in range(len(xTest)):
            yPredict = xTest[i] @ trainedWeight + trainedBias
            print("--------------------------------------")
            print(f"Pregnancies: {xTest[i][0]}\nGlucose: {xTest[i][1]}\nBloodPressure: {xTest[i][2]}\nSkinThickness: {xTest[i][3]}\nInsulin: {xTest[i][4]}\nBMI: {xTest[i][5]}")
            print(f"Previsão: {yPredict} | Real: {yTest[i]}")
        
        print(f"Pesos encontrados: {trainedWeight}\nVies encontrado: {trainedBias}\nPerca final: {finalLoss[len(finalLoss) - 1]:4f}")


    def plotLoss(self):
        self.showResults()
        plt.plot(self.losses)
        plt.xlabel("Época")
        plt.ylabel("Erro (MSE)")
        plt.title("Perca por período de treinamento")
        plt.grid()
        plt.show()
    
    def plotDataRelation(self):
        index = 0

        for i in range(len(self.xTrue[0])):
            plt.plot(self.xTrue[:,index], self.yTrue, '.', )
            plt.grid()
            plt.ylabel("Age")
            plt.xlabel(f"{self.data.columns[index]}")
            plt.title(f"Relation between Age and {self.data.columns[index]}")
            plt.show()
            index += 1
