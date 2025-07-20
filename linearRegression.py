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
        self.epochs = 4000
        self.n = len(self.yTrue)
        self.losses = []
        self.batchSize = 300

    def funcMSE(self, yTrue, yPred):
        return np.mean(yTrue - yPred)

    def trainModel(self):
        for epoch in range(self.epochs):
            indexShuffled = np.random.permutation(self.n)
            yShuffled = self.yTrue[indexShuffled]
            xShuffled = self.xTrue[indexShuffled]

            for i in range(0, self.n, self.batchSize):
                xBatch = xShuffled[i:i+self.batchSize]
                yBatch = yShuffled[i:i+self.batchSize]

                yPred = xBatch @ self.weight + self.bias
                error = yPred - yBatch

                dw = (2/self.batchSize) * (xBatch.T @ error)
                db = (2/self.batchSize) * np.sum(error)

                self.weight -= dw * self.lr
                self.bias -= db * self.lr

            epoch_pred = self.xTrue @ self.weight + self.bias
            lossValue = self.funcMSE(self.yTrue, epoch_pred)
            self.losses.append(lossValue)
            #print(f"Loss --> {lossValue:4f}")

        print(f"Pesos encontrados: {self.weight}\nVies encontrado: {self.bias}\nPerca final: {lossValue}")
        xTest = [5, 165, 70, 35, 210, 35.6]
        yTest = xTest @ self.weight + self.bias
        print(f"\n\nTeste com valores reais:\nPregnancies: {xTest[0]}\nGlucose: {xTest[1]}\nBloodPressure: {xTest[2]}\nSkinThickness: {xTest[3]}\nInsulin: {xTest[4]}\nBMI: {xTest[5]}\nIdade de previsão para aderir a doença: {yTest}")

    def plotLoss(self):
        self.trainModel()
        plt.plot(self.losses)
        plt.xlabel("Época")
        plt.ylabel("Erro (MSE)")
        plt.title("Convergência da Regressão Linear Múltipla")
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
