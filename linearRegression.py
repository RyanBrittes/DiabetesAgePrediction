import numpy as np
from loadData import LoadData
from lossClass import LossClass

class LinearRegression():
    def __init__(self):
        self.data = LoadData()
        self.loss = LossClass()
        self.shuffledData = self.data.get_shuffle_separe_train_test()
        self.xTrain = self.shuffledData[0]
        self.xTest = self.shuffledData[1]
        self.yTrain = self.shuffledData[2]
        self.yTest = self.shuffledData[3]
        self.weight = np.zeros(self.xTrain.shape[1])
        self.bias = 0
        self.lr = 0.003
        self.epochs = 8000
        self.batchSize = 30
        self.train_losses = []
        self.test_losses = []

    def trainModel(self):
        
        for epoch in range(self.epochs):

            for i in range(0, len(self.xTrain), self.batchSize):
                xBatch = self.xTrain[i:i+self.batchSize]
                yBatch = self.yTrain[i:i+self.batchSize]
                
                yPred = np.array(xBatch @ self.weight + self.bias).reshape(-1,1)
                error = yPred - yBatch
                
                dw = np.array((2/self.batchSize) * (xBatch.T @ error)).flatten()
                db = (2/self.batchSize) * np.sum(error)

                self.weight -= dw * self.lr
                self.bias -= db * self.lr

            train_pred = self.xTrain @ self.weight + self.bias
            test_pred = self.xTest @ self.weight + self.bias

            train_loss = self.loss.loss_MSE(self.yTrain, train_pred)
            test_loss = self.loss.loss_MSE(self.yTest, test_pred)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            print(f"Train Loss --> {train_loss:4f}\nTest Loss --> {test_loss:4f}")
            
        trainedParams = [self.weight, self.bias, self.train_losses, self.xTrain, self.xTest, self.yTrain, self.yTest]

        return trainedParams
    
    def showResults(self):
        results = self.trainModel()

        xTest = results[4]
        yTest = self.data.get_log_denormalize_list(results[6])
        trainedWeight = results[0]
        trainedBias = results[1]
        finalLoss = results[2]

        for i in range(len(xTest)):
            yPredict = self.data.get_log_denormalize_value(xTest[i] @ trainedWeight + trainedBias)
            print("--------------------------------------")
            print(f"Previsão: {yPredict} | Real: {yTest[i]}")
        
        print(f"Pesos encontrados: {trainedWeight}\nVies encontrado: {trainedBias}\nPerca final: {finalLoss[len(finalLoss) - 1]:4f}")

