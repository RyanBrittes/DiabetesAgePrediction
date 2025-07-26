import numpy as np

class LossClass():

    def loss_MSE(self, yTrue, yPred):
        return np.mean((yTrue - yPred) ** 2)
    
    def loss_MAE(self, yTrue, yPred):
        return np.mean(yTrue - yPred)
    
