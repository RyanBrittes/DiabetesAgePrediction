import numpy as np

class LossModel():

    def loss_MSE(self, y_true, y_predict):
        return np.mean((y_true - y_predict) ** 2)
    
    def loss_MAE(self, y_true, y_predict):
        return np.mean(y_true - y_predict)
    
