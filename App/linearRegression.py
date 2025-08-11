import numpy as np
from loadData import LoadData
from lossModel import LossModel

class LinearRegression():
    def __init__(self):
        self.data = LoadData()
        self.loss = LossModel()
        self.rate_test = 0.1
        self.shuffled_data = self.data.get_shuffle_separe_train_test(self.rate_test)
        self.x_train = self.shuffled_data[0]
        self.x_test = self.shuffled_data[1]
        self.y_train = self.shuffled_data[2]
        self.y_test = self.shuffled_data[3]
        self.weight = np.zeros(self.x_train.shape[1])
        self.bias = 0
        self.lr = 0.003
        self.epochs = 8000
        self.batch_size = 30
        self.train_losses = []
        self.test_losses = []

    def train_model(self):
        
        for epoch in range(self.epochs):

            for i in range(0, len(self.x_train), self.batch_size):
                x_batch = self.x_train[i:i+self.batch_size]
                y_batch = self.y_train[i:i+self.batch_size]
                
                y_predict = np.array(x_batch @ self.weight + self.bias).reshape(-1,1)
                error = y_predict - y_batch
                
                dw = np.array((2/self.batch_size) * (x_batch.T @ error)).flatten()
                db = (2/self.batch_size) * np.sum(error)

                self.weight -= dw * self.lr
                self.bias -= db * self.lr

            train_pred = self.x_train @ self.weight + self.bias
            test_pred = self.x_test @ self.weight + self.bias

            train_loss = self.loss.loss_MSE(self.y_train, train_pred)
            test_loss = self.loss.loss_MSE(self.y_test, test_pred)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            print(f"Train Loss --> {train_loss:4f}\nTest Loss --> {test_loss:4f}")
            
        return [self.weight, self.bias, self.train_losses, self.x_train, self.x_test, self.y_train, self.y_test]
    
    def show_results(self):
        results = self.train_model()

        x_test = results[4]
        y_test = self.data.get_log_denormalize_list(results[6])
        trained_weight = results[0]
        trained_bias = results[1]
        final_loss = results[2]

        for i in range(len(x_test)):
            y_predict = self.data.get_log_denormalize_value(x_test[i] @ trained_weight + trained_bias)
            print("--------------------------------------")
            print(f"Previsão: {y_predict} | Real: {y_test[i]}")
        
        print(f"Pesos encontrados: {trained_weight}\nVies encontrado: {trained_bias}\nPerca final: {final_loss[-1]:4f}")

