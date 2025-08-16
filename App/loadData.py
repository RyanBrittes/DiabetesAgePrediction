import pandas as pd
from normalizeModel import NormalizeModel
import numpy as np

class LoadData():
    def __init__(self):
        self.norm = NormalizeModel()
        self.__data = pd.read_csv('files/diabetes.csv')
        self.__y_true = self.__data['Age'].values
        self.__x_true = self.__data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values
    
    def get_dataset(self):
        return self.__data

    def get_x_value(self):
        return self.__x_true

    def get_y_value(self):
        return self.__y_true

    def get_score_z(self, raw_value):
        return self.norm.calc_score_z(raw_value)

    def get_log(self, raw_value):
        return self.norm.calc_log_normalize(raw_value)

    def get_log_denormalize_list(self, rawValue):
        return self.norm.calc_log_denormalize_list(rawValue)
    
    def get_log_denormalize_value(self, rawValue):
        return self.norm.calc_log_denormalize(rawValue)

    def get_shuffle_separe_train_test(self, rate_test):
        x_values = self.get_log(self.__x_true)
        y_values = self.get_log(self.__y_true)

        np.random.seed(42)
        index_shuffled = np.random.permutation(len(y_values))

        y_shuffled = y_values[index_shuffled]
        x_shuffled = x_values[index_shuffled]
        
        rate_train = 1 - rate_test
        len_train = np.floor(len(y_values)*rate_train).astype(int)

        x_train = x_shuffled[0:len_train]
        y_train = y_shuffled[0:len_train]
        x_test = x_shuffled[len_train:-1]
        y_test = y_shuffled[len_train:-1]

        return [x_train, x_test, y_train, y_test]
