from linearRegression import LinearRegression
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from loadData import LoadData
from normalizeModel import NormalizeModel

class PlotGraphic():
    def __init__(self):
        self.model = LinearRegression()
        self.data = LoadData()
        self.normalize = NormalizeModel()
    
    def plot_loss(self):
        list_loss = self.model.train_model()
        plt.plot(self.normalize.calc_log_denormalize_list(list_loss[2]))
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("Perca por período de treinamento")
        plt.grid()
        plt.show()

    def plot_relations(self):
        index = 0

        for i in range(len(self.data.__x_true[0])):
            plt.plot(self.data.__x_true[:,index], self.data.__y_true, '.', )
            plt.grid()
            plt.ylabel("Age")
            plt.xlabel(f"{self.data.__data.columns[index]}")
            plt.title(f"Relation between Age and {self.data.__data.columns[index]}")
            plt.show()
            index += 1

A = PlotGraphic()

A.plot_loss()