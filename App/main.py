from linearRegression import LinearRegression
from normalizeModel import NormalizeModel
import math

linear_regression = LinearRegression()
normalize = NormalizeModel()

print("Processando informações...")

results = linear_regression.train_model()
x_test = results[4]
y_test = normalize.calc_log_denormalize_list(results[6])
trained_weight = results[0]
trained_bias = results[1]
final_loss = normalize.calc_log_denormalize_list(results[2])

for i in range(len(x_test)):
    y_predict = normalize.calc_log_denormalize(x_test[i] @ trained_weight + trained_bias)
    print("--------------------------------------")
    print(f"Pregnancies: {math.ceil(normalize.calc_log_denormalize(x_test[i][0]))}\nGlucose: {math.ceil(normalize.calc_log_denormalize(x_test[i][1]))}\nBloodPressure: {math.ceil(normalize.calc_log_denormalize(x_test[i][2]))}\nSkinThickness: {math.ceil(normalize.calc_log_denormalize(x_test[i][3]))}\nInsulin: {math.ceil(normalize.calc_log_denormalize(x_test[i][4]))}\nBMI: {math.ceil(normalize.calc_log_denormalize(x_test[i][5]))}")
    print(f"Previsão: {y_predict} | Real: {y_test[i]}")

print("--------------------------------------")
print(f"Pesos encontrados: {trained_weight}\nVies encontrado: {trained_bias}\nPerca final: {final_loss[-1]:4f}")
