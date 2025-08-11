from linearRegression import LinearRegression
from normalizeModel import NormalizeModel

linear_regression = LinearRegression()
normalize = NormalizeModel()

results = linear_regression.train_model()
x_test = results[4]
y_test = normalize.calc_log_denormalize_list(results[6])
trained_weight = results[0]
trained_bias = results[1]
final_loss = results[2]

for i in range(len(x_test)):
    y_predict = normalize.calc_log_denormalize(x_test[i] @ trained_weight + trained_bias)
    print("--------------------------------------")
    print(f"Previsão: {y_predict} | Real: {y_test[i]}")

print(f"Pesos encontrados: {trained_weight}\nVies encontrado: {trained_bias}\nPerca final: {final_loss[-1]:4f}")