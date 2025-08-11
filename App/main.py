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
    print(f"Pregnancies: {normalize.calc_log_denormalize(x_test[i][0])}\nGlucose: {normalize.calc_log_denormalize(x_test[i][1])}\nBloodPressure: {normalize.calc_log_denormalize(x_test[i][2])}\nSkinThickness: {normalize.calc_log_denormalize(x_test[i][3])}\nInsulin: {normalize.calc_log_denormalize(x_test[i][4])}\nBMI: {normalize.calc_log_denormalize(x_test[i][5])}")
    print(f"Previsão: {y_predict} | Real: {y_test[i]}")

print("--------------------------------------")
print(f"Pesos encontrados: {trained_weight}\nVies encontrado: {trained_bias}\nPerca final: {normalize.calc_log_denormalize(final_loss[-1]):4f}")
