import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linearRegression import LinearRegression

A = LinearRegression()

listValue = A.shuffleData()

separe = A.separeTrainTest(listValue[0], listValue[1])

print(A.trainModel())
