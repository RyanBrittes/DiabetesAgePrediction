import numpy as np

class NormalizeClass():

    def meanFunction(self, rawValue):
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += rawValue[i]

        return sumValue/len(rawValue)

    def standardDeviationFunction(self, rawValue):
        mean = self.meanFunction(rawValue)
        sumValue = 0
        for i in range(len(rawValue)):
            sumValue += (rawValue[i] - mean) ** 2
        
        return (sumValue / len(rawValue)) ** 0.5

    def scoreZFunction(self, rawValue):
        meanValue = self.meanFunction(rawValue)
        stdValue = self.standardDeviationFunction(rawValue)
        listValue = []

        for i in range(len(rawValue)):
            value = (rawValue[i] - meanValue) / stdValue
            listValue.append(value)
        
        return np.vstack(listValue)

    def logNormalizeFunction(self, rawValue):
        listValue = []
        const_not_zero = 1e-8
        for i in range(len(rawValue)):
            listValue.append(np.log(rawValue[i] + const_not_zero))
        return np.vstack(listValue)