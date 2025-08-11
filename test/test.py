import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from App.plotGraphic import PlotGraphic
from App.linearRegression import LinearRegression

A = PlotGraphic()
B = LinearRegression()

A.plot_loss()
#B.showResults() --> another example