import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from App.plotGraphic import PlotGraphic

A = PlotGraphic()

A.plot_loss()
