from sys import path
path.append("./../")

from utils import plot_curves

curve1 = [x**2 for x in range(10)]
curve2 = [2*x**2 + 1 for x in range(10)]


plot_curves(curve_1=curve1, label_1="squared", show=True, fig_name="./../images/test1")


plot_curves(curve_1=curve1, label_1="squared", curve_2=curve2, label_2 = "sqdksjaf", fig_name="./../images/test2", show=True)