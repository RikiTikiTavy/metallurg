import sys


print("версия Python: {}".format(sys.version))
import pandas as pd
print("версия pandas: {}".format(pd.__version__))
import matplotlib
print("версия matplotlib: {}".format(matplotlib.__version__))
import numpy as np
print("версия NumPy: {}".format(np.__version__))
import scipy as sp
print("версия SciPy: {}".format(sp.__version__))
import IPython
print("версия IPython: {}".format(IPython.__version__))
import sklearn
print("версия scikit-learn: {}".format(sklearn.__version__))

import matplotlib.pyplot as plt
import mglearn
from IPython.display import display
plt.rc('font', family='Verdana')
#plt.show

################

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys {}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'])