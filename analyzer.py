import psutil
import time
import numpy as np
from sklearn.linear_model import LinearRegression

cpu_history = []
ram_history = []

model_cpu = LinearRegression()
model_ram = LinearRegression()