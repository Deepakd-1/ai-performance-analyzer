import psutil
import time
import numpy as np
from sklearn.linear_model import LinearRegression

cpu_history = []
ram_history = []

model_cpu = LinearRegression()
model_ram = LinearRegression()

def predict_future(history, model):
    if len(history) < 5:
        return None  # Not enough data
    
    X = np.arange(len(history)).reshape(-1, 1)
    y = np.array(history)

    model.fit(X, y)
    future = model.predict([[len(history) + 5]])  # Predict 5 sec ahead
    return round(future[0], 2)

def get_suggestions(cpu, ram):
    suggestions = []
    if cpu > 70:
        suggestions.append("High CPU detected → Close heavy apps or background tasks")
    if ram > 80:
        suggestions.append("High RAM usage → Consider upgrading RAM or closing unused apps")
    if not suggestions:
        suggestions.append("System running normally")
    return suggestions