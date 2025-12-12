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

print("Starting AI-powered OS Performance Analyzer...\n")

while True:
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    
    cpu_history.append(cpu)
    ram_history.append(ram)

    cpu_forecast = predict_future(cpu_history, model_cpu)
    ram_forecast = predict_future(ram_history, model_ram)


    print(f"CPU Usage: {cpu}% | RAM Usage: {ram}%")
    
    high_procs = sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']),
                        key=lambda p: p.info['cpu_percent'],
                        reverse=True)[:5]

    print("\nTop Perfomance:")
    for p in high_procs:
        print(f"PID {p.info['pid']} | {p.info['name']} | CPU {p.info['cpu_percent']}% | RAM {p.info['memory_percent']:.2f}%")


    print("\nSuggestions:")
    for s in get_suggestions(cpu, ram):
        print("- " + s)

    if cpu_forecast:
        print(f"\nAI Prediction: CPU in 5 sec → {cpu_forecast}%")
    if ram_forecast:
        print(f"AI Prediction: RAM in 5 sec → {ram_forecast}%")

    print("\n--------------------------------------\n")
    time.sleep(3)