import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict():
    try:
        vals = [float(entry.get()) for entry in entries]
        prediction = model.predict([vals])[0]
        result_label.config(text=f"Predicted Iris Type: {prediction}", fg="blue")
    except:
        messagebox.showerror("Invalid input", "Please enter valid numerical values.")

root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("400x450")
root.configure(bg="#e8f4fa")

title = tk.Label(root, text="🌸 Iris Flower Classifier", font=("Segoe UI", 16, "bold"), bg="#e8f4fa", fg="#0d47a1")
title.pack(pady=20)

labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
entries = []

for label in labels:
    lbl = tk.Label(root, text=label, bg="#e8f4fa", font=("Segoe UI", 12))
    lbl.pack()
    ent = tk.Entry(root, font=("Segoe UI", 12))
    ent.pack(pady=5)
    entries.append(ent)

btn = tk.Button(root, text="Predict Flower Type", command=predict, bg="#0d47a1", fg="white", font=("Segoe UI", 12))
btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Segoe UI", 14, "bold"), bg="#e8f4fa")
result_label.pack(pady=10)

root.mainloop()
