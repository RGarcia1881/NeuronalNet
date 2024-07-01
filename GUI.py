import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Función para actualizar el estado del texto y el color
def update_status(status_text_id, new_text, new_color):
    # Actualiza el texto y el color de un elemento del canvas
    canvas.itemconfig(status_text_id, text=new_text, fill=new_color)

# Función para actualizar la barra de progreso
def update_progress_bar(progress):
    # Elimina la barra de progreso existente y crea una nueva con el progreso especificado
    canvas.delete("progress_bar")
    create_custom_progress_bar(canvas, x=150, y=650, width=800, height=20, progress=progress, bg_color="#e0f0ff", fg_color="#00bfff", tag="progress_bar")

# Función para generar datos simulados
def generate_data():
    num_days = 365  # Número de días para generar datos
    # Genera datos simulados
    data = {
        'Day': np.repeat(np.arange(num_days), 48),
        'Measurement': np.tile(np.arange(48), num_days),
        'Temperature': np.random.uniform(0, 35, num_days * 48),
        'Dissolved_Oxygen': np.random.uniform(0, 14, num_days * 48),
        'pH': np.random.uniform(6.5, 9, num_days * 48),
        'Dissolved_Solids': np.random.uniform(0, 5000, num_days * 48),
        'Suspended_Solids': np.random.uniform(0, 1000, num_days * 48)
    }
    df = pd.DataFrame(data)
    # Abre un cuadro de diálogo para guardar el archivo CSV
    csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if csv_path:
        df.to_csv(csv_path, index=False)
        messagebox.showinfo("Data Generated", f"Data has been saved to {csv_path}")
        # Actualiza el estado y la barra de progreso
        update_status(data_status_text, "Completado", "green")
        update_progress_bar(33)

# Función para entrenar el modelo
def train_model():
    # Abre un cuadro de diálogo para seleccionar un archivo CSV
    csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if csv_path:
        df = pd.read_csv(csv_path)
        np.random.seed(42)
        df['Potable'] = np.random.choice([0, 1], size=len(df))  # Añade una columna 'Potable' aleatoria
        daily_data = df.groupby('Day').mean()  # Agrupa los datos por día y calcula la media
        daily_data['Potable'] = daily_data['Potable'].round().astype(int)
        X = daily_data[['Temperature', 'Dissolved_Oxygen', 'pH', 'Dissolved_Solids', 'Suspended_Solids']]
        y = daily_data['Potable']
        # Divide los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Training Complete", f"Model trained with accuracy: {accuracy:.2f}")
        # Abre un cuadro de diálogo para guardar el modelo
        model_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("PKL files", "*.pkl")])
        if model_path:
            joblib.dump(mlp, model_path)
            messagebox.showinfo("Model Saved", f"Model has been saved to {model_path}")
            # Actualiza el estado y la barra de progreso
            update_status(training_status_text, "Completado", "green")
            update_progress_bar(66)

# Función para probar la calidad del agua
def test_water_quality():
    # Abre un cuadro de diálogo para seleccionar un archivo de modelo
    model_path = filedialog.askopenfilename(filetypes=[("PKL files", "*.pkl")])
    if model_path:
        mlp = joblib.load(model_path)
        # Genera datos de prueba simulados
        data = {
            'Temperature': np.random.uniform(0, 35, 48),
            'Dissolved_Oxygen': np.random.uniform(0, 14, 48),
            'pH': np.random.uniform(6.5, 9, 48),
            'Dissolved_Solids': np.random.uniform(0, 5000, 48),
            'Suspended_Solids': np.random.uniform(0, 1000, 48)
        }
        df = pd.DataFrame(data)
        potable = np.random.choice([0, 1])  # Genera un valor aleatorio de potabilidad
        df['Potable'] = potable
        # Abre un cuadro de diálogo para guardar el archivo de datos de prueba
        test_csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if test_csv_path:
            df.to_csv(test_csv_path, index=False)
            test_day_data = df.mean().to_frame().T
            test_day_data = test_day_data[['Temperature', 'Dissolved_Oxygen', 'pH', 'Dissolved_Solids', 'Suspended_Solids']]
            test_day_prediction = mlp.predict(test_day_data)
            result = "Potable" if test_day_prediction[0] == 1 else "Not Potable"
            messagebox.showinfo("Test Result", f"The water quality is: {result}\nExpected result: {'Potable' if potable == 1 else 'Not Potable'}")
            # Actualiza el estado y la barra de progreso
            update_status(simulation_status_text, "Completado", "green")
            update_status(simulation_result_text, "POTABLE" if result == "Potable" else "NO POTABLE", "green" if result == "Potable" else "red")
            update_progress_bar(100)

# Crear la ventana principal
root = tk.Tk()
root.title("Calidad del Agua")
root.geometry("1100x700")
root.configure(bg="white")

# Cargar la imagen de fondo
image_path = "drop.png"  # Reemplaza con la ruta a tu imagen
bg_image = Image.open(image_path)

# Redimensionar la imagen al tamaño deseado
new_width = 400
new_height = int(bg_image.height * (new_width / bg_image.width))
bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)

# Convertir la imagen a PhotoImage
bg_image = ImageTk.PhotoImage(bg_image)

# Crear un Canvas para mostrar la imagen de fondo
canvas = tk.Canvas(root, bg="white", width=1100, height=700, highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Mostrar la imagen de fondo
canvas.create_image(550, 350, image=bg_image, anchor="center")

# Título
canvas.create_text(550, 50, text="Calidad del Agua", font=("Helvetica", 32, "bold"), fill="#0E8CFF", anchor="center")

# Calidad de la simulación
canvas.create_text(100, 150, text="Calidad de la simulación:", font=("Helvetica", 20), fill="gray", anchor="w")

# Resultado de la simulación
simulation_result_text = canvas.create_text(550, 250, text="-----", font=("Helvetica", 48, "bold"), fill="gray", anchor="center")

# Estado de datos generados
canvas.create_text(100, 400, text="Datos Generados:", font=("Helvetica", 20), fill="gray", anchor="w")
data_status_text = canvas.create_text(350, 400, text="No completado", font=("Helvetica", 20), fill="red", anchor="w")

# Estado del entrenamiento
canvas.create_text(100, 450, text="Entrenamiento:", font=("Helvetica", 20), fill="gray", anchor="w")
training_status_text = canvas.create_text(350, 450, text="No completado", font=("Helvetica", 20), fill="red", anchor="w")

# Estado de la simulación
canvas.create_text(100, 500, text="Simulación:", font=("Helvetica", 20), fill="gray", anchor="w")
simulation_status_text = canvas.create_text(350, 500, text="No completado", font=("Helvetica", 20), fill="red", anchor="w")

# Crear botones redondeados utilizando imágenes
def create_rounded_button(text, command, x, y):
    # Crear un botón redondeado con una imagen
    btn_img = Image.open("rounded_button.png")  # Reemplaza con la ruta a tu imagen de botón redondeado
    btn_img = btn_img.resize((150, 30), Image.LANCZOS)
    btn_img_tk = ImageTk.PhotoImage(btn_img)
    button = tk.Button(root, text=text, font=("Helvetica", 16), fg="white", bg="white", command=command, image=btn_img_tk, compound="center", borderwidth=0)
    button.image = btn_img_tk  # Necesario para evitar que la imagen sea eliminada por el recolector de basura
    button.place(x=x, y=y, width=150, height=30)
    return button

# Función para crear una barra de progreso personalizada
def create_custom_progress_bar(canvas, x, y, width, height, progress, bg_color, fg_color, tag):
    # Dibujar el fondo de la barra de progreso
    canvas.create_oval(x, y, x + height, y + height, fill=bg_color, outline=bg_color, tag=tag)
    canvas.create_oval(x + width - height, y, x + width, y + height, fill=bg_color, outline=bg_color, tag=tag)
    canvas.create_rectangle(x + height / 2, y, x + width - height / 2, y + height, fill=bg_color, outline=bg_color, tag=tag)
    
    # Dibujar el progreso de la barra
    progress_width = width * (progress / 100)
    canvas.create_oval(x, y, x + height, y + height, fill=fg_color, outline=fg_color, tag=tag)
    canvas.create_oval(x + progress_width - height, y, x + progress_width, y + height, fill=fg_color, outline=fg_color, tag=tag)
    canvas.create_rectangle(x + height / 2, y, x + progress_width - height / 2, y + height, fill=fg_color, outline=fg_color, tag=tag)

# Crear botones y asignar funciones a cada uno
generate_button = create_rounded_button("Generar Datos", generate_data, 800, 400)
train_button = create_rounded_button("Entrenar IA", train_model, 800, 450)
simulate_button = create_rounded_button("Simulación", test_water_quality, 800, 500)

# Etiqueta del proceso
canvas.create_text(100, 620, text="Proceso:", font=("Helvetica", 11), fill="gray", anchor="w")

# Iniciar con una barra de progreso vacía
create_custom_progress_bar(canvas, x=150, y=650, width=800, height=20, progress=0, bg_color="#e0f0ff", fg_color="#00bfff", tag="progress_bar")

# Iniciar el bucle principal de la GUI
root.mainloop()