
import os
import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
import pandas as pd

# Model ve etiket dosyalarının yolları
model_path = r"C:\Users\Hp\Desktop\traffic_sign_model.keras"
labels_csv_path = r"C:\Users\Hp\Desktop\Deneme\labels.csv"

# Etiketleri yükle ve temizle
def load_labels(labels_csv_path):
    df = pd.read_csv(labels_csv_path, encoding='latin1')
    class_names = df['Name'].apply(lambda x: x.encode('latin1').decode('utf-8')).tolist()
    return class_names

class_names = load_labels(labels_csv_path)

# Eğitilmiş modeli yükle
model = load_model(model_path)

# Görüntüyü modele uygun formata dönüştürme fonksiyonu
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Resmi yükle
    img = cv2.resize(img, (90, 90))  # Görüntüyü modele uygun boyuta getir
    img = img / 255.0  # Normalizasyon
    img = np.expand_dims(img, axis=0)  # Modelin beklediği 4 boyutlu tensöre dönüştür
    return img

# Tahmin yapma fonksiyonu
def predict_class(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Görüntüyü seç ve sonucu göster
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    # Görüntüyü göster
    image_label.config(text=f"Seçilen Görüntü: {os.path.basename(file_path)}")
    
    # Tahmini al ve sonucu göster
    predicted_class = predict_class(file_path)6
    result_label.config(text=f"Tahmin Edilen Sınıf: {predicted_class}")

# GUI arayüzü
app = tk.Tk()
app.title("Trafik İşareti Tanıma")
app.geometry("400x300")

# Arayüz elemanları
welcome_label = Label(app, text="Trafik İşareti Tanıma", font=("Arial", 16))
welcome_label.pack(pady=10)

image_label = Label(app, text="Görüntü seçilmedi", font=("Arial", 12))
image_label.pack(pady=5)

select_button = Button(app, text="Görüntü Seç", command=select_image)
select_button.pack(pady=10)

result_label = Label(app, text="Tahmin sonucu burada görünecek.", font=("Arial", 12))
result_label.pack(pady=20)

# Uygulamayı başlat
app.mainloop()
