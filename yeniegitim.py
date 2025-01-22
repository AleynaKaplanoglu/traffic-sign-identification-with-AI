import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Etiketler ve veriler için dosya yolları
labels_csv_path = r"C:\Users\Hp\Desktop\Deneme\labels.csv"  # Etiket dosyasının yolu
inputBasePath = r"C:\Users\Hp\Desktop\trafikisaretleri"  # Veri kümesinin ana yolu 
#Kullanılan veriseti: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Etiketleri yükle
class_names = pd.read_csv(labels_csv_path, encoding='latin1')['Name'].tolist()

# Görselleri yüklemek için fonksiyon
def fetch_images(class_folders, base_path, folder):
    images = []
    labels = []
    
    for class_folder in class_folders:
        class_path = os.path.join(base_path, folder, str(class_folder))
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (90, 90))  # Görselleri 90x90 boyutuna yeniden boyutlandırıyoruz
                images.append(img)
                labels.append(class_folder)

    return np.array(images), np.array(labels)

# Eğitim ve doğrulama verilerini yükle
class_folders = os.listdir(os.path.join(inputBasePath, 'Train'))
X, y = fetch_images(class_folders, inputBasePath, 'Train')

# Veriyi normalize et
X = X / 255.0  # [0, 255] arası değerleri [0, 1] aralığına getirme
y = np.array(y)

# Etiketleri one-hot encoding yapmak için
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=len(class_folders))

# Eğitim ve test verilerini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(90, 90, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(232, activation='relu'),
    Dense(116, activation='relu'),
    Dense(len(class_folders), activation='softmax')  # Çıkış katmanı, sınıf sayısına göre
])

# Modeli derle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin kaydedileceği dosya yolu
model_path = r"C:\Users\Hp\Desktop\traffic_sign_model.keras"  # .h5 yerine .keras kullanıyoruz

# Model checkpoint callback'ini tanımla
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Modeli eğit
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Modelin en iyi versiyonunu kaydet
model.save(model_path)

print("Model başarıyla eğitildi ve kaydedildi.")
