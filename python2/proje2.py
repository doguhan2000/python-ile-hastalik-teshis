import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Veriyi oku
file_path = 'C:/Users/ilyas/OneDrive/Masaüstü/python2/.vscode/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Veriyi temizleme ve dönüştürme
df = df.drop(columns=['ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1)
df.replace({'gender': {'Male': 0, 'Female': 1, 'Other': 2}}, inplace=True)

# NaN değerleri doldurmak için SimpleImputer kullanma
imputer = SimpleImputer(strategy='mean')  # veya 'median', 'most_frequent' gibi stratejileri seçebilirsiniz
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # Sütun isimlerini koruma

# Özellik ve hedef değişkenleri ayırma
X = df_imputed.drop(columns=['hypertension'], axis=1)  # 'hypertension' sütunu
Y_hypertension = df_imputed['hypertension']

# Veri ölçekleme
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X_standard, Y_hypertension, test_size=0.2, stratify=Y_hypertension, random_state=2)

# SVM Modeli oluşturma ve eğitme
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Eğitim setinde doğruluk hesaplama
train_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
print("Eğitim seti doğruluğu:", train_data_accuracy*100)

# Test setinde doğruluk hesaplama
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)
print("Test seti doğruluğu:", test_data_accuracy*100)