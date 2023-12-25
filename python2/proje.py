import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Veriyi oku
file_path = 'C:/Users/ilyas/OneDrive/Masaüstü/python2/.vscode/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

# Veriyi görselleştir
sns.set()

# Cinsiyet Dağılımı
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
sns.countplot(x='gender', data=df)

# Medeni Durum Dağılımı
plt.subplot(2, 3, 2)
sns.countplot(x='ever_married', data=df)

# Çalışma Türü Dağılımı
plt.subplot(2, 3, 3)
sns.countplot(x='work_type', data=df)

# İkamet Türü Dağılımı
plt.subplot(2, 3, 4)
sns.countplot(x='Residence_type', data=df)

# Sigara İçme Durumu Dağılımı
plt.subplot(2, 3, 5)
sns.countplot(x='smoking_status', data=df)

# Hipertansiyon Dağılımı
plt.subplot(2, 3, 6)
sns.countplot(x='hypertension', data=df)

plt.tight_layout()
plt.show()

# Veri temizleme ve dönüştürme
df = df.drop(columns=['ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1)
df.replace({'gender': {'Male': 0, 'Female': 1, 'Other': 2}}, inplace=True)

# Özellik ve hedef değişkenleri ayırma
X = df.drop(columns=['gender', 'hypertension', 'heart_disease', 'stroke'], axis=1)
Y_hypertension = df['hypertension']
Y_heartdisease = df['heart_disease']
Y_stroke = df['stroke']

# Veri ölçekleme
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)