import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos
try:
    data = pd.read_csv("Iris.csv")
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print("Error: ¡No se encontró el archivo 'Iris.csv' en la carpeta!")
    exit()

# -------------------------------------------------------------------------
# GRÁFICO EXPLORATORIO 
# -------------------------------------------------------------------------
print("\nGenerando gráfico de dispersión por Sépalos...")
plt.figure(figsize=(10, 7))

# Usamos las características del SÉPALO 
sns.scatterplot(data=data, x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', s=100) 

plt.title('Dispersión por Largo y Ancho del Sépalo', fontsize=16) 
plt.xlabel('Largo del Sépalo (cm)') 
plt.grid(True)
plt.legend(title='Especie')
plt.show()
# -------------------------------------------------------------------------

# 2. Preparar variables
features = ['SepalLengthCm', 'SepalWidthCm']
X = data[features]

# Codificar 'Species' como números (0,1,2)
le = LabelEncoder()
y = le.fit_transform(data['Species'])

# 3. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar modelos
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

modelo_logistico = LogisticRegression(max_iter=200)
modelo_logistico.fit(X_train, y_train)

print("\n✅ Modelos entrenados.")

# 5. Predicciones con Regresión Lineal
predicciones_continuas = modelo_lineal.predict(X_test)
predicciones_clases = np.round(predicciones_continuas)
predicciones_clases = np.clip(predicciones_clases, 0, 2).astype(int)

# 6. Predicciones con Regresión Logística
predicciones_logisticas = modelo_logistico.predict(X_test)

# 7. Evaluar precisión
precision_lineal = accuracy_score(y_test, predicciones_clases)
precision_logistica = accuracy_score(y_test, predicciones_logisticas)

print(f"\n📊 Precisión comparativa:")
print(f" - Regresión Lineal: {precision_lineal * 100:.2f}%")
print(f" - Regresión Logística: {precision_logistica * 100:.2f}%")

# -------------------------------------------------------------------------
# MATRIZ DE CONFUSIÓN
# -------------------------------------------------------------------------
cm_lineal = confusion_matrix(y_test, predicciones_clases)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_lineal, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)

plt.title('Matriz de Confusión - Regresión Lineal (con Sépalos)')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()
