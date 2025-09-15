# 🌸 Clasificación de Especies de Iris

Proyecto académico que compara dos modelos, **Regresión Lineal (adaptada)** y **Regresión Logística**, para clasificar las especies de flores del famoso dataset Iris. El análisis se enfoca intencionalmente en las características del **sépalo**, que presentan un mayor desafío para la clasificación.

---

## 🚀 Características Principales

* Carga y exploración del dataset `Iris.csv`.
* **Análisis Visual Exploratorio (EDA)** para entender la distribución de los datos.
* Entrenamiento de dos modelos: `LinearRegression` y `LogisticRegression`.
* Comparación de rendimiento basada en la métrica de **Accuracy (Precisión)**.
* Generación de visualizaciones clave para interpretar los resultados:
    * Gráfico de dispersión.
    * Matriz de confusión.

---

## 📂 Estructura del Repositorio

```
📦 Clasificacion-Iris
 ┣ 📂 Dataset
 ┃ ┗ 📜 Iris.csv
 ┣ 📂 Graficos
 ┃ ┣ 📊 1_dispersion_sepalos.png
 ┃ ┣ 📊 2_matriz_confusion.png
 ┣ 📂 Informe
 ┃ ┣ 📂 pdf
 ┃ ┃ ┗ 📜 Informe_final.pdf
 ┃ ┣ 📜 main.tex
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🛠️ Requisitos

Para ejecutar este proyecto, necesitas tener Python instalado. Luego, instala las dependencias necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

### 📦 Librerías Principales

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

---

## ▶️ Ejecución

Para entrenar los modelos y generar los gráficos, ejecuta el script principal desde tu terminal:

```bash
python analisis_iris.py
```

Esto realizará dos acciones:
1.  Imprimirá las métricas de precisión de ambos modelos en la consola.
2.  Guardará automáticamente los dos gráficos generados en la carpeta `Graficos/`.

---

## 📊 Gráficos y Resultados

### 🔹 1. Dispersión por Largo y Ancho del Sépalo
Este gráfico muestra la distribución de las tres especies de Iris usando solo las medidas del sépalo. Se puede observar una superposición significativa entre *Iris-versicolor* e *Iris-virginica*, lo que explica por qué el problema es más desafiante que si se usaran las medidas del pétalo.

### 🔹 2. Matriz de Confusión
La matriz de confusión visualiza el rendimiento del modelo de Regresión Lineal adaptado. Muestra cuántas predicciones fueron correctas (la diagonal) y en qué clases se equivocó el modelo.

---

## 📈 Métricas de Rendimiento

Los resultados obtenidos al ejecutar el script son los siguientes (pueden variar ligeramente por la aleatoriedad en la división de datos):

```
📊 Precisión comparativa:
 - Regresión Lineal: 86.67%
 - Regresión Logística: 90.00%
```

---

## 📌 Conclusiones

* La **Regresión Logística** supera a la Regresión Lineal adaptada para esta tarea de clasificación, demostrando ser un modelo más adecuado.
* El uso exclusivo de las características del **sépalo** reduce la precisión en comparación con el uso de las características del pétalo, creando un escenario de prueba más realista y desafiante.
* El análisis visual es fundamental para comprender tanto la naturaleza de los datos como el rendimiento y los errores específicos de los modelos.

---

## ✍️ Autor

Proyecto desarrollado como actividad académica en **Sistemas e Ingeniería**.
