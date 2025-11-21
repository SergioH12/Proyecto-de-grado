# Proyecto de Grado — Clasificación de Citologías (HSIL / LSIL / NEGATIVO)

Este repositorio contiene un notebook (PROYECTO_GRADO.ipynb) que implementa y entrena una red neuronal convolucional (CNN) para la clasificación de imágenes citológicas en 3 clases: HSIL, LSIL y NEGATIVO. El código está escrito en Python usando TensorFlow / Keras y utilidades de scikit-learn para evaluación.

Contenido principal
- PROYECTO_GRADO.ipynb — notebook con todo el flujo: importación de librerías, preparación de generadores, definición del modelo, entrenamiento, carga del mejor modelo y evaluación (matriz de confusión, reporte de clasificación, curvas ROC).
- best_model.keras — archivo guardado durante el entrenamiento (si se ejecuta localmente).
- Estructura esperada de datos (ver sección "Datos").
- PREDICCION.ipynb - notebook con la evaluación del modelo con un dataset de test

Resumen rápido de lo que hace el notebook
- Lee imágenes desde carpetas organizadas por clases usando ImageDataGenerator (modo `grayscale`).
- Aplica aumentos (rotación, shifts, zoom, flip) en el generador de entrenamiento.
- Define una CNN secuencial con varias capas Conv2D, MaxPooling, Flatten y Dense.
- Compila y entrena el modelo, guardando el mejor checkpoint en `best_model.keras`.
- Evalúa el modelo en el conjunto de validación: genera predicciones, matriz de confusión, reporte de clasificación y curvas ROC/AUC.

Requisitos (ejemplo)
- Python 3.8+
- TensorFlow (2.x)
- scikit-learn
- opencv-python
- pandas
- numpy
- matplotlib
- seaborn
- jupyter / notebook

Instalación rápida (ejemplo usando pip)
```bash
python -m venv .venv
source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install --upgrade pip
pip install tensorflow scikit-learn opencv-python pandas numpy matplotlib seaborn jupyter
```

Estructura de datos esperada
La notebook usa un `base_path` (ruta absoluta) y espera las siguientes carpetas:
- {base_path}/train/<CLASE>/*.jpg (o png)
- {base_path}/val/<CLASE>/*.jpg
- {base_path}/REAL/  (opcional, se usa en el notebook para pruebas reales)

Donde `<CLASE>` son subcarpetas con nombres de clase (por ejemplo: HSIL, LSIL, NEGATIVO). En el notebook original la variable `base_path` está configurada como:
```python
base_path = r\"C:\\Users\\user\\Documents\\AccidentsDataset\"\n"
```
Asegúrate de actualizar `base_path` a la ruta en tu sistema.

Cómo ejecutar
1. Abrir el notebook `PROYECTO_GRADO.ipynb` con Jupyter Notebook o JupyterLab.
2. Ajustar `base_path`, `img_height`, `img_width`, `batch_size` y `epochs` según tus recursos.
3. Ejecutar las celdas en orden. El entrenamiento guardará el mejor modelo en `best_model.keras` (ruta relativa donde se ejecute el notebook).

Notas de evaluación y observaciones importantes
- El notebook muestra un historial de entrenamiento con accuracy de entrenamiento que crece (hasta ~0.70) mientras que el reporte de clasificación en validación muestra una accuracy global menor (~0.37). Esto sugiere posible sobreajuste y/o desequilibrio entre clases.
- Se utiliza `color_mode='grayscale'`. Si se desea usar modelos preentrenados en RGB (por ejemplo VGG16), convierta las imágenes a 3 canales o adapte la entrada.

  ```
- La arquitectura actual tiene una capa Dense muy grande (512) tras un Flatten que produce muchos parámetros. Para reducir sobreajuste se recomienda:
  - Reemplazar Flatten() + Dense(...) por GlobalAveragePooling2D() seguido de Dense con menos neuronas.
  - Usar regularización (L2), aumentar dropout o disminuir tamaño de la capa densa.
  - Probar transfer learning con VGG16 / MobileNet (con peso `imagenet`) y fine-tuning.
  - Balancear el dataset o usar `class_weight` al entrenar.
  - Aumentar el set de validación o usar K-fold cross validation.


Cargar el modelo entrenado
```python
from tensorflow.keras.models import load_model
best_model = load_model("/ruta/a/best_model.keras")
```

---
