# Diagnóstico Precoz de Enfermedad Cardíaca

## Dominio
Salud Digital / Medicina

## Descripción del Problema
Un centro de salud digital busca desarrollar una herramienta de pre-diagnóstico para ayudar a los médicos a identificar pacientes con alto riesgo de enfermedad cardíaca basándose en datos clínicos rutinarios.

## Dataset
El dataset utilizado es el [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) de Kaggle. Contiene atributos como edad, sexo, presión arterial, colesterol, entre otros.

## Tarea
Entrenar un clasificador MLP (Multi-Layer Perceptron) para predecir la presencia (`target=1`) o ausencia (`target=0`) de enfermedad cardíaca.

## Estructura del Proyecto

### Dependencias
- pandas
- numpy
- scikit-learn

### Archivos
- `heart_disease_processed_entrenamiento.csv`: Archivo CSV con los datos de entrenamiento.
- `heart_disease_processed_test.csv`: Archivo CSV con los datos de prueba.

### Código
El código está estructurado de la siguiente manera:

1. **Carga de Datos**: Se cargan los datos desde los archivos CSV.
2. **Preprocesamiento**: Se extraen las características y las etiquetas, y se convierten las etiquetas a formato binario.
3. **Definición del Modelo MLP**: Se define la clase `MLP` que implementa un perceptrón multicapa.
4. **Entrenamiento del Modelo**: Se entrena el modelo con los datos de entrenamiento.
5. **Evaluación del Modelo**: Se evalúa el modelo con los datos de prueba y se imprime la matriz de confusión y el reporte de clasificación.

### Uso
1. Asegúrate de tener instaladas las dependencias necesarias.
2. Coloca los archivos CSV en el mismo directorio que el script.
3. Ejecuta el script para entrenar y evaluar el modelo.

### Ejemplo de Salida
El script imprimirá las predicciones en el conjunto de prueba, la matriz de confusión y el reporte de clasificación.

### Matriz de Confusión
La matriz de confusión se imprime en un formato legible para facilitar la interpretación de los resultados.

### Reporte de Clasificación
El reporte de clasificación incluye métricas como precisión, recall y f1-score para evaluar el rendimiento del modelo.

## Utilidad
Este proyecto es útil para el diagnóstico precoz de enfermedades cardíacas y puede ser utilizado por médicos para identificar pacientes con alto riesgo basándose en datos clínicos rutinarios.

