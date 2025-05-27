# Clustering de Vehículos - Análisis Exploratorio y Modelado No Supervisado

Este repositorio presenta un proyecto de análisis no supervisado orientado al agrupamiento de vehículos utilizando técnicas de preprocesamiento, reducción de dimensionalidad y algoritmos de clustering, junto con evaluaciones internas y externas de rendimiento.

## Estructura del Proyecto

- `utils.py`: Módulo con funciones reutilizables para la carga, limpieza, transformación, modelado y evaluación de datos.
- `test_utils.py`: Suite de pruebas automatizadas implementada con PyTest para validar la integridad funcional de `utils.py`.
- `LAB10-RUELAS.ipynb`: Notebook principal donde se aplica el flujo completo del análisis de clustering, incluyendo visualización y comparación de modelos.
- `data/`: Carpeta opcional para almacenar archivos de datos originales o preprocesados.

## Tecnologías Utilizadas

- Python 3.13+
- Pandas, NumPy
- Scikit-learn
- PyTest
- Jupyter Notebook

## Flujo de Trabajo

1. **Carga y limpieza de datos**
   - Eliminación de columnas irrelevantes
   - Manejo de valores nulos
   - Tratamiento de outliers mediante IQR

2. **Preprocesamiento**
   - Separación de variables numéricas y categóricas
   - Estandarización (`StandardScaler`, `RobustScaler`)
   - Codificación con dummies
   - Reducción de dimensionalidad (PCA)

3. **Modelado**
   - K-Means (con optimización de K)
   - DBSCAN
   - Agglomerative Clustering

4. **Evaluación**
   - Silhouette Score
   - Davies-Bouldin Index
   - Adjusted Rand Index (ARI)

5. **Validación**
   - Pruebas automatizadas para todas las funciones críticas usando PyTest

## Ejecución de Tests

Para ejecutar la suite de pruebas:

```bash
pytest test_utils.py
````

## Autor

- César Diego Ruelas Flores
- Estudiante de Big Data y Ciencia de Datos – TECSUP
- Fecha del proyecto: mayo 2025

## Licencia

Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
