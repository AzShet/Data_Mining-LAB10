import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from utils import (
    cargar_datos, eliminar_columnas, eliminar_filas_nan,
    separar_variables, tratar_outliers_iqr, escalar_datos,
    dummizar_variables, aplicar_kmeans, evaluar_clustering_interno,
    evaluar_clustering_externo, escalar_robusto,
    reducir_dimensionalidad_pca, buscar_mejor_k, aplicar_dbscan,
    aplicar_agglomerative
)

def test_eliminar_columnas():
    df = pd.DataFrame({"a": [1], "b": [2]})
    resultado = eliminar_columnas(df, ["b"])
    assert "b" not in resultado.columns

def test_eliminar_filas_nan():
    df = pd.DataFrame({"a": [1, np.nan], "b": [2, 3]})
    resultado = eliminar_filas_nan(df)
    assert resultado.shape[0] == 1

def test_separar_variables():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    num, cat = separar_variables(df)
    assert "num" in num.columns and "cat" in cat.columns

def test_tratar_outliers_iqr():
    df = pd.DataFrame({"x": [1, 2, 3, 1000]})
    result = tratar_outliers_iqr(df)
    assert result["x"].max() < 1000

def test_escalar_datos():
    df = pd.DataFrame({"x": [1, 2, 3]})
    escalado = escalar_datos(df)
    assert np.isclose(escalado.mean(), 0, atol=1e-7)

def test_dummizar_variables():
    df = pd.DataFrame({"sexo": ["M", "F"]})
    dummy = dummizar_variables(df)
    assert dummy.shape[1] == 1

def test_aplicar_kmeans():
    X, _ = make_blobs(n_samples=10, centers=2, random_state=42)
    df = pd.DataFrame(X)
    modelo = aplicar_kmeans(df, 2)
    assert hasattr(modelo, 'labels_')

def test_evaluar_clustering_interno():
    X, _ = make_blobs(n_samples=10, centers=2, random_state=42)
    df = pd.DataFrame(X)
    modelo = aplicar_kmeans(df, 2)
    sil, db = evaluar_clustering_interno(df, modelo.labels_)
    assert 0 <= sil <= 1 and db >= 0

def test_evaluar_clustering_externo():
    etiquetas = [0, 0, 1, 1]
    predichas = [0, 0, 1, 1]
    ari = evaluar_clustering_externo(etiquetas, predichas)
    assert ari == 1.0

def test_escalar_robusto():
    df = pd.DataFrame({"x": [1, 2, 3, 100]})
    escalado = escalar_robusto(df)
    assert np.isclose(np.median(escalado), 0, atol=1e-7)

def test_reducir_dimensionalidad_pca():
    df = pd.DataFrame(np.random.rand(10, 5))
    reducido, pca = reducir_dimensionalidad_pca(df, varianza=0.9)
    assert reducido.shape[1] <= df.shape[1]

def test_buscar_mejor_k():
    X, _ = make_blobs(n_samples=50, centers=4, random_state=42)
    df = pd.DataFrame(X)
    k, scores = buscar_mejor_k(df, 2, 5)
    assert 2 <= k <= 5

def test_aplicar_dbscan():
    X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
    df = pd.DataFrame(X)
    labels = aplicar_dbscan(df, eps=0.5, min_samples=3)
    assert len(labels) == 20

def test_aplicar_agglomerative():
    X, _ = make_blobs(n_samples=20, centers=3, random_state=42)
    df = pd.DataFrame(X)
    labels = aplicar_agglomerative(df, 3)
    assert len(set(labels)) == 3
