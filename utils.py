"""
Funciones reutilizables para el proyecto de clustering de vehículos.

Autor: César Diego Ruelas Flores
Fecha: 24-may-2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def cargar_datos(path):
    return pd.read_csv(path)

def eliminar_columnas(df, columnas):
    return df.drop(columns=columnas, errors='ignore')

def eliminar_filas_nan(df):
    return df.dropna(how='any')

def separar_variables(df):
    num = df.select_dtypes(include=['int64', 'float64']).reset_index(drop=True)
    cat = df.select_dtypes(include=['object', 'category']).reset_index(drop=True)
    return num, cat

def tratar_outliers_iqr(df):
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < limite_inferior) | (df[col] > limite_superior), df[col].median(), df[col])
    return df

def escalar_datos(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def dummizar_variables(df):
    return pd.get_dummies(df, drop_first=True)

def aplicar_kmeans(df, n_clusters, random_state=42, n_init=10):
    modelo = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=n_init)
    modelo.fit(df)
    return modelo

def evaluar_clustering_interno(df, etiquetas):
    sil = silhouette_score(df, etiquetas)
    db = davies_bouldin_score(df, etiquetas)
    return sil, db

def evaluar_clustering_externo(referencia, etiquetas):
    referencia_factorizada, _ = pd.factorize(pd.Series(referencia))
    ari = adjusted_rand_score(referencia_factorizada, etiquetas)
    return ari

# ======================================================================
# --- FUNCIONES AVANZADAS para clustering de vehículos --------------------
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering

def escalar_robusto(df):
    """Escala con mediana y rango intercuartílico (menos sensible a outliers)."""
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def reducir_dimensionalidad_pca(df, varianza=0.95, random_state=42):
    """Aplica PCA reteniendo la fracción de varianza indicada."""
    pca = PCA(n_components=varianza, random_state=random_state)
    datos_reducidos = pca.fit_transform(df)
    return pd.DataFrame(datos_reducidos), pca

def buscar_mejor_k(df, k_min=2, k_max=10, random_state=42, n_init=10):
    """Devuelve el K con mejor Silhouette (y la lista de puntajes)."""
    mejores_scores = []
    for k in range(k_min, k_max + 1):
        modelo = aplicar_kmeans(df, k, random_state, n_init)
        labels = modelo.labels_
        score = silhouette_score(df, labels)
        mejores_scores.append((k, score))
    # Elegir k con Silhouette máximo
    k_opt, _ = max(mejores_scores, key=lambda t: t[1])
    return k_opt, mejores_scores

def aplicar_dbscan(df, eps=0.5, min_samples=5):
    """Agrupamiento DBSCAN; devuelve etiquetas."""
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    labels = modelo.fit_predict(df)
    return labels

def aplicar_agglomerative(df, n_clusters, linkage="ward"):
    """Clustering jerárquico aglomerativo."""
    modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = modelo.fit_predict(df)
    return labels
