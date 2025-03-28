import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import wandb
import os
import argparse

# Inicializar WandB
wandb.init(project="Prueba-Clustering-Diplomado")

# Cargar el dataset Iris como un DataFrame de pandas
iris = load_iris(as_frame=True)
data = iris.frame
target = iris.target

# Separar las características (features) del target
X = data.drop('target', axis=1)
y = data['target']

# Inicializar el escalador estándar
scaler = StandardScaler()

# Ajustar el escalador a los datos y transformarlos
X_scaled = scaler.fit_transform(X)

# Convertir el array NumPy escalado de vuelta a un DataFrame de pandas (opcional, pero útil para inspección)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Registrar los parámetros de preprocesamiento en WandB
wandb.log({"preprocesamiento/scaler": "StandardScaler"})
wandb.log({"preprocesamiento/n_muestras_original": X.shape[0]})
wandb.log({"preprocesamiento/n_caracteristicas_original": X.shape[1]})

print("Datos originales:")
print(X.head())
print("\nDatos preprocesados (escalados):")
print(X_scaled_df.head())

# Crear un artefacto de WandB para los datos preprocesados
nombre_artefacto = "iris_preprocesado"
descripcion_artefacto = "Datos del dataset Iris preprocesados con StandardScaler."
tipo_artefacto = "dataset"

artefacto_preprocesado = wandb.Artifact(
    name=nombre_artefacto,
    type=tipo_artefacto,
    description=descripcion_artefacto
)

# Crear una tabla de WandB a partir del DataFrame preprocesado
tabla_preprocesada = wandb.Table(dataframe=X_scaled_df)

# Agregar la tabla al artefacto
artefacto_preprocesado.add(tabla_preprocesada, "preprocessed_data.csv") # El nombre del archivo dentro del artefacto

# Guardar el artefacto en WandB
wandb.log_artifact(artefacto_preprocesado)

