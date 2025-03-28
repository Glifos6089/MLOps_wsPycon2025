import pickle
import os
import argparse
import wandb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import wandb.sklearn
from sklearn.datasets import load_iris

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Función para cargar un artefacto de WandB como DataFrame
def load_artifact_as_dataframe(run, artifact_name, artifact_type):
    artifact = run.use_artifact(f'{artifact_name}:latest', type=artifact_type)
    artifact_dir = artifact.download()
    ruta_archivo = f"{artifact_dir}/{artifact_name}.csv.table.json"

    try:
        table = artifact.get(f"{artifact_name}.csv")
        data = table.data
        columns = table.columns
        df = pd.DataFrame(data, columns=columns)
        if 'target' in df.columns:
            df = df.drop('target', axis=1)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
        return None

def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="Prueba-Clustering-Diplomado",
        name=f"Train KMeans ExecId-{args.IdExecution} ExperimentId-{experiment_id}-Clusters-{config.get('n_clusters', 'default')}",
        job_type="train-model", config=config) as run:
        config = wandb.config

        # Cargar los datos de entrenamiento preprocesados desde el artefacto
        train_data_df = load_artifact_as_dataframe(run, 'iris_train_preprocesado', 'dataset')
        if train_data_df is None:
            print("Error al cargar los datos de entrenamiento. Saliendo.")
            return None

        n_clusters = config.n_clusters

        # Crear una instancia del modelo K-Means con la configuración actual
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')

        # "Entrenar" el modelo
        model.fit(train_data_df)

        # Registrar métricas relevantes (Silhouette Score)
        try:
            labels = model.predict(train_data_df)
            silhouette = silhouette_score(train_data_df, labels)
            wandb.log({"train/silhouette_score": silhouette})
            print(f"Silhouette Score (n_clusters={n_clusters}): {silhouette:.4f}")
        except Exception as e:
            print(f"Error al calcular o registrar métricas: {e}")

        # Registrar la gráfica del Codo usando wandb.sklearn
        wandb.sklearn.plot_elbow_curve(model, train_data_df)

        # Registrar la gráfica de la Silueta usando wandb.sklearn
        iris = load_iris()
        feature_names = iris.feature_names
        wandb.sklearn.plot_silhouette(model, train_data_df, feature_names)

        # Guardar el modelo entrenado como un nuevo artefacto
        model_name = f"kmeans_model_clusters_{n_clusters}"
        trained_model_artifact = wandb.Artifact(
            model_name, type="model",
            description=f"K-Means clustering model with {n_clusters} clusters",
            metadata=dict(config))

        trained_model_path = f"trained_{model_name}.pkl"
        with open(trained_model_path, 'wb') as file:
            pickle.dump(model, file)
        trained_model_artifact.add_file(trained_model_path)
        wandb.save(trained_model_path)
        run.log_artifact(trained_model_artifact)

        return model

# Lista de diferentes valores para el número de clusters
n_clusters_list = [2, 3, 4, 5, 6]

# Realizar 5 entrenamientos con diferentes parámetros
for i, n_clusters in enumerate(n_clusters_list):
    train_config = {"n_clusters": n_clusters}
    trained_model = train_and_log(train_config, experiment_id=i)

    if trained_model:
        print(f"Entrenamiento {i+1} con {n_clusters} clusters completado.")

# No olvides finalizar la ejecución de WandB si no vas a hacer más operaciones
wandb.finish()
