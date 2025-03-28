import pickle
import os
import argparse
import wandb
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

def save_model_as_artifact(config, model, model_name="KMeans", model_description="Simple K-Means Clustering Model"):
    with wandb.init(project="Prueba-Clustering-Diplomado",
                    name=f"save Model ExecId-{args.IdExecution}",
                    job_type="save-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"trained_model_{model_name}.pkl"

        # Guardar el modelo usando pickle
        with open(f"./model/{name_artifact_model}", 'wb') as file:
            pickle.dump(model, file)

        # Añadir el archivo del modelo al artefacto
        model_artifact.add_file(f"./model/{name_artifact_model}")

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)


# Configuración del modelo K-Means
n_clusters = 3  # Ejemplo: definimos 3 clusters
kmeans_config = {"n_clusters": n_clusters,
                 "random_state": 42}

# Crear una instancia del modelo K-Means
kmeans_model = KMeans(**kmeans_config)

# Guardar el modelo entrenado como un artefacto
save_model_as_artifact(kmeans_config, kmeans_model, "kmeans_model", "K-Means clustering model trained on Iris dataset")

