import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import wandb

# Asegúrate de que WandB esté inicializado (si no lo está ya en tu script principal) a
if wandb.run is None:
    wandb.init(project="Prueba-Clustering-Diplomado")

# Cargar el dataset Iris como un DataFrame de pandas
iris = load_iris(as_frame=True)
data = iris.frame

# Separar las características (features) del target
X = data.drop('target', axis=1)
y = data['target']

# Dividir el dataset en entrenamiento (80%) y el resto (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Dividir el conjunto restante (temp) en validación (50% de temp = 10% del total) y prueba (50% de temp = 10% del total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convertir los conjuntos a DataFrames de pandas (si aún no lo son)
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['target'] = y_train.values

val_df = pd.DataFrame(X_val, columns=X.columns)
val_df['target'] = y_val.values

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.values

# Registrar el tamaño de los conjuntos en WandB
wandb.log({"data_split/train_size": len(train_df)})
wandb.log({"data_split/validation_size": len(val_df)})
wandb.log({"data_split/test_size": len(test_df)})

print("Tamaño del conjunto de entrenamiento:", len(train_df))
print("Tamaño del conjunto de validación:", len(val_df))
print("Tamaño del conjunto de prueba:", len(test_df))

# Opcional: Guardar los conjuntos como artefactos en WandB
def guardar_como_artefacto(df, nombre, descripcion, tipo="dataset"):
    artefacto = wandb.Artifact(name=nombre, type=tipo, description=descripcion)
    tabla = wandb.Table(dataframe=df)
    artefacto.add(tabla, f"{nombre}.csv")
    wandb.log_artifact(artefacto)

guardar_como_artefacto(train_df, "iris_train", "Conjunto de entrenamiento del dataset Iris.")
guardar_como_artefacto(val_df, "iris_validation", "Conjunto de validación del dataset Iris.")
guardar_como_artefacto(test_df, "iris_test", "Conjunto de prueba del dataset Iris.")
