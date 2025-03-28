import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import wandb

# Inicializar WandB (asegúrate de que el nombre del proyecto coincida) lol
run = wandb.init(project="Prueba-Clustering-Diplomado")

artifact = run.use_artifact('rgarzona-universidad-nacional-de-colombia/Prueba-Clustering-Diplomado/iris_preprocesado:v0', type='dataset')
artifact_dir = artifact.download()



# Construye la ruta al archivo de datos dentro del artefacto descargado
# Suponiendo que el archivo dentro del artefacto se llama 'preprocessed_data.csv'
ruta_archivo = f"{artifact_dir}/preprocessed_data.csv.table.json"

# Lee el archivo CSV en un DataFrame de pandas
try:

    table = artifact.get("preprocessed_data.csv")

    data = table.data
    columns = table.columns

    data_preprocesada = pd.DataFrame(data, columns=columns)
    print("Datos preprocesados cargados exitosamente en el DataFrame:")
    print(data_preprocesada)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo 'preprocessed_data.csv' en la ruta: {ruta_archivo}")
    print(f"Asegúrate de que el artefacto '{artifact.name}:{artifact.version}' contenga este archivo.")


wandb.init(project="Prueba-Clustering-Diplomado")

X_prep = data_preprocesada
iris = load_iris(as_frame=True)
y_original = iris.target
X_train_prep, X_temp_prep, y_train, y_temp = train_test_split(
    X_prep, y_original, test_size=0.2, random_state=42, stratify=y_original
)
X_val_prep, X_test_prep, y_val, y_test = train_test_split(
    X_temp_prep, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
train_prep_df = pd.DataFrame(X_train_prep, columns=X_prep.columns)
train_prep_df['target'] = y_train.values
val_prep_df = pd.DataFrame(X_val_prep, columns=X_prep.columns)
val_prep_df['target'] = y_val.values
test_prep_df = pd.DataFrame(X_test_prep, columns=X_prep.columns)
test_prep_df['target'] = y_test.values

wandb.log({"data_split_preprocesada/train_size": len(train_prep_df)})
wandb.log({"data_split_preprocesada/validation_size": len(val_prep_df)})
wandb.log({"data_split_preprocesada/test_size": len(test_prep_df)})

print("Tamaño del conjunto de entrenamiento preprocesado:", len(train_prep_df))
print("Tamaño del conjunto de validación preprocesado:", len(val_prep_df))
print("Tamaño del conjunto de prueba preprocesado:", len(test_prep_df))

# Función para guardar un DataFrame como un artefacto en WandB
def guardar_como_artefacto(df, nombre, descripcion, tipo="dataset"):
    artefacto = wandb.Artifact(name=nombre, type=tipo, description=descripcion)
    tabla = wandb.Table(dataframe=df)
    artefacto.add(tabla, f"{nombre}.csv")
    wandb.log_artifact(artefacto)

# Guardar los conjuntos de entrenamiento, validación y prueba como artefactos
guardar_como_artefacto(
    train_prep_df,
    "iris_train_preprocesado",
    "Conjunto de entrenamiento preprocesado del dataset Iris."
)

guardar_como_artefacto(
    val_prep_df,
    "iris_validation_preprocesado",
    "Conjunto de validación preprocesado del dataset Iris."
)

guardar_como_artefacto(
    test_prep_df,
    "iris_test_preprocesado",
    "Conjunto de prueba preprocesado del dataset Iris."
)
