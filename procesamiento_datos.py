import pandas as pd
import numpy as np

path = "./datos/covtype.csv"

#Crear el dataframe
dataset_datos = pd.read_csv(path)

np.random.seed(77)
#Indices para todo el dataset original
dataset_indices_completo = dataset_datos.index.tolist()

#Se saca una muestra aleatoria de 20.000 indices para las 20.000 parcelas
indices_referencia = np.random.choice(dataset_indices_completo, size=20000, replace=False)
dataset_referencia = dataset_datos.loc[indices_referencia].copy()

#Sacamos todos los indices restantes para no tener duplicados en las otras 20.000 parcelas que necesitamos
indices_referencia_set = set(indices_referencia)
indices_restantes = [idx for idx in dataset_indices_completo if idx not in indices_referencia_set]

#Sacamos la muestra aletoria de las otras 20.000 parcelas para la consulta, eliminando la columan de 'Cover_Type'
indices_consulta = np.random.choice(indices_restantes, size=20000, replace=False)
dataset_consulta = dataset_datos.loc[indices_consulta].copy()
dataset_consulta = dataset_consulta.drop(columns=["Cover_Type"])

for col in dataset_consulta.columns:

      #Media de las variables de referencia, excepto Cover_Type porque esta no está en las variables de consulta
      media = np.mean(dataset_referencia[col])

      #Desviación estandar de las variables de referencia
      if np.std(dataset_referencia[col]) == 0:
            desviacion_estandar = 1
      else:
            desviacion_estandar = np.std(dataset_referencia[col])

      #Se normalizan las variables del conjunto de referencia y consulta, solo con la media y desviación que se obtuvo del conjunto de referencia
      dataset_referencia[col] = (dataset_referencia[col] - media) / desviacion_estandar
      dataset_consulta[col] = (dataset_consulta[col] - media) / desviacion_estandar

# 1. Separar las variables de las etiquetas en el conjunto de referencia
# Convertimos todo a los tipos de datos nativos de C (float32 para variables, int32 para etiquetas)
etiquetas_referencia = dataset_referencia['Cover_Type'].astype(np.int32)
variables_referencia = dataset_referencia.drop(columns=['Cover_Type']).astype(np.float32)
variables_consulta = dataset_consulta.astype(np.float32)

# 2. Exportar a formato binario plano (.bin)
variables_referencia.to_numpy().tofile("./bin/variables_referencia.bin")
etiquetas_referencia.to_numpy().tofile("./bin/etiquetas_referencia.bin")
variables_consulta.to_numpy().tofile("./bin/variables_consulta.bin")