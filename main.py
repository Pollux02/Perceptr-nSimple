import numpy as np
import csv
import matplotlib.pyplot as plt
import random

from Perceptron import Perceptron

datos = [] 
newDatos = []
y = []
newY = []
datosPrueba = []
resultadosPrueba = []

nombreArchivoEntrenamiento = input("Ingrese el nombre de el archivo .csv para extraer los datos de entrenamiento y prueba (Ejemplo: dataset1.csv): ")

with open(nombreArchivoEntrenamiento, 'r') as archivo_csv:
    lector_csv = csv.reader(archivo_csv)
    for fila in lector_csv:
        datos.append([float(fila[0]), float(fila[1]),  float(fila[2])])
        y.append(float(fila[3]))

porcentajeEntrenamiento = input("Ingrese el porcentaje de datos de entrenamiento (0-100 el porcentaje restante será de prueba): ")

etaStr = input("Ingrese la tasa de aprendizaje (0-1): ")
nIterStr = input("Ingrese el número de iteraciones: ")

case = input("\n1)Primero entrenamiento, luego pruebas\n2)Primero pruebas, luego entrenamiento\n3)Al azar\n4)Uno y uno\n5)Por bloques\nIngrese el modo de partición: ")

numEntrenamiento = int(len(datos) * int(porcentajeEntrenamiento) / 100)
numPrueba = len(datos)-numEntrenamiento

if case == '1':
    newDatos = datos[0:numEntrenamiento]
    newY = y[0:numEntrenamiento]
    datosPrueba = datos[numEntrenamiento: len(datos)]
    resultadosPrueba = y[numEntrenamiento: len(datos)]

    x = np.array(newDatos)
        
elif case == '2':
    datosPrueba = datos[0: numPrueba]
    resultadosPrueba = y[0:numPrueba]
    newDatos = datos[numPrueba:len(datos)]
    newY = y[numPrueba:len(datos)]

    x = np.array(newDatos)

elif case == '3':
    indices = list(range(len(datos)))
    random.shuffle(indices)

    # Seleccionar los índices para entrenamiento y prueba
    indices_entrenamiento = indices[:numEntrenamiento]
    indices_prueba = indices[numEntrenamiento:]

    # Crear conjuntos de entrenamiento y prueba usando los índices
    newDatos = [datos[i] for i in indices_entrenamiento]
    newY = [y[i] for i in indices_entrenamiento]

    datosPrueba = [datos[i] for i in indices_prueba]
    resultadosPrueba = [y[i] for i in indices_prueba]

    x = np.array(newDatos)

elif case == '4':
    for indice in range(len(datos)):
        if(indice % 2 == 0):
            if(len(newDatos)<numEntrenamiento):
                newDatos.append(datos[indice])
                newY.append(y[indice])
            else:
                datosPrueba.append(datos[indice])
                resultadosPrueba.append(y[indice])
        else:
            if(len(datosPrueba)<numPrueba):
                datosPrueba.append(datos[indice])
                resultadosPrueba.append(y[indice])
            else:
                newDatos.append(datos[indice])
                newY.append(y[indice])

    x = np.array(newDatos)

elif case == '5':
    bloqueEntrenamiento = numEntrenamiento/10
    bloquePrueba = numPrueba/10


    iBloqueEntrenamiento = 0
    iBloquePrueba = -1

    for indice in range(len(datos)):
        if(iBloqueEntrenamiento < bloqueEntrenamiento):
            newDatos.append(datos[indice])
            newY.append(y[indice])
            iBloqueEntrenamiento = iBloqueEntrenamiento + 1

        elif(iBloqueEntrenamiento == bloqueEntrenamiento):
            iBloquePrueba = 0
            iBloqueEntrenamiento = iBloqueEntrenamiento + 1

        if(iBloquePrueba>=0 and iBloquePrueba<bloquePrueba):
            datosPrueba.append(datos[indice])
            resultadosPrueba.append(y[indice])
            iBloquePrueba = iBloquePrueba + 1
            
        elif(iBloquePrueba>= bloquePrueba):
            iBloqueEntrenamiento = 0
            iBloquePrueba = -1
        
    x = np.array(newDatos)

ppn = Perceptron(eta = float(etaStr), nIter = int(nIterStr))

ppn.fit(x, newY)

plt.plot(range(1, len(ppn.errores_)+1), ppn.errores_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Número de actualizaciones')
plt.show()


i = 0
aciertos = 0
positivos_reales = sum(resultadosPrueba)  # Número total de positivos reales

# Hacer predicciones usando el perceptrón entrenado
for dato in datosPrueba:
    prediccion = ppn.predice(dato)
    resultado = resultadosPrueba[i]
    i = i+1
    
    print(f'Entrada: {dato}, Predicción: {prediccion}, Resultado: {resultado}')

     # Calcular aciertos
    if prediccion == resultado:
        aciertos += 1

# Calcular precisión y sensibilidad
precision = aciertos / len(datosPrueba)
sensibilidad = aciertos / positivos_reales 

print(f'Precisión: {precision:.2f}')
print(f'Sensibilidad: {sensibilidad:.2f}')