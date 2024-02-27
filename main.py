
import numpy as np
import csv
import matplotlib.pyplot as plt

from Perceptron import Perceptron

datos = [] 
y = []

with open('OR_trn.csv', 'r') as archivo_csv:
    lector_csv = csv.reader(archivo_csv)
    for fila in lector_csv:
        datos.append([float(fila[0]), float(fila[1])])
        y.append(float(fila[2]))

x = np.array(datos)

etaStr = input("Ingrese la tasa de aprendizaje (0-1): ")
nIterStr = input("Ingrese el número de iteraciones: ");

ppn = Perceptron(eta = float(etaStr), nIter = int(nIterStr))

ppn.fit(x, y)

plt.plot(range(1, len(ppn.errores_)+1), ppn.errores_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Número de actualizaciones')
plt.show()


nuevos_datos = []

with open('OR_tst.csv', 'r') as archivo_csv:
    lector_csv = csv.reader(archivo_csv)
    for fila in lector_csv:
        nuevos_datos.append([float(fila[0]), float(fila[1])])

# Hacer predicciones usando el perceptrón entrenado
for dato in nuevos_datos:
    prediccion = ppn.predice(dato)
    print(f'Entrada: {dato}, Predicción: {prediccion}')


# Visualizar los datos de entrenamiento junto con la recta de separación
plt.scatter(x[0:3, 0], x[0:3, 1], color='red', marker='o', label='Positivo')
plt.scatter(x[3, 0], x[3, 1], color='blue', marker='x', label='Negativo')

# Calcular los valores para la recta de separación
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.array([ppn.predice(np.array([xx1, yy1])) for xx1, yy1 in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Dibujar la región de decisión
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper center')

plt.show()