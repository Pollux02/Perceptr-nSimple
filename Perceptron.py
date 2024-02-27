import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.5, nIter = 50, randomState = 1):
        self.eta = eta #Tasa de aprendizaje (0 - 1)
        self.nIter = nIter #Número de iteraciones que va a pasar el conjunto de datos completo
        self.randomState = randomState #Semilla del generadot de números aleatorios

    def fit(self, x, y):
        #x = vector de datos de entrenamiento
        #nFeatures = número de características
        #y = vector de etiquetas de respuesta

        rgen = np.random.RandomState(self.randomState) #Generamos números aleatorios
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + x.shape[1]) #Números aleatorios con dev est 0.01
        #self.w_ = [0,0,0] #Inicializando pesos en cero
        self.errores_ = [] #Lista vacía para errores

        print('Pesos iniciales', self.w_)

        for _ in range(self.nIter): #Ciclo que se repite según el número de iteraciones
            errores = 0
            for xi, etiqueta in zip(x,y): #Ciclo que se repite según número de muestras.
                actualizacion = self.eta*(etiqueta - self.predice(xi))
                self.w_[1:] += actualizacion * xi
                self.w_[0] += actualizacion
                errores += int(actualizacion != 0)
            self.errores_.append(errores)
            print('Pesos en Epoch', _ , ':' , self.w_)
    
    def entrada_neta(self, x):
        #Cálculo de la entrada neta
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predice(self, x):
        #Etiqueta de clase de retorno después del paso unitario
        return np.where(self.entrada_neta(x) >= 0, 1, -1)