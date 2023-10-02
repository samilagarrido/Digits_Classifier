import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
import numpy.linalg as LA
import pandas as pd
import random
from random import sample 
from tqdm import tqdm

class PLA:

    def __init__(self, iterations=1000, n_min=50, n_max=200):
        self.iterations = iterations
        self.n_min = n_min #número mínimo de pontos para o treinamento
        self.n_max = n_max


    def fit(self, X, y, iterations=None):
        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(X.shape[1])

        for j in tqdm(range(self.iterations)):
            n = random.randint(self.n_min, self.n_max)
            indexes = np.random.randint(len(X) - 1, size=n)
            X_ = X[indexes]
            y_ = y[indexes]

            for i in range(n):
                while len(self.constroi_listaPCI(X_, y_, self.w)[0]) != 0:
                    PCI, PCI_y = self.constroi_listaPCI(X_, y_, self.w)
                    index = random.randint(0, len(PCI) - 1)
                    self.w = self.w + PCI[index] * PCI_y[index]
                    break


        self.w = self.w

    def constroi_listaPCI(self, X, y, w):
        X = np.array(X)
        y = np.array(y)
        w = np.array(w)
        condition = np.sign(X.dot(w)) != y
        PCI = X[condition]
        PCI_y = y[condition]
        return PCI, PCI_y
    
    def get_weights(self):
        return self.w

    def set_w(self, w):
        self.w = w

    def h(self, x):
        return np.sign(np.dot(self.w, x))

    def error_in(self, X, y):
        return np.mean(np.sign(np.dot(self.w, X.T)) != y)

    def predict(self, X):
        return [self.h(x) for x in X]

class RegressaoLinear():

    def __init__(self, iterador=None):
        self.iter = iterador

    def __str__(self):
        return "Regressão Linear"

    def fit(self, X, y):
        h = np.dot(X.T, X)
        g = np.dot(X.T, y)
        self.w = np.dot((inv(h)), g)

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T))
    
    def get_weights(self):
        return self.w

class RegressaoLogistica:
    
    def __init__(self, eta=0.1, iterador=1000, batch=2048):
        self.eta = eta
        self.iterador = iterador
        self.batch = batch
        self.w = None

    def __str__(self):
        return "Regressão Logística"
    
    def fit(self, X, y, iterador=None, lamb=1e-6):

        if iterador is not None:
            self.iter = iterador

        N, d = X.shape
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        w = np.zeros(d)

        for t in tqdm(range(self.iterador)):
            if self.batch < N:
                rand_indexes = np.random.choice(N, self.batch, replace=False)
                X_batch, y_batch = X[rand_indexes], y[rand_indexes]
                N_batch = self.batch
            else:
                X_batch, y_batch = X, y
                N_batch = N

            sigm = 1 / (1 + np.exp(y_batch * np.dot(w, X_batch.T).reshape(-1, 1)))
            gt = - 1 / N_batch * np.sum(X_batch * y_batch * sigm, axis=0) 

            if np.linalg.norm(gt) < 1e-10:
                break
            
            w -= self.eta  * gt 

        self.w = w
    
    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def predict(self, X):
        pred = self.predict_prob(X)
        y = np.where(pred >= 0.5, 1, -1)
        return y 

    def get_weights(self):
        return self.w
    
    def set_w(self, w):
        self.w = w

class OneVsAllClassifier:

    def __init__(self, model=None, digits=None, iterations=None):
        self.model = model
        self.digits = digits
        self.weights = []
        self.iterations = iterations

    def fit(self, X_train, y_train):
        X = X_train.copy()
        y = y_train.copy()

        for i, digit in enumerate(self.digits[:-1]):
            if i == 0:
                y_i = np.where(y == digit, 1, -1)
                if self.iterations is None:
                    self.model.fit(X, y_i)
                else:
                    self.model.fit(X, y_i, iterations=self.iterations[i])
                self.weights.append(self.model.get_weights())
                previous_digit = digit
            else:
                X = np.delete(X, np.where(y == previous_digit), axis=0)
                y = np.delete(y, np.where(y == previous_digit))
                y_i = np.where(y == digit, 1, -1)
                if self.iterations is None:
                    self.model.fit(X, y_i)
                else:
                    self.model.fit(X, y_i, iterations=self.iterations[i])
                self.weights.append(self.model.get_weights())
                previous_digit = digit

    def predict(self, X):
        predictions = []
        for i, x in enumerate(X):
            for j, digit in enumerate(self.digits[:-1]):
                if np.sign(np.dot(self.weights[j], x)) == 1:
                    predictions.append(digit)
                    break
            if len(predictions) < i + 1:
                predictions.append(self.digits[-1])
        return np.array(predictions)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save_weights(self, file='best_weights.csv'):
        weights_df = pd.read_csv(file)
        new_row = {
            "weights_0": self.weights[0],
            "weights_1": self.weights[1],
            "weights_2": self.weights[2],
            "digits": self.digits,
            "model_info": str(self.model)
        }
        weights_df = weights_df.append(new_row, ignore_index=True)
        weights_df.to_csv(file, index=False)

    def load_weights(self, file='best_weights.csv', index=0):
        weights_df = pd.read_csv(file)
        row = weights_df.iloc[index, :]

        weights_0 = [float(w) for w in row['weights_0'][1:-1].strip().split(" ") if w != '']
        weights_1 = [float(w) for w in row['weights_1'][1:-1].strip().split(" ") if w != '']
        weights_2 = [float(w) for w in row['weights_2'][1:-1].strip().split(" ") if w != '']
        self.weights = np.array([weights_0, weights_1, weights_2])
        self.digits = [int(digit) for digit in row['digits'][1:-1].split(", ")]



