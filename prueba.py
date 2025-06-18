import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# 1. Cargar los datos desde CSV
# ---------------------------
# Asegúrate de que los archivos "heart_disease_processed_entrenamiento.csv" y
# "heart_disease_processed_test.csv" estén en el mismo directorio que este script
train_data = pd.read_csv("heart_disease_processed_entrenamiento.csv")
test_data  = pd.read_csv("heart_disease_processed_test.csv")

# Extraer características y etiqueta
# Se supone que la columna "num" es la etiqueta. 
X_train = train_data.drop("num", axis=1).values
y_train = train_data["num"].values.reshape(-1, 1)
X_test  = test_data.drop("num", axis=1).values
y_test  = test_data["num"].values.reshape(-1, 1)

# ---------------------------
# 2. Convertir la etiqueta a formato binario
# Si se asume que "0" significa ausencia y cualquier valor distinto de 0, presencia
y_train = (y_train != 0).astype(int)
y_test  = (y_test != 0).astype(int)

# ---------------------------
# 3. Definición del Modelo MLP
# ---------------------------
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        # Inicializar pesos y sesgos (valores pequeños aleatorios y ceros)
        self.w_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.b_h  = np.zeros(hidden_size)
        self.w_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.b_o  = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, s):
        return s * (1 - s)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        # Capa oculta: activación ReLU
        self.z_h = np.dot(X, self.w_ih) + self.b_h
        self.a_h = self.relu(self.z_h)
        # Capa de salida: activación Sigmoide (para producir probabilidades)
        self.z_o = np.dot(self.a_h, self.w_ho) + self.b_o
        self.a_o = self.sigmoid(self.z_o)
        return self.a_o
    
    def backward(self, X, y_true, y_pred):
        # Calcular error en la salida
        error_o = y_true - y_pred
        delta_o = error_o * self.sigmoid_derivative(y_pred)
        
        # Propagar el error hacia la capa oculta
        error_h = np.dot(delta_o, self.w_ho.T)
        delta_h = error_h * self.relu_derivative(self.z_h)
        
        # Actualizar pesos y sesgos de la capa de salida
        self.w_ho += np.dot(self.a_h.T, delta_o) * self.lr
        self.b_o  += delta_o.sum(axis=0) * self.lr
        
        # Actualizar pesos y sesgos de la capa oculta
        self.w_ih += np.dot(X.T, delta_h) * self.lr
        self.b_h  += delta_h.sum(axis=0) * self.lr
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            if (epoch + 1) % 100 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"Epoch {epoch + 1} - Loss: {loss:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X)
        # Clasificación binaria: si la probabilidad >= 0.5 se asigna 1, sino 0
        return (y_pred >= 0.5).astype(int)

# ---------------------------
# 4. Entrenamiento del Modelo
# ---------------------------
input_dim = X_train.shape[1]  # número de características
model = MLP(input_size=input_dim, hidden_size=10, output_size=1, learning_rate=0.01)

print("Entrenando el modelo...")
model.train(X_train, y_train, epochs=1000)

# ---------------------------
# 5. Evaluación del Modelo en el Conjunto de Prueba
# ---------------------------
predictions = model.predict(X_test)

print("\nPredicciones en el conjunto de prueba (primeras 10 muestras):")
print(predictions[:10])

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, predictions))

print("\nReporte de Clasificación:")
print(classification_report(y_test, predictions))