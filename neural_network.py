
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
