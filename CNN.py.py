from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Load and preprocess data
train_features = pd.read_csv('lish-moa/train_features.csv')
train_targets = pd.read_csv('lish-moa/train_targets_scored.csv')
train_features = train_features.drop('sig_id', axis=1)
train_targets = train_targets.drop('sig_id', axis=1)

train_features, test_features, train_scored, test_scored = train_test_split(
    train_features, train_targets, test_size=0.2, random_state=42)

# One-hot encoding
def oneHotEncoding(data):
    discrete_features = ['cp_dose', 'cp_time', 'cp_type']
    return pd.get_dummies(data, columns=discrete_features, dtype=int)

X_train = oneHotEncoding(train_features)
X_test = oneHotEncoding(test_features)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA transformation
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Reshape data for 1D CNN
X_train_reshaped = X_train.reshape(-1, 50, 1)
X_test_reshaped = X_test.reshape(-1, 50, 1)

# Model training
# optimizers = [SGD(learning_rate=0.001)]
activations = ['relu', 'sigmoid', 'tanh']

best_loss = float('inf')
best_optimizer = None
best_activation = None

training_losses = []
testing_losses = []
activation_losses = {}

# for optimizer in optimizers:
for activation in activations:
    print(f"activation: {activation}")
    model = Sequential([
        # Conv1D(32, 3, activation=activation, input_shape=(50, 1)),
        # MaxPooling1D(2),
        # Dropout(0.2),
        Conv1D(64, 3, activation=activation),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation=activation),
        Dropout(0.2),
        Dense(train_scored.shape[1], activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, train_scored.values, epochs=50, batch_size=32, verbose=0)
    loss = model.evaluate(X_test_reshaped, test_scored.values, verbose=0)[0]
    loss1 = model.evaluate(X_train_reshaped, train_scored.values, verbose=0)[0]
    print(f"loss1 i.e. trainig: {loss1}")
    print(f"loss i.e. testing: {loss}")
    activation_losses[activation] = loss
    training_losses.append(loss)
    print(f"loss: {loss}")
    if loss < best_loss:
        best_loss = loss
        # best_optimizer = optimizer
        best_activation = activation


plt.figure(figsize=(10, 6))
plt.bar(activation_losses.keys(), activation_losses.values())
plt.xlabel('Activation Function')
plt.ylabel('Loss')
plt.title('Activation Function vs Loss')
plt.show()
