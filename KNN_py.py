from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train_features = pd.read_csv('train_features.csv')
train_targets = pd.read_csv('train_targets_scored.csv')
train_features = train_features.drop('sig_id', axis=1)
train_targets = train_targets.drop('sig_id', axis=1)

train_features, test_features, train_scored, test_scored = train_test_split(
    train_features, train_targets, test_size=0.5, random_state=42)

train_features = train_features.reset_index(drop=True)
train_targets = train_targets.reset_index(drop=True)
train_scored = train_scored.reset_index(drop=True)
test_scored = test_scored.reset_index(drop=True)

train_features = train_features[:5000]
test_features = test_features[:5000]
train_scored = train_scored[:5000]
test_scored = test_scored[:5000]

test_features.shape


def oneHotEncoding(data):
    # performing one hot encoding on discrete features
    discrete_features = ['cp_dose', 'cp_time', 'cp_type']
    data = pd.get_dummies(data, columns=discrete_features, dtype=int)
    return data


def cross_entropy(predicted_probabilities, true_labels):
    # Compute cross-entropy loss
    # A small constant to prevent numerical instability (avoid log(0))
    epsilon = 1e-15
    predicted_probabilities = np.clip(
        predicted_probabilities, epsilon, 1 - epsilon)  # Clip probabilities
    cross_entropy_loss = -np.mean(true_labels * np.log(predicted_probabilities) + (
        1 - true_labels) * np.log(1 - predicted_probabilities))
    return cross_entropy_loss


# Assuming you have defined the oneHotEncoding function
# You can adjust the range of neighbors as needed

param_grid = {'n_neighbors': [100, 200, 300, 400, 500]}

best_loss = float('inf')
best_params = None

neighbor_counts = []
testing_losses = []

for n_neighbors in param_grid['n_neighbors']:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    X_train = train_features
    y_train = train_scored

    X_test = test_features
    y_test = test_scored

    X_train = oneHotEncoding(X_train)
    X_test = oneHotEncoding(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=30)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    loss = 0
    loss_train = 0

    # Use tqdm to create a progress bar for the loop
    for col in tqdm(y_train.columns, desc=f'Processing columns with {n_neighbors} neighbors'):
        if y_train[col].unique().shape[0] == 1:
            y_pred = [0.001] * len(X_test)
            y_train_pred = [0.001] * len(X_train)
        else:
            model.fit(X_train, y_train[col])
            y_train_pred = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]

        loss += log_loss(y_test[col].values, y_pred, labels=[0, 1])
        loss_train += log_loss(y_train[col].values,
                               y_train_pred, labels=[0, 1])

    average_loss_train = loss_train / len(y_train.columns)
    average_loss = loss / len(y_train.columns)

    neighbor_counts.append(n_neighbors)
    testing_losses.append(average_loss)

    print(f"Training loss with {n_neighbors} neighbors: {average_loss_train}")
    print(f"Testing loss with {n_neighbors} neighbors: {average_loss}")

    # Keep track of the best hyperparameters
    if average_loss < best_loss:
        best_loss = average_loss
        best_params = {'n_neighbors': n_neighbors}

print(f"Best hyperparameters: {best_params}")
print(f"Best testing loss: {best_loss}")

# Plotting neighbor count vs loss
plt.plot(neighbor_counts, testing_losses, marker='o')
plt.title('Neighbor Count vs Testing Loss')
plt.xlabel('Number of Neighbors')
plt.ylabel('Testing Loss')
plt.show()
