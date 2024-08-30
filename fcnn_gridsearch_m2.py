import io
import sys
import logging
import warnings
import numpy as np
import hdf5storage
import scipy.io as sio
import tensorflow as tf
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from contextlib import redirect_stdout
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import (
    Input,
    LeakyReLU,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    AveragePooling2D,
    BatchNormalization,
)




if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set directory paths
cwd = "/home/016231605/ME232/project/"
data_dir = cwd + "data/"
results_dir = cwd + "results_v2/"

# Setup basic configuration for logging
logging.basicConfig(filename=results_dir + 'training_log.txt', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


# Load data
mat = hdf5storage.loadmat(data_dir + "imageMatrix28x28_M2_60000-1.mat")
labeldata = sio.loadmat(data_dir + "X_in4_Y_out11_kamin035_kamax045-1.mat")

labelsFull = labeldata["Y"].transpose()
image = mat["imageMatrix"]


N = len(labelsFull)
Ntrain = int(0.9 * (len(labelsFull)))
Y_train = labelsFull[:Ntrain, :]
X_train = image[:, :, :, :Ntrain]
X_train = X_train.reshape(28 * 28, Ntrain).transpose().reshape(-1, 28, 28, 1)
X_train = X_train.astype("float32")
X_train_orig = X_train[:, :, :, 0]
X_train = X_train_orig.reshape(X_train_orig.shape[0], 28 * 28)

Y_test = labelsFull[Ntrain:, :]
X_test = image[:, :, :, Ntrain:]

X_test = X_test.reshape(28 * 28, N - Ntrain).transpose().reshape(-1, 28, 28, 1)
X_test = X_test.astype("float32")
X_test_orig = X_test[:, :, :, 0]
X_test = X_test_orig.reshape(X_test_orig.shape[0], 28 * 28)

del labeldata, labelsFull, image  # Free memory

logging.info("Data shapes:")
logging.info("X_train: %s", X_train.shape)
logging.info("X_test: %s", X_test.shape)
logging.info("Y_train: %s", Y_train.shape)
logging.info("Y_test: %s", Y_test.shape)

# Splitting the training data set into training and validation
random_seed = 1
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=random_seed
)
fit_params = {'validation_split': 0.1, 'shuffle': True}

##### Model Definition with Improvements
def FCNN_model(learn_rate=0.001, d1_neurons=64, d2_neurons=64, d3_neurons=64):
    model = Sequential([
        Dense(d1_neurons, input_dim=28 * 28, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(d2_neurons, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(d3_neurons, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(11, activation='relu')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), loss='mse', metrics=['accuracy'])
    return model

# Initialize and compile the model
model_instance = FCNN_model()

# Function to capture the model summary
def capture_model_summary(model):
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()

# Capturing the model summary
model_summary = capture_model_summary(model_instance)

# Logging the model summary
logging.info("Model Summary:\n%s", model_summary)

# Define parameters for GridSearchCV
param_grid = {
    'batch_size': [50, 100],
    'epochs': [10, 50, 100],
    'learn_rate': [0.01, 0.001, 0.0001],
    'd1_neurons': [8, 16, 32],
    'd2_neurons': [8, 16, 32],
    'd3_neurons': [8, 16, 32]
}

# Perform Grid Search

model = KerasRegressor(build_fn=FCNN_model, verbose=1)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(X_train, Y_train)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, return_train_score=True)
grid_result = grid.fit(X_train, Y_train, **fit_params)


# Define best_model once after grid search is complete
best_model = grid_result.best_estimator_.model

# Capturing and logging the model summary
best_model_summary = capture_model_summary(best_model)
logging.info("Best Model Summary:\n%s", best_model_summary)


# Output best results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
logging.info("Best: %f using %s", grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
# for mean, stdev, param in zip(means, stds, params):
#     print(" %f (%f) with: %r" % (mean, stdev, param))
for mean, stdev, param in zip(means, stds, params):
    logging.info("%f (%f) with: %r", mean, stdev, param)


# Extract the best model's training history
best_model_history = best_model.history.history
best_accuracy = max(best_model_history['accuracy'])
logging.info("Best accuracy: {:.4f}".format(best_accuracy))
#print("Best accuracy: {:.4f}".format(best_accuracy))


# Extract and plot training and validation loss
train_loss = best_model_history['loss']
val_loss = best_model_history.get('val_loss', [])  # Safely handle missing validation loss

# Number of epochs
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
if val_loss:
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
loss_plot_path = results_dir + "training_validation_loss.png"
plt.savefig(loss_plot_path)
plt.close()
logging.info("Loss graph saved to {}".format(loss_plot_path))

# Save the best model
model_save_path = results_dir + "FCNN_M2_best_model.h5"
best_model.save(model_save_path)
logging.info("Model saved to {}".format(model_save_path))
#print("Model saved to {}".format(model_save_path))

# Save the best accuracy to a text file or similar
accuracy_save_path = results_dir + "FCNN_M2_best_accuracy.txt"
with open(accuracy_save_path, 'w') as f:
    f.write("Best accuracy: {:.4f}\n".format(best_accuracy))
logging.info("Best accuracy saved to {}".format(accuracy_save_path))
# print("Best accuracy saved to {}".format(accuracy_save_path))



results = grid_result.best_estimator_.model.predict(X_test)

# Plot results
plt.figure(figsize=(15, 28))
for i in range(6):
    n = np.random.randint(0, len(X_test))
    plt.subplot(6, 1, i + 1)
    predictions = best_model.predict(X_test[n].reshape(1, -1))
    plt.plot(predictions.flatten(), 'r-', label='Predicted')
    plt.plot(Y_test[n], 'b--', label='True Value')
    plt.legend()
plt.tight_layout()
plt.savefig(results_dir + "model_performance.png")

# Plotting the testing results of the model. Each color coded line represents a different image
plt.figure(figsize=(15, 28))
label = ["*", "o", "v", "^", "s", "d", "x", "+", "p", "H", "h", ">", "<", "_"]
color = ["b", "g", "r", "c", "m", "y", "k", "w"]
for i in range(6):
    for j in range(3):
        n = np.random.randint(0, Ntrain * 0.1)
        plt.subplot(5, 2, i + 1)
        plt.plot(results[n, :], marker=label[j], linestyle="--", color=color[j])
        plt.plot(Y_test[n, :], marker=label[j], color=color[j])
        plt.ylabel("$\sigma$", fontsize=15)
        plt.xlabel("Index for Ka", fontsize=15)
    plt.legend(["FCNN Output", "Exact"], fontsize=20)
plt.show()
plt.tight_layout()
plt.savefig(results_dir + "ka_vs_TSCS_3x2.png")

for i in range(11):
    figure(figsize=(7, 5))
    plot(results[50:100, i])
    plot(Y_test[50:100, i])
    plt.xlabel("Frequency")
    plt.ylabel("pf")
    plt.legend(["FCNN Output", "Exact"], fontsize=10)
    plt.title(f"Predicted TSCS ka = 0.{35+i}")
    plt.savefig(results_dir + f"TSCS_0{35+i}_50x.png")

