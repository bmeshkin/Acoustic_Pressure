import os
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


# Set directory paths
cwd = "/home/016231605/ME232/project/CNN"
data_dir = os.path.join(cwd, "data/")
results_dir = os.path.join(cwd, "results/")

# Create results directory if it does not exist
os.makedirs(results_dir, exist_ok=True)

# Load data
mat = hdf5storage.loadmat(os.path.join(data_dir, "imageMatrix28x28_M2_60000-1.mat"))
labeldata = sio.loadmat(os.path.join(data_dir, "X_in4_Y_out11_kamin035_kamax045-1.mat"))

# Print to confirm shapes
print("Loaded image matrix shape:", mat['imageMatrix'].shape)
print("Loaded label matrix shape:", labeldata['Y'].shape)


images = mat['imageMatrix']
labels = labeldata['Y'].transpose()

# Splitting the data into training and testing set and reshape the image
N = len(labels)
Ntrain = int(0.9 * N)
Y_train = labels[:Ntrain, :]
X_train = images[:, :, :, :Ntrain]
X_train = X_train.reshape(28 * 28, Ntrain).transpose().reshape(-1, 28, 28, 1) # reshape image to data type CNN can interpret
X_train = X_train.astype('float32')

Y_test2 = labels[Ntrain:, :]
X_test = images[:, :, :, Ntrain:]
X_test = X_test.reshape(28 * 28, N - Ntrain).transpose().reshape(-1, 28, 28, 1)
X_test = X_test.astype('float32')

# Print data shapes to verify correct loading and splitting
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test2.shape)

print("images shape:", images.shape)
print("labels shape:", labels.shape)

# Prepare data
images = images.transpose(3, 0, 1, 2)  # Reshape to (samples, height, width, channels)
images = images.astype('float32') / 255.0  # Normalize pixel values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

def create_model(filter1=16, filter2=32, kernel_size=3, optimizer='Adam', learn_rate=0.001):
    model = Sequential([
        Conv2D(filter1, (kernel_size, kernel_size), strides=1, padding='same', activation='elu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filter2, (kernel_size, kernel_size), strides=1, padding='same', activation='elu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='elu'),
        Dropout(0.2),
        Dense(11, activation='elu')
    ])
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learn_rate)
    else:
        opt = RMSprop(learning_rate=learn_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model

param_grid = {
    'batch_size': [128, 256],
    'epochs': [50, 100],
    'filter1': [16, 32],
    'filter2': [32, 64],
    'kernel_size': [3, 5],
    'optimizer': ['Adam', 'RMSprop'],
    'learn_rate': [0.001, 0.0005]
}

# Testing model creation
try:
    test_model = create_model()
    print("Model was created successfully!")
except Exception as e:
    print("Error creating model:", e)

model = KerasRegressor(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, Y_train)
# Print best configuration and results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Use the best estimator to make predictions
best_model = grid_result.best_estimator_.model
test_accuracy = best_model.evaluate(X_test, Y_test)[1]
print("Test accuracy:", test_accuracy)
predictions = best_model.predict(X_test)


# Save the best model
best_model = grid_result.best_estimator_.model
model_save_path = os.path.join(results_dir, "CNN_M2_best_model.h5")
best_model.save(model_save_path)
print(f"Model saved to {model_save_path}")


# Plot training and validation loss for the best model
plt.figure(figsize=(10, 5))
plt.plot(grid_result.best_estimator_.model.history.history['loss'], label='Training Loss')
plt.plot(grid_result.best_estimator_.model.history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'best_model_loss.png'))
print("Training and validation loss plot saved to:", os.path.join(results_dir, 'best_model_loss.png'))
plt.close()


# Plot predictions vs. true values
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot(predictions[i], 'r-', label='Predicted' if i == 0 else "")
    plt.plot(Y_test[i], 'b--', label='True Value' if i == 0 else "")
plt.legend()
plt.title('Comparison of Predictions and True Values')
plt.xlabel('Output Index')
plt.ylabel('Output Value')
plt.savefig(os.path.join(results_dir, "predictions_comparison.png"))
print("Predictions comparison plot saved to:", os.path.join(results_dir, "predictions_comparison.png"))
plt.close()


Q = np.random.randint(0, len(X_test), (6, 3))

plt.figure(figsize=(15, 28))
labels = ['*', 'o', 'v', '^', 's', 'd', 'x', '+', 'p', 'H', 'h', '>', '<', '_']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for i in range(6):
    for j in range(3):
        p = Q[i][j]
        plt.subplot(6, 3, i * 3 + j + 1)  # Adjust subplot to fit all 18 plots
        plt.plot(predictions[p], marker=labels[j % len(labels)], linestyle='--', color=colors[j % len(colors)])
        plt.plot(Y_test2[p], marker=labels[j % len(labels)], color=colors[j % len(colors)])
        plt.ylabel('Pf', fontsize=15)
        plt.xlabel('Index for Ka', fontsize=15)

plt.suptitle('CNN Output vs. Exact Values Across Different Indices', fontsize=20)
plt.legend(['CNN Output', 'Exact'], fontsize=20, loc='upper left')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title

# Save the figure
plot_path = os.path.join(results_dir, "detailed_predictions_comparison.png")
plt.savefig(plot_path)
plt.close()

print("Detailed comparison plot saved to:", plot_path)