import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Random data generation

true_w = 1.0
true_b = 2.0
N = 100 # Number of datapoints
np.random.seed(42)
# x is the feature (input)
x = np.random.rand(N, 1)
# add noise to the dataset to mimic real world data
noise_level = 0.1
epsilon = (noise_level * np.random.randn(N, 1))
# y are the labels (output)
y = true_b + true_w * x + epsilon

# Train test split

# Shuffles the indices 
idx = np.arange(N) 
np.random.shuffle(idx) 
# Uses first 80 random indices for train (8%)
train_idx = idx[:int(N*.8)] 
# Uses the remaining indices for validation 
val_idx = idx[int(N*.8):] 
# Generates train and validation sets 
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Scale the feature(s) so that their loss gradients have a more equal steepness
# This is called normalization or standardization of the data.
# This centers the mean to 0 and the standard deviation to 1
scalar = StandardScaler(with_mean=True, with_std=True) # We use the TRAIN set ONLY to fit the scalar
scalar.fit(x_train)
 # Now we can use the already fit scalar to TRANSFORM # both TRAIN and VALIDATION sets
scaled_x_train = scalar.transform(x_train) 
scaled_x_val = scalar.transform(x_val)

# Training the model

# Start finding the b and w
#Initializes parameters "b" and "w" randomly between and 1
b = np.random.randn(1) 
w = np.random.randn(1)

#Computes our model's predicted output - forward propagation 
def predict(b, w, data):
    yhat = b + w * data
    return yhat

# Mini batch gradient descent - Back propagation

# Set the learning rate
# An lr of 0.1 is high for a learning rate
lr = 0.01
epochs = 1000
batch_size = 16
n_batches = int(len(scaled_x_train) / batch_size)
for i in range(epochs):
	for current_batch in range(n_batches):
	    # Create the batch
		start = current_batch * batch_size
		end = (current_batch * batch_size) + batch_size
		x_train_batch = scaled_x_train[start:end]
		y_train_batch = y_train[start:end]
		# Prediction
		yhat = predict(b, w, x_train_batch)
		# Calculate error
		error = yhat - y_train_batch
		# Calculate loss using Mean Square Error (MSE)
		loss = (error ** 2).mean()
		# Calculate gradients of the parameters
		b_grad = 2 * error.mean()
		w_grad = 2 * (x_train_batch * error).mean()
		# Update the parameters
		b = b - lr * b_grad
		w = w - lr * w_grad

# ---------------------------
# NOTE this part is not part of a standard linear regression, but for educational purposes

# we have to split the ranges in 100 evenly spaced intervals each 
# NOTE: In reality true_b and true_w wouldn't be known.
b_range = np.linspace(true_b - 3.0, true_b + 3.0, 101) 
w_range = np.linspace(true_w - 3.0, true_w + 3.0, 101) 
# meshgrid is a function that generates a grid of b and w values for all combinations 
bs, ws = np.meshgrid(b_range, w_range)

# Create predictions for all possible combinations of b and w
# NOTE: Calculating all possible predictions, errors, and losses for all combinations of w and b is computationally impossible for datasets with more than 1 feature.
all_predictions = np.apply_along_axis(
        func1d=lambda x: bs + ws * x,
        axis=1,
        arr=scaled_x_train
    )
# Reshape the labels to fit the matrices of the predictions
all_labels = y_train.reshape(-1, 1, 1)
# Calculate the errors for all possible predictions
all_errors = (all_predictions - all_labels)
# Calculate the loss function (The mean square of the error)
# This results in the "loss surface". A 3D space Where x=w, y=b, and z=loss value.
all_losses = (all_errors ** 2).mean(axis=0)
# ---------------------------

# Plotting

plt.grid(True)

# Plot the dataset
plt.scatter(scaled_x_train, y_train)
plt.scatter(scaled_x_val, y_val)

# Plot the initial untrained randomized prediction
plt.plot(scaled_x_train, predict(np.random.randn(1), np.random.randn(1), scaled_x_train))
#plt.scatter(x_train, error)

# Plot the final trained model
plt.plot(scaled_x_train, predict(b, w, scaled_x_train))

# Plot the loss function as a gradient
contours = plt.contour(bs, ws, all_losses, 20)
plt.clabel(contours, inline=True, fontsize=8)
cbar = plt.colorbar()
cbar.set_label('Loss values')

plt.show()
