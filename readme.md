This repository contains the source code for one of my internship tasks centered on Deep Learning. To proceed with this task, I chose a Kaggle dataset focusing on COVID-19 statistics, such as ("Recovered", "Confirmed", etc.)

## 1. Architecture Choices
To predict daily COVID-19 deaths as a continuous number, I built a custom Multi-Layer Perceptron (MLP) regression model using PyTorch. 
- Before handing the data to the network, I converted the text categories (Country, State) into binary numbers using `pd.get_dummies()`. 
- I chose to work with a simple two-layer network. The first layer uses 64 neurons with a ReLU function to catch complex, curved patterns, and a Dropout layer to prevent the model from overfitting. The second layer uses 32 neurons to spot deeper connections between testing numbers, active cases, and deaths.
- Since this is a regression problem, I used a single linear output neuron with no activation function. This lets the model freely predict any positive number across a continuous scale.
- I split the data into 60% training, 20% validation, and 20% testing. To avoid data leakage, I scaled the features by fitting the StandardScaler only on the training set. Then, I used it to transform the validation and test sets into clean, standard normal distributions.

## 2. What I Tuned
As part of this assignment, I ran multiple tests on the code to observe how performance metrics and error rates reacted:
- Experiment 1 (Baseline): I started with a 0.01 learning rate, 150 epochs, and a light 0.1 dropout. The model overfitted immediately (Training loss tanked while validation loss stayed high). (Final Test Mean Absolute Error: 1925.46 deaths).
- Experiment 2 (Adding Regularization & Pace): I dropped the learning rate to 0.001 to stabilize updates, raised dropout to 0.3 to force distributed learning, and cut training short at 80 epochs to prevent overfitting.
- Experiment 3 (Expanding Network Capacity): I expanded the network to a 128 -> 64 configuration to give the model more capacity to catch sharp, non-linear spikes in regional cases.
- Experiment 4 (Increasing Batch Size): I increased the batch size from 16 to 32 to see if processing more data at once would smooth out the noisy updates during training.

## 3. What I Learned
- A learning rate of 0.01 was too aggressive for this dataset. The loss curves jumped around because the gradient steps kept overshooting the target. Dropping it to 0.001 completely stabilized the training.
- In my first run, simple linear regression easily beat the neural network (696.41 MAE vs. 1925.46 MAE). Through this, I learned that complex models overfit quickly on small, linear datasets. To make deep learning work here, strong regularization and shorter training times are necessary.
- Checking the loss_curves.png chart between runs allowed me to see the training error pull away from the validation curve (overfitting), telling me exactly when to stop training early and raise up the dropout.