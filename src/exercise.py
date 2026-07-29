import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Getting the absolute path of exercise.py file
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data")) # Getting the path to the 'data' folder, which contains this file
    
    csv_path = os.path.join(data_dir, 'COVID19_Statistics_200_Rows-1.csv') # Accessing the dataset file
        
    df = pd.read_csv(csv_path) # read_csv is used to read the dataset file

    # Dropping structural tracking elements and the target
    X = df.drop(columns=['Record_ID', 'Date', 'Deaths']) 
    y = df['Deaths'].values

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int) # Converting categories with text to binary numeric values

    # Splitting the data rows for developing the model and for evaluation
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Training and initializing the Linear Regression model
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    print(f"Baseline Linear Regression Test Mean Absolute Error: {mean_absolute_error(y_test, baseline_predictions):.2f} deaths")

    scaler = StandardScaler() # Initializing the StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # fit_transform is used to train data to prevent data leakage
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Converting numpy arrays to PyTorch FloatTensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train).unsqueeze(1)) # unsqueeze(1) is used to convert vectors from a one-dimensional structure to a two-dimensional structure
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test).unsqueeze(1))

    class CovidRegressionNet(nn.Module):
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.1):
            super(CovidRegressionNet, self).__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim1) # Accepts input dimension maps and scales them to hidden_dim1
            self.act1 = nn.ReLU() # ReLU is used for learning non-linear shapes
            self.dropout1 = nn.Dropout(dropout_rate) # Convets weights to 0 to prevent feature co-adaptation
            
            self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.act2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            
            self.output_layer = nn.Linear(hidden_dim2, 1)
            
        def forward(self, x):
            x = self.dropout1(self.act1(self.layer1(x)))
            x = self.dropout2(self.act2(self.layer2(x)))
            return self.output_layer(x)

    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    EPOCHS = 150
    HIDDEN_1 = 64
    HIDDEN_2 = 32
    DROPOUT = 0.1

    # Loading data to optimize batch memory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Isolating parameters to dynamically construct the network layers
    input_dim = X_train_scaled.shape[1]
    model = CovidRegressionNet(input_dim, HIDDEN_1, HIDDEN_2, dropout_rate=DROPOUT)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam adjusts unique learning step limits for the node weights

    train_losses = []
    val_losses = []

    print("Starting Network Training")
    for epoch in range(EPOCHS):
        model.train() # Activates structural regularizations
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() # Separates calculations from previous batch loops
            predictions = model(batch_X) # Passes elements forward
            loss = criterion(predictions, batch_y) # Assess target metric values
            loss.backward() # Calculates gradients backwards
            optimizer.step() # Alters parameters based on optimization guidelines
            running_train_loss += loss.item() * batch_X.size(0)
            
        model.eval() # Removes structural regularizations
        with torch.no_grad(): # Deactivates graph tracking history
            val_X, val_y = val_dataset.tensors
            val_preds = model(val_X)
            val_loss = criterion(val_preds, val_y)
            running_val_loss = val_loss.item() * val_X.size(0)
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_val_loss = running_val_loss / len(val_dataset)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS}, Train MSE Loss: {epoch_train_loss:.2f}, Val MSE Loss: {epoch_val_loss:.2f}")

    # Plotting the chart
    chart_path = os.path.join(script_dir, "loss_curves.png")
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss', color='teal')
    plt.plot(val_losses, label='Validation Loss', color='magenta', linestyle='--')
    plt.title('COVID-19 Model Convergence Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(chart_path, dpi=150)
    plt.close()

    model.eval()
    with torch.no_grad():
        test_X, test_y = test_dataset.tensors
        test_predictions = model(test_X).numpy()
        actual_deaths = test_y.numpy()

    final_mae = mean_absolute_error(actual_deaths, test_predictions)
    final_r2 = r2_score(actual_deaths, test_predictions)

    print(f"Neural Network Final Test Mean Absolute Error: {final_mae:.2f} deaths")
    print(f"Neural Network Variance (R2 Score): {final_r2:.2%}")

if __name__ == "__main__":
    main()