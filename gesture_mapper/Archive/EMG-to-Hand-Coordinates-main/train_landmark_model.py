# train_landmark_model.py
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define paths and constants
DATA_INPUT_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\csvEmgData"
MODEL_SAVE_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\model\model.h5"
METADATA_SAVE_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\model\landmark_metadata.pkl"

# Load CSV files from the data folder
files = list(Path(DATA_INPUT_PATH).glob("*.csv"))
if not files:
    raise ValueError("No CSV files found in the data folder.")

dfs = []
for file in files:
    df = pd.read_csv(str(file))
    dfs.append(df)
df = pd.concat(dfs, axis=0)

# Assume each CSV contains 8 EMG columns named "s1" to "s8"
# And 63 landmark columns starting from "Wrist_x" to "Pinky_TIP_z"
emg_columns = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
landmark_columns = df.columns[df.columns.get_loc("Wrist_x") : df.columns.get_loc("Pinky_Tip_z") + 1]

print("EMG columns:", emg_columns)
print("Landmark columns:", list(landmark_columns))

# Prepare features (X) and labels (y)
X = df[emg_columns].values    # Shape: (num_samples, 8)
y = df[landmark_columns].values  # Shape: (num_samples, 63)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the EMG features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Build a simple dense neural network for regression (63 outputs)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(63, activation='linear')  # 63 continuous landmark coordinates
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mse']
)

model.summary()

# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Evaluate the model
test_loss, test_mse = model.evaluate(X_test_scaled, y_test)
print("Test MSE:", test_mse)

# Save the trained model and metadata (scaler and EMG column names)
# Uncomment the following lines to save your model and metadata when ready:
model.save(MODEL_SAVE_PATH)
with open(METADATA_SAVE_PATH, "wb") as f:
    pickle.dump((scaler, emg_columns), f)
#
# print("Model and metadata saved.")
