import asyncio
import numpy as np
import pickle
import tensorflow as tf
from bleak import BleakScanner
from pymyo import Myo
from constants import MYO_ADDRESS
from pymyo.types import EmgMode, EmgValue, SleepMode
import socket

# Paths to your saved model and metadata (scaler and EMG column names)
MODEL_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\model\model.h5"
METADATA_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\model\landmark_metadata.pkl"

async def main():
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1)

    # Load the scaler and list of EMG column names from metadata
    with open(METADATA_PATH, "rb") as f:
        scaler, emg_columns = pickle.load(f)
    print("Scaler and metadata loaded.")

    # Locate the MYO device
    print("Searching for MYO device...")
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find MYO device with address {MYO_ADDRESS}")

    async with Myo(myo_device) as myo:
        print("Connected to MYO band. Listening for EMG data...")

        # Define the callback that will be triggered on each new EMG reading.
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data: EmgValue):
            # Convert the incoming EMG tuple to a numpy array (shape: 1x8)
            emg_array = np.array(emg_data).reshape(1, -1)

            # Scale the input using the loaded scaler
            emg_scaled = scaler.transform(emg_array)

            # Get prediction from the model (output shape: (1, 63))
            prediction = model.predict(emg_scaled)

            # Get predicted coordinates as a list of floats
            pred_coords = prediction[0]

            # Create a comma-separated string of predicted coordinates
            coords_str = ", ".join([f"{x:.4f}" for x in pred_coords])
            print(coords_str)

            # Sends string to godot
            client_socket.sendto(coords_str.encode(), ("127.0.0.1", 5051))
            


        # Set the MYO band to stream smoothed EMG data
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        print("MYO band is set to stream EMG data.")

        # Keep the event loop running to continuously receive EMG data.
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Terminating live prediction...")

if __name__ == "__main__":
    asyncio.run(main())
