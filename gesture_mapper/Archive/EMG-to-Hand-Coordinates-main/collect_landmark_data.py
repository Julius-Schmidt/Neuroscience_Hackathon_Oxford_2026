# collect_landmark_data.py
import asyncio
import pandas as pd
import uuid
import random
from datetime import datetime
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue, SleepMode
from constants import MYO_ADDRESS

# You can adjust these constants or import them from your constants file 
COLLECTION_TIME = 5  # Seconds to collect data per repetition

# CSV dataset path for landmark coordinates (update this path as needed)
LANDMARKS_DATASET_PATH = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\hand_landmarks_claw.csv"

# Define the expected landmark column names (63 columns)
landmark_names = [
    "Wrist_x", "Wrist_y", "Wrist_z",
    "Thumb_CMC_x", "Thumb_CMC_y", "Thumb_CMC_z",
    "Thumb_MCP_x", "Thumb_MCP_y", "Thumb_MCP_z",
    "Thumb_IP_x", "Thumb_IP_y", "Thumb_IP_z",
    "Thumb_Tip_x", "Thumb_Tip_y", "Thumb_Tip_z",
    "Index_MCP_x", "Index_MCP_y", "Index_MCP_z",
    "Index_PIP_x", "Index_PIP_y", "Index_PIP_z",
    "Index_DIP_x", "Index_DIP_y", "Index_DIP_z",
    "Index_Tip_x", "Index_Tip_y", "Index_Tip_z",
    "Middle_MCP_x", "Middle_MCP_y", "Middle_MCP_z",
    "Middle_PIP_x", "Middle_PIP_y", "Middle_PIP_z",
    "Middle_DIP_x", "Middle_DIP_y", "Middle_DIP_z",
    "Middle_Tip_x", "Middle_Tip_y", "Middle_Tip_z",
    "Ring_MCP_x", "Ring_MCP_y", "Ring_MCP_z",
    "Ring_PIP_x", "Ring_PIP_y", "Ring_PIP_z",
    "Ring_DIP_x", "Ring_DIP_y", "Ring_DIP_z",
    "Ring_Tip_x", "Ring_Tip_y", "Ring_Tip_z",
    "Pinky_MCP_x", "Pinky_MCP_y", "Pinky_MCP_z",
    "Pinky_PIP_x", "Pinky_PIP_y", "Pinky_PIP_z",
    "Pinky_DIP_x", "Pinky_DIP_y", "Pinky_DIP_z",
    "Pinky_Tip_x", "Pinky_Tip_y", "Pinky_Tip_z"
]

async def collect_pose_data(myo, pose_name, landmarks_list, collection_time):
    """
    Collects EMG data while the user holds a specific hand pose.
    For each EMG reading, a random set of 63 landmark values is chosen
    from a provided landmarks dataset (landmarks_list) and attached to the sample.
    """
    data = []
    print(f"\nPreparing to collect data for pose '{pose_name}'...")
    print("Get ready in 2 seconds...")
    await asyncio.sleep(2)
    print(f"START - Hold the '{pose_name}' pose")

    try:
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data: EmgValue):
            current_time = datetime.now()
            timestamp = current_time.timestamp()
            sample = {
                'id': str(uuid.uuid4()),
                'time': timestamp,
                'pose_name': pose_name,
                **{f's{i+1}': value for i, value in enumerate(emg_data)}
            }
            # Select a random row of landmark coordinates from the dataset
            random_landmark = random.choice(landmarks_list)
            for key in landmark_names:
                sample[key] = random_landmark[key]
            data.append(sample)

        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        await asyncio.sleep(collection_time)
        print(f"DONE - Data for pose '{pose_name}' collected")
        return data

    except Exception as e:
        print(f"Error during data collection for pose {pose_name}: {str(e)}")
        return []

async def main():
    all_data = []
    start_time = datetime.now()

    # Load the landmarks dataset CSV.
    # The header (line 1) is used for column names, and all subsequent rows are landmark data.
    try:
        landmarks_df = pd.read_csv(LANDMARKS_DATASET_PATH)
        # Convert the DataFrame to a list of dictionaries for fast random selection
        landmarks_list = landmarks_df.to_dict('records')
        if not landmarks_list:
            raise ValueError("Landmarks dataset is empty.")
    except Exception as e:
        raise RuntimeError(f"Failed to load landmarks dataset: {str(e)}")

    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")

    async with Myo(myo_device) as myo:
        print("Starting landmark data collection session...")
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)

        # Loop: prompt the user for hand poses until they type 'done'
        while True:
            pose_name = input("Enter pose name (or 'done' to finish): ").strip()
            if pose_name.lower() == 'done' or pose_name == "":
                break

            # Inform the user that random landmark coordinates will be used.
            print(f"Collecting EMG data for pose '{pose_name}' using random landmark coordinates from the dataset.")

            # Collect multiple repetitions for the same pose (e.g., 5 repetitions)
            for repetition in range(5):
                print(f"\nPerform pose '{pose_name}', repetition {repetition + 1}/5")
                try:
                    # Reset EMG mode before collection
                    await myo.set_mode(emg_mode=None)
                    await asyncio.sleep(0.5)
                    
                    pose_data = await collect_pose_data(
                        myo,
                        pose_name,
                        landmarks_list,
                        COLLECTION_TIME
                    )
                    all_data.extend(pose_data)
                    
                    # Vibrate to indicate end of collection
                    await myo.vibrate2((100, 200), (50, 255))
                    await myo.set_mode(emg_mode=None)
                except Exception as e:
                    print(f"Error processing pose {pose_name}: {str(e)}")
                    continue

                print("\nTake a short break - 5 seconds until next repetition")
                await asyncio.sleep(5)
            
            print("\nTake a break - 5 seconds before next pose")
            await asyncio.sleep(5)

        if not all_data:
            raise RuntimeError("No data was collected")

        # Define column order: id, time, pose_name, s1...s8, then 63 landmark columns
        columns = ['id', 'time', 'pose_name'] + [f's{i}' for i in range(1, 9)] + landmark_names
        df = pd.DataFrame(all_data)
        df = df[columns]

        # Replace the random IDs with sequential ones
        df['id'] = range(len(df))

        timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
        filename = r"C:\Users\blacb\Desktop\MYOBandProject\PythonFiles\emg_landmarks.csv"
        df.to_csv(filename, index=False)
        print(f"\nAll landmark data saved to {filename}")
        print(f"Total samples collected: {len(df)}")

        print("\nSummary of collected data:")
        summary = df.groupby('pose_name').size()
        for pose, count in summary.items():
            print(f"{pose}: {count} samples")

if __name__ == "__main__":
    asyncio.run(main())
