import asyncio
import struct
import time
import pyautogui
import keyboard
from collections import deque
from bleak import BleakClient, BleakScanner

# --- CONFIGURATION ---
MYO_ADDRESS = "DD:31:D8:40:BC:22"
# MYO_ADDRESS = "FF:39:C8:DC:AC:BA"

# SENSITIVITY
JUMP_THRESHOLD = 30       # Spike (short - long baseline) to trigger click
RESET_THRESHOLD = 10      # Must drop below this to re-arm (Schmitt trigger)
REFRACTORY_PERIOD = 0.2   # Minimum seconds between clicks

# WINDOW SIZES (in samples at 200Hz)
LONG_WINDOW_SIZE = 400    # ~2 seconds  — slow-moving baseline
SHORT_WINDOW_SIZE = 20 # ~50ms       — instantaneous burst detector

# --- UUIDS ---
CMD_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG_DATA_UUID = "d5060105-a904-deb9-4748-2c7f4a124842"


def common_mode_reject(sample):
    """
    Remove the common signal across all 8 pods, then return
    the sum of absolute residuals. This rejects noise that
    appears equally on every electrode (mains hum, motion
    artifacts) while preserving localized muscle activity.
    """
    mean_val = sum(sample) / len(sample)
    residuals = [val - mean_val for val in sample]
    return sum(abs(r) for r in residuals)


class FlappyController:
    def __init__(self):
        # Two separate sliding windows at 200Hz
        self.long_window = deque(maxlen=LONG_WINDOW_SIZE)
        self.short_window = deque(maxlen=SHORT_WINDOW_SIZE)

        # Pre-fill so we don't divide by zero early on
        for _ in range(LONG_WINDOW_SIZE):
            self.long_window.append(0)
        for _ in range(SHORT_WINDOW_SIZE):
            self.short_window.append(0)

        self.last_click_time = 0
        self.is_holding_flex = False

    def _process_single_sample(self, sample):
        """
        Process one 8-channel sample through the full pipeline:
        common-mode rejection -> dual window -> threshold logic.
        Called twice per BLE packet (once per time-step at 200Hz).
        """
        # 1. Common-mode rejection across the 8 pods
        activity = common_mode_reject(sample)

        # 2. Push into both windows
        self.long_window.append(activity)
        self.short_window.append(activity)

        # 3. Compute baselines
        long_baseline = sum(self.long_window) / len(self.long_window)
        short_baseline = sum(self.short_window) / len(self.short_window)

        # 4. Spike = short-term burst minus long-term resting level
        spike = short_baseline - long_baseline

        # 5. Threshold logic with Schmitt trigger + refractory
        current_time = time.time()
        time_since_last = current_time - self.last_click_time

        if spike > JUMP_THRESHOLD and not self.is_holding_flex:
            if time_since_last > REFRACTORY_PERIOD:
                self.trigger_jump(spike)
                self.last_click_time = current_time
                self.is_holding_flex = True

        elif spike < RESET_THRESHOLD:
            self.is_holding_flex = False

        # 6. Visual feedback
        self.print_status(long_baseline, short_baseline, spike, time_since_last)

    def process_emg(self, sender, data):
        """
        BLE callback. Each packet has 16 signed bytes:
        bytes  0-7  = sample at time T      (8 pods)
        bytes  8-15 = sample at time T+5ms  (8 pods)
        Process them sequentially to maintain 200Hz rate.
        """
        try:
            raw = struct.unpack('<16b', data[:16])
        except Exception:
            return

        sample1 = raw[:8]   # Time T
        sample2 = raw[8:]   # Time T + 5ms

        # Process each independently — this is 200Hz, not 100Hz
        self._process_single_sample(sample1)
        self._process_single_sample(sample2)

    def trigger_jump(self, spike):
        pyautogui.click()
        print(f"\n*** CLICK! (Spike: {int(spike)}) ***")

    def print_status(self, long_base, short_base, spike, delta):
        bar_len = int(max(0, spike) / 2)
        bar = '█' * min(bar_len, 40)

        if delta < REFRACTORY_PERIOD:
            status = "COOLDOWN"
        elif self.is_holding_flex:
            status = "HOLDING "
        else:
            status = "READY   "

        print(
            f"[{status}] LongBase:{int(long_base):3} "
            f"ShortBase:{int(short_base):3} "
            f"Spike:{int(spike):4} | {bar:<40}",
            end="\r"
        )


async def run():
    print(f"Looking for Myo at {MYO_ADDRESS}...")
    device = await BleakScanner.find_device_by_address(MYO_ADDRESS)

    if not device:
        print("Myo not found.")
        return

    print("Connecting...")
    async with BleakClient(device) as client:
        print("Connected! Press 'q' to quit.")

        # 1. Keep Awake
        await client.write_gatt_char(CMD_UUID, bytes([0x09, 0x01, 0x01]))

        # 2. Set Mode to RAW
        # Command 0x01, length 0x03, EMG mode 0x03 (raw), IMU 0x00, classifier 0x00
        await client.write_gatt_char(CMD_UUID, bytes([0x01, 0x03, 0x03, 0x00, 0x00]))

        # 3. Start Stream
        controller = FlappyController()
        await client.start_notify(EMG_DATA_UUID, controller.process_emg)

        try:
            while True:
                if keyboard.is_pressed('q'):
                    print("\nQuitting...")
                    break
                await asyncio.sleep(0.01)
        finally:
            # Let the band fall asleep again on its own
            await client.write_gatt_char(CMD_UUID, bytes([0x09, 0x01, 0x00]))


if __name__ == "__main__":
    asyncio.run(run())