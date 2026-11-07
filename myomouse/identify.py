"""Find both bands over Bluetooth and record which one is on which arm (writes bands.json).

    python -m myomouse.identify

Wear both bands. Each band vibrates in turn and you type which arm felt it.
Make sure no other program (Myo Connect, myo_controller.py) is connected to the bands.
"""

import asyncio
import json
import sys

from . import config as C
from .device import COMMAND_UUID, CMD_NEVER_SLEEP, cmd_vibrate, discover_myos


async def buzz(address):
    from bleak import BleakClient, BleakScanner

    dev = await BleakScanner.find_device_by_address(address, timeout=10.0)
    if dev is None:
        print(f"  could not reconnect to {address}")
        return False
    async with BleakClient(dev) as client:
        await client.write_gatt_char(COMMAND_UUID, CMD_NEVER_SLEEP, response=True)
        for _ in range(3):
            await client.write_gatt_char(COMMAND_UUID, cmd_vibrate(2), response=True)
            await asyncio.sleep(0.7)
    return True


def ask(prompt):
    """Ask l/r/s. Ignores keys typed while the band was still buzzing; accepts 'left' etc."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass
    while True:
        ans = input(prompt).strip().lower()[:1]
        if ans in ("l", "r", "s"):
            return ans


def main():
    print("scanning for Myos over Bluetooth for 6 s... (tap a band's logo to wake it)")
    found = asyncio.run(discover_myos(6.0))
    if not found:
        raise SystemExit("No Myo found. Check Bluetooth is on, the terminal has Bluetooth permission "
                         "(System Settings > Privacy & Security > Bluetooth), and the bands are awake.")
    for addr, name, rssi in found:
        print(f"  {addr}  name={name!r}  rssi={rssi}")

    bands = {}
    for addr, name, _ in found:
        print(f"\nbuzzing {name} ({addr}) ...")
        if not asyncio.run(buzz(addr)):
            continue
        ans = ask(f"which arm vibrated ({name})? [l]eft / [r]ight / [s]kip: ")
        if ans != "s":
            bands["left" if ans == "l" else "right"] = addr

    C.BANDS_FILE.write_text(json.dumps(bands, indent=2))
    print(f"\nsaved {C.BANDS_FILE}: {bands}")
    missing = {"left", "right"} - bands.keys()
    if missing:
        print("missing:", ", ".join(sorted(missing)), "- rerun when that band is awake.")


if __name__ == "__main__":
    main()
