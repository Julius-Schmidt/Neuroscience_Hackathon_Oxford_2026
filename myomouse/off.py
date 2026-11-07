"""Turn the bands off.

    python -m myomouse.off          # back to normal sleep: the band sleeps once it is still
    python -m myomouse.off --deep   # power off now; wake it again by plugging in USB

Uses the bands in bands.json, or every Myo nearby if that file is missing.
"""
import argparse
import asyncio

from .device import COMMAND_UUID, CMD_DEEP_SLEEP, CMD_NORMAL_SLEEP, CMD_STREAMS_OFF, discover_myos, load_bands


async def turn_off(address, deep):
    from bleak import BleakClient, BleakScanner

    dev = await BleakScanner.find_device_by_address(address, timeout=10.0)
    if dev is None:
        print(f"  {address}: not found (already asleep?)")
        return
    async with BleakClient(dev) as client:
        await client.write_gatt_char(COMMAND_UUID, CMD_STREAMS_OFF, response=True)
        await client.write_gatt_char(COMMAND_UUID, CMD_NORMAL_SLEEP, response=True)
        if deep:
            # The band drops the connection straight away, so don't wait for a reply.
            try:
                await client.write_gatt_char(COMMAND_UUID, CMD_DEEP_SLEEP, response=False)
            except Exception:
                pass
    print(f"  {address}: {'powered off' if deep else 'normal sleep restored'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", action="store_true", help="power off now (wake by plugging in USB)")
    args = ap.parse_args()

    try:
        addresses = list(load_bands().values())
    except FileNotFoundError:
        print("no bands.json, scanning for Myos for 6 s...")
        addresses = [a for a, _, _ in asyncio.run(discover_myos(6.0))]
    if not addresses:
        raise SystemExit("No Myo found.")
    for addr in addresses:
        asyncio.run(turn_off(addr, args.deep))


if __name__ == "__main__":
    main()
