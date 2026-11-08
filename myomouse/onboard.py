"""Try the Myo's OWN gesture classifier: runs inside the armband, no training at all.

    python -m myomouse.onboard                 # the band in bands.json
    python -m myomouse.onboard --side right
    python -m myomouse.onboard --sound mypop   # also play a sound on every fist (iPhone sound switch)

The Myo firmware recognises 5 poses on the band itself: fist, wave in, wave out, fingers
spread, double tap. To turn it on you must do the SYNC GESTURE each time you put the band on:
wear it with the USB port facing your wrist, gently bend your wrist back (away from your palm,
"wave out") and hold it until the band vibrates. After that, poses stream in.

Our own LDA is trained on YOU; this one is Thalmic's general model, trained on many people,
so compare: does it catch your fists as reliably, and how many false ones does it give?
Keys: Space = reset counts, Esc = quit.
"""

import argparse
import asyncio
import queue
import struct
import threading
import time

from . import config as C
from .device import COMMAND_UUID, _uuid, cmd_vibrate, load_bands

CLASSIFIER_UUID = _uuid(0x0103)
CMD_CLASSIFIER_ON = bytes([0x01, 0x03, 0x00, 0x01, 0x01])   # emg off, imu on, classifier on
CMD_UNLOCK_HOLD = bytes([0x0A, 0x01, 0x02])                 # stay unlocked: report every pose
CMD_NEVER_SLEEP = bytes([0x09, 0x01, 0x01])
CMD_ALL_OFF = bytes([0x01, 0x03, 0x00, 0x00, 0x00])

POSES = {0: "rest", 1: "fist", 2: "wave in", 3: "wave out", 4: "fingers spread", 5: "double tap", 0xFFFF: "unknown"}


def parse_event(data: bytes):
    """Classifier event -> (kind, detail)."""
    t = data[0]
    if t == 1:
        arm = {1: "right", 2: "left"}.get(data[1], "unknown")
        xdir = {1: "USB toward wrist", 2: "USB toward elbow"}.get(data[2], "?") if len(data) > 2 else "?"
        return "synced", f"{arm} arm, {xdir}"
    if t == 2:
        return "unsynced", "band moved / taken off: do the sync gesture again"
    if t == 3:
        return "pose", POSES.get(struct.unpack("<H", bytes(data[1:3]))[0], "unknown")
    if t == 4:
        return "unlocked", ""
    if t == 5:
        return "locked", ""
    if t == 6:
        return "sync failed", "too hard? try a gentler, held wrist bend"
    if t == 7:
        return "warm-up", "done" if len(data) > 1 and data[1] == 1 else "still warming up (keep it on a minute)"
    return f"event {t}", data.hex()


def ble_thread(address, q, stop):
    async def main():
        from bleak import BleakClient, BleakScanner
        q.put(("status", f"looking for {address} ..."))
        dev = await BleakScanner.find_device_by_address(address, timeout=15.0)
        if dev is None:
            q.put(("error", "band not found: wake it (tap the logo), close other scripts / Myo Connect"))
            return
        async with BleakClient(dev) as client:
            await client.write_gatt_char(COMMAND_UUID, CMD_NEVER_SLEEP, response=True)
            await client.write_gatt_char(COMMAND_UUID, CMD_CLASSIFIER_ON, response=True)

            def on_event(_c, data):
                kind, detail = parse_event(data)
                q.put((kind, detail))
                if kind == "synced":
                    asyncio.get_event_loop().create_task(
                        client.write_gatt_char(COMMAND_UUID, CMD_UNLOCK_HOLD, response=True))

            await client.start_notify(CLASSIFIER_UUID, on_event)
            await client.write_gatt_char(COMMAND_UUID, CMD_UNLOCK_HOLD, response=True)
            await client.write_gatt_char(COMMAND_UUID, cmd_vibrate(1), response=True)
            q.put(("status", "connected - now do the sync gesture"))
            while not stop.is_set():
                await asyncio.sleep(0.1)
            try:
                await client.write_gatt_char(COMMAND_UUID, CMD_ALL_OFF, response=True)
            except Exception:
                pass
    try:
        asyncio.run(main())
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default=None, help="which band in bands.json (default: the first)")
    ap.add_argument("--sound", default=None, help="play this sound on every fist (see myomouse.sound_switch)")
    args = ap.parse_args()
    bands = load_bands()
    side = args.side or next(iter(bands))
    player = None
    if args.sound:
        from .sound_switch import Player, sound_path
        player = Player(sound_path(args.sound))

    q, stop = queue.Queue(), threading.Event()
    threading.Thread(target=ble_thread, args=(bands[side], q, stop), daemon=True).start()

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((900, 560))
    pygame.display.set_caption("Myo on-board classifier")
    big, mid, small = (pygame.font.SysFont("consolas", s) for s in (84, 30, 20))
    colors = {"fist": (90, 160, 255), "wave in": (120, 220, 140), "wave out": (240, 200, 80),
              "fingers spread": (230, 120, 200), "double tap": (255, 140, 90), "rest": (120, 125, 140)}
    pose, status, synced = "-", "starting...", False
    counts = {p: 0 for p in POSES.values() if p not in ("rest", "unknown")}
    log = []
    t0 = time.perf_counter()
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    counts = {k: 0 for k in counts}
                    t0 = time.perf_counter()
            while True:
                try:
                    kind, detail = q.get_nowait()
                except queue.Empty:
                    break
                stamp = f"{time.perf_counter() - t0:6.1f}s"
                if kind == "pose":
                    pose = detail
                    if detail in counts:
                        counts[detail] += 1
                    if detail == "fist" and player:
                        player.play()
                elif kind == "synced":
                    synced, status = True, f"synced: {detail}"
                elif kind == "unsynced":
                    synced, status, pose = False, detail, "-"
                elif kind in ("status", "error", "sync failed", "warm-up"):
                    status = f"{kind}: {detail}" if kind not in ("status",) else detail
                log.append(f"{stamp}  {kind} {detail}")
                print(f"{stamp}  {kind} {detail}", flush=True)
                log = log[-9:]
            screen.fill((18, 20, 26))
            screen.blit(mid.render(status, True, (120, 220, 140) if synced else (240, 200, 80)), (24, 20))
            if not synced:
                for i, line in enumerate(["SYNC: band on with USB toward your wrist,",
                                          "gently bend your wrist back (wave out) and HOLD",
                                          "until the band vibrates."]):
                    screen.blit(small.render(line, True, (200, 200, 210)), (24, 70 + 26 * i))
            screen.blit(big.render(pose.upper(), True, colors.get(pose, (225, 228, 235))), (24, 170))
            mins = max((time.perf_counter() - t0) / 60, 1e-6)
            screen.blit(small.render("   ".join(f"{k}: {v}" for k, v in counts.items())
                                     + f"    ({mins:.1f} min, Space resets)", True, (160, 165, 180)), (24, 290))
            for i, line in enumerate(log):
                screen.blit(small.render(line, True, (120, 125, 140)), (24, 330 + 24 * i))
            pygame.display.flip()
            clock.tick(60)
    finally:
        stop.set()
        time.sleep(0.4)
        pygame.quit()
        print("counts:", counts)


if __name__ == "__main__":
    main()
