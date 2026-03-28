# ============================
# MERGED COLLECTOR (ANNOTATED)
# BASE: data_collection_imu (508 lines)
# ADDITION: Wrist IMU (ESP32 serial) threaded, PC timestamp, aligned, per-trial CSV
# ============================

import asyncio
import struct
import time
import os
import csv
import math
import traceback
from datetime import datetime
from bleak import BleakScanner, BleakClient
from pylsl import StreamInlet, resolve_streams
import keyboard
import nest_asyncio
import logging

# ==== NEW IMPORTS FOR WRIST IMU THREAD ====
import serial
import threading
from queue import Queue, Empty

nest_asyncio.apply()
logging.getLogger("bleak").setLevel(logging.WARNING)


# -----------------------
# CONFIG
# -----------------------
TRIAL_COUNT = 5
RECORD_DURATION = 6             # seconds per trial
REST_BETWEEN_TRIALS = 1         # seconds rest
WAIT_AFTER_SPACE = 0

BASE_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "EMG_IMU_Test_Trials_13_02_26_set7")

EMG_NAME = "Myo"
MYO_COMMAND_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"
MYO_EMG_UUIDS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842"
]
IMU_UUID_PRIMARY   = "d5060205-a904-deb9-4748-2c7f4a124842"
IMU_UUID_ALTERNATE = "d5060402-a904-deb9-4748-2c7f4a124842"

# reconnect behaviour
RECONNECT_ATTEMPTS = 3
RECONNECT_WAIT = 1.5  # seconds

# ==== NEW CONFIG FOR WRIST IMU ====
WRIST_PORT = "COM3"
WRIST_BAUD = 115200
WRIST_TIMEOUT = 0.01    # small non-blocking read timeout
WRIST_FILENAME = "imu_wrist_data.csv"


# -----------------------
# GLOBALS
# -----------------------
notif_queue: asyncio.Queue = asyncio.Queue()
_current_emg_container = None
_current_imu_container = None
_current_eeg_container = None
_current_eeg_channels = None
_processor_stop_event = None
_started_notifications = set()
_observed_imu_packet_sizes = {}

##disconnection globals
_abort_all = False
_disconnect_during_trial = False

# EEG globals
_chosen_stream = None
_eeg_inlet = None
_eeg_channel_count = None

# ==== NEW GLOBALS FOR WRIST IMU ====
_wrist_thread = None
_wrist_thread_stop = None
_wrist_queue = None   # queue for parsed wrist IMU samples (timestamp, ax, ay, az, gx, gy, gz)


# -----------------------
# HELPERS: file + CSV
# -----------------------
def ensure_folder():
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)
        print(f"Created base folder at: {BASE_FOLDER}")

def get_next_trial_number():
    if not os.path.exists(BASE_FOLDER):
        return 1

    max_trial = 0
    for name in os.listdir(BASE_FOLDER):
        if name.startswith("trial_"):
            try:
                num = int(name.split("_")[1])
                max_trial = max(max_trial, num)
            except:
                continue

    return max_trial + 1


def save_to_csv(data, filename, trial_num):
    filepath = os.path.join(BASE_FOLDER, f"trial_{trial_num:02d}", filename)
    headers = list(data.keys())
    lengths = [len(data[h]) for h in headers]
    min_len = min(lengths) if lengths else 0
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(min_len):
            row = []
            for h in headers:
                v = data[h][i]
                if isinstance(v, float) and math.isnan(v):
                    row.append('nan')
                else:
                    row.append(v)
            writer.writerow(row)
    print(f"Saved: {filepath}")

# ---- EMG CONTAINER FACTORY (original) ----
def make_emg_container():
    d = {'timestamp': []}
    for i in range(8):
        d[f'emg{i+1}'] = []
    return d

# ---- MYO IMU CONTAINER FACTORY (original) ----
def make_imu_container():
    return {
        'timestamp': [],
        'qw': [], 'qx': [], 'qy': [], 'qz': [],
        'acc_x': [], 'acc_y': [], 'acc_z': [],
        'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
        'raw_hex': [], 'src_uuid': []
    }

# ---- EEG CONTAINER FACTORY (original) ----
def make_eeg_container(ch_count):
    d = {'timestamp': []}
    for i in range(ch_count):
        d[f'ch{i+1}'] = []
    return d


# ==== NEW: WRIST IMU CONTAINER FACTORY ====
def make_wrist_container():
    return {
        "timestamp": [],
        "acc_x": [],
        "acc_y": [],
        "acc_z": [],
        "gyro_x": [],
        "gyro_y": [],
        "gyro_z": []
    }


# -----------------------
# IMU decoding (existing Myo IMU)
# -----------------------
def decode_imu_packet(raw_bytes):
    L = len(raw_bytes)
    if L >= 20:
        try:
            w = struct.unpack('<10h', raw_bytes[:20])
        except struct.error:
            return None
        qw,qx,qy,qz = (w[0]/16384.0, w[1]/16384.0, w[2]/16384.0, w[3]/16384.0)
        ax,ay,az = (w[4]/2048.0, w[5]/2048.0, w[6]/2048.0)
        gx,gy,gz = (w[7]/16.0, w[8]/16.0, w[9]/16.0)
        return {'qw':qw,'qx':qx,'qy':qy,'qz':qz,
                'acc_x':ax,'acc_y':ay,'acc_z':az,
                'gyro_x':gx,'gyro_y':gy,'gyro_z':gz}
    if L == 16:
        try:
            w = struct.unpack('<8h', raw_bytes)
        except struct.error:
            return None
        qw,qx,qy,qz = (w[0]/16384.0, w[1]/16384.0, w[2]/16384.0, w[3]/16384.0)
        ax,ay,az = (w[4]/2048.0, w[5]/2048.0, w[6]/2048.0)
        gx = w[7]/16.0
        return {'qw':qw,'qx':qx,'qy':qy,'qz':qz,
                'acc_x':ax,'acc_y':ay,'acc_z':az,
                'gyro_x':gx,'gyro_y':math.nan,'gyro_z':math.nan}
    return None


# -----------------------
# Wrist IMU worker thread
# -----------------------
# Parses lines like:
# $IMU <t_esp> <acc_x> <acc_y> <acc_z> <gyro_x> <gyro_y> <gyro_z>
# but we ignore ESP timestamp and use PC timestamps for alignment

def wrist_worker(port, baud, stop_event, out_queue):
    try:
        ser = serial.Serial(port, baud, timeout=WRIST_TIMEOUT)
    except Exception as e:
        print(f"[WRIST] Failed to open {port}: {e}")
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except Exception:
            continue
        if not line.startswith("$IMU"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        try:
            # parts[1] is ESP time but we're not using it
            ax = float(parts[2])
            ay = float(parts[3])
            az = float(parts[4])
            gx = float(parts[5])
            gy = float(parts[6])
            gz = float(parts[7])
            ts = datetime.now().strftime('%H:%M:%S.%f')
            out_queue.put((ts, ax, ay, az, gx, gy, gz))
        except Exception:
            continue

    try:
        ser.close()
    except Exception:
        pass
# -----------------------
# Silent BLE handlers
# -----------------------
def make_notify_handler_with_uuid(uuid_str, kind):
    def handler(sender, data):
        try:
            notif_queue.put_nowait((kind, uuid_str, datetime.now(), bytes(data)))
        except Exception:
            pass
    return handler


# -----------------------
# Background processor (existing Myo EMG & IMU)
# -----------------------
async def processor_task(stop_event: asyncio.Event):
    global _current_emg_container, _current_imu_container, _observed_imu_packet_sizes
    while not stop_event.is_set():
        try:
            kind, uuid_str, ts, raw = await asyncio.wait_for(notif_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        ts_str = ts.strftime('%H:%M:%S.%f')
        try:
            if kind == 'emg' and _current_emg_container is not None:
                try:
                    vals = struct.unpack('<16b', raw)
                except Exception:
                    vals = tuple(raw)
                _current_emg_container['timestamp'].append(ts_str)
                for i in range(8):
                    _current_emg_container[f'emg{i+1}'].append(vals[i] if i < len(vals) else 0)

            elif kind == 'imu' and _current_imu_container is not None:
                _current_imu_container['timestamp'].append(ts_str)
                hx = raw.hex()
                _current_imu_container['raw_hex'].append(hx)
                _current_imu_container['src_uuid'].append(uuid_str)
                _observed_imu_packet_sizes.setdefault(uuid_str, set()).add(len(raw))
                dec = decode_imu_packet(raw)
                if dec:
                    _current_imu_container['qw'].append(dec['qw'])
                    _current_imu_container['qx'].append(dec['qx'])
                    _current_imu_container['qy'].append(dec['qy'])
                    _current_imu_container['qz'].append(dec['qz'])
                    _current_imu_container['acc_x'].append(dec['acc_x'])
                    _current_imu_container['acc_y'].append(dec['acc_y'])
                    _current_imu_container['acc_z'].append(dec['acc_z'])
                    _current_imu_container['gyro_x'].append(dec['gyro_x'])
                    _current_imu_container['gyro_y'].append(dec['gyro_y'])
                    _current_imu_container['gyro_z'].append(dec['gyro_z'])
                else:
                    for k in ['qw','qx','qy','qz','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']:
                        _current_imu_container[k].append(math.nan)

        except Exception:
            traceback.print_exc()
        finally:
            try:
                notif_queue.task_done()
            except Exception:
                pass


# -----------------------
# Start/stop Myo notifications
# -----------------------
async def start_notifications(client):
    started = set()
    for u in MYO_EMG_UUIDS:
        try:
            await client.start_notify(u, make_notify_handler_with_uuid(u, 'emg'))
            started.add(u)
            await asyncio.sleep(0.02)
        except Exception:
            print(f"[WARN] start_notify EMG failed: {u}")
    for u in (IMU_UUID_PRIMARY, IMU_UUID_ALTERNATE):
        try:
            await client.start_notify(u, make_notify_handler_with_uuid(u, 'imu'))
            started.add(u)
            await asyncio.sleep(0.02)
        except Exception:
            pass
    print(f"[INFO] Notifications started on {len(started)} characteristics.")
    return started

async def stop_notifications_safe(client, started_set):
    for u in list(started_set):
        try:
            await client.stop_notify(u)
        except KeyError:
            pass
        except Exception:
            print(f"[WARN] stop_notify failed for {u}")
    print("[INFO] Notifications stopped.")


# -----------------------
# Reconnect helper
# -----------------------
async def try_reconnect(device, attempts=RECONNECT_ATTEMPTS):
    for attempt in range(1, attempts+1):
        try:
            client = BleakClient(device)
            await client.connect()
            if client.is_connected:
                print(f"[INFO] Reconnected to {device.address} (attempt {attempt})")
                return client
            else:
                try:
                    await client.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(RECONNECT_WAIT)
    return None


# -----------------------
# MAIN
# -----------------------
async def main():
    global _current_emg_container, _current_imu_container, _current_eeg_container, _current_eeg_channels
    global _processor_stop_event, _started_notifications
    global _chosen_stream, _eeg_inlet, _eeg_channel_count
    global _wrist_thread, _wrist_thread_stop, _wrist_queue

    ensure_folder()
    print("\nPress SPACE to start trials...")
    keyboard.wait('space')
    await asyncio.sleep(WAIT_AFTER_SPACE)

    # ==== EEG SELECTION (unchanged) ====
    print("Searching for EEG stream...")
    try:
        streams = resolve_streams()
    except Exception:
        streams = []
    if not streams:
        print("No LSL EEG found.")
        return

    streams_by_src = {}
    for s in streams:
        try:
            src = s.source_id()
        except Exception:
            src = None
        if src not in streams_by_src:
            streams_by_src[src] = s

    chosen_stream = None
    for src, s in streams_by_src.items():
        try:
            if ('EEG' in s.name()) or ('EEG' in s.type()):
                chosen_stream = s
                break
        except Exception:
            continue
    if chosen_stream is None:
        chosen_stream = next(iter(streams_by_src.values()))
    try:
        print(f"[INFO] EEG={chosen_stream.name()} source={chosen_stream.source_id()} chans={chosen_stream.channel_count()}")
    except Exception:
        pass

    try:
        inlet = StreamInlet(chosen_stream, max_chunklen=12)
    except Exception as e:
        print("Failed to create EEG inlet:", e)
        return

    initial_sample = None
    for _ in range(8):
        try:
            sample,_ = inlet.pull_sample(timeout=1.0)
        except Exception:
            sample = None
        if sample is not None:
            initial_sample = sample
            break
        await asyncio.sleep(0.3)

    if initial_sample is None:
        try:
            ch_count = chosen_stream.channel_count()
        except Exception:
            ch_count = 24
    else:
        ch_count = len(initial_sample)

    _chosen_stream = chosen_stream
    _eeg_inlet = inlet
    _eeg_channel_count = ch_count


    # ==== MYO SCAN + CONNECT ====
    print("Scanning for Myo...")
    device = await BleakScanner.find_device_by_name(EMG_NAME, timeout=10.0)
    if device is None:
        print("Myo not found.")
        return

    client = BleakClient(device)
    try:
        await client.connect()
    except Exception as e:
        print("Myo connect failed:", e)
        return
    if not client.is_connected:
        print("Couldn't connect to Myo.")
        return
    print(f"Connected to Myo: {device.address}")

    # Unlock & enable streaming
    try:
        await client.write_gatt_char(MYO_COMMAND_UUID, struct.pack('<3B', 0x0A, 0x01, 0x01))
        await asyncio.sleep(0.08)
        await client.write_gatt_char(MYO_COMMAND_UUID, struct.pack('<5B', 0x01, 3, 3, 1, 1))
        await asyncio.sleep(0.08)
    except Exception:
        print("[WARN] unlock/enable failed")

    # Start notifications & processor
    _started_notifications = await start_notifications(client)
    _processor_stop_event = asyncio.Event()
    processor = asyncio.create_task(processor_task(_processor_stop_event))


    # ============================
    # TRIAL LOOP + WRIST INTEGRATION
    # ============================
    trial = get_next_trial_number()
    valid_trials_done = 0
    while valid_trials_done < TRIAL_COUNT:
        try:
            if not client.is_connected:
                print(f"[WARN] Myo disconnected before trial {trial}, reconnecting...")
                new_client = await try_reconnect(device)
                if new_client is None:
                    print("Reconnect failed. Aborting.")
                    break
                try:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                    client = new_client
                    _started_notifications = await start_notifications(client)
                except Exception as e:
                    print("Failed to restart notifications:", e)
                    break

            print(f"\n=== Trial {trial} (session {valid_trials_done+1}/{TRIAL_COUNT}) — {RECORD_DURATION}s ===")

            # INITS
            _current_emg_container = make_emg_container()
            _current_imu_container = make_imu_container()
            _current_eeg_container = make_eeg_container(_eeg_channel_count)
            _current_eeg_channels = [_current_eeg_container[f'ch{i+1}'] for i in range(_eeg_channel_count)]
            _current_wrist_container = make_wrist_container()
            print(f"[INFO] Starting trial {trial}")

            # Insert initial EEG sample if present
            if initial_sample is not None and valid_trials_done == 0:
                ts_str = datetime.now().strftime('%H:%M:%S.%f')
                _current_eeg_container['timestamp'].append(ts_str)
                for i,v in enumerate(initial_sample):
                    if i < _eeg_channel_count:
                        _current_eeg_channels[i].append(v)
                initial_sample = None

            # ==== START WRIST THREAD FOR THIS TRIAL ====
            _wrist_queue = Queue()
            _wrist_thread_stop = threading.Event()
            _wrist_thread = threading.Thread(
                target=wrist_worker,
                args=(WRIST_PORT, WRIST_BAUD, _wrist_thread_stop, _wrist_queue),
                daemon=True
            )
            _wrist_thread.start()

            # ==== RECORD LOOP ====
            start_time = time.time()
            last_status = time.time()
            _disconnect_during_trial = False  # reset at start of trial
            while time.time() - start_time < RECORD_DURATION:
                
                #  CHECK HERE (TOP of loop)
                if not client.is_connected:
                    print("\n[FATAL] Myo disconnected DURING trial. Terminating.")
                    _disconnect_during_trial = True
                    _abort_all = True
                    break
                # EEG pull
                try:
                    eeg_sample,_ = _eeg_inlet.pull_sample(timeout=0.0)
                except Exception:
                    eeg_sample = None
                if eeg_sample:
                    ts_str = datetime.now().strftime('%H:%M:%S.%f')
                    _current_eeg_container['timestamp'].append(ts_str)
                    for i,v in enumerate(eeg_sample):
                        if i < _eeg_channel_count:
                            _current_eeg_channels[i].append(v)

                # WRIST queue drain
                while True:
                    try:
                        ts,ax,ay,az,gx,gy,gz = _wrist_queue.get_nowait()
                    except Empty:
                        break
                    _current_wrist_container['timestamp'].append(ts)
                    _current_wrist_container['acc_x'].append(ax)
                    _current_wrist_container['acc_y'].append(ay)
                    _current_wrist_container['acc_z'].append(az)
                    _current_wrist_container['gyro_x'].append(gx)
                    _current_wrist_container['gyro_y'].append(gy)
                    _current_wrist_container['gyro_z'].append(gz)

                # Status once/sec
                if time.time() - last_status >= 1.0:
                    last_status = time.time()
                    emg_count = len(_current_emg_container['timestamp'])
                    imu_count = len(_current_imu_container['timestamp'])
                    eeg_count = len(_current_eeg_container['timestamp'])
                    wrist_count = len(_current_wrist_container['timestamp'])
                    print(f"[rec] EEG={eeg_count} EMG={emg_count} IMU={imu_count} WRIST={wrist_count}")

                await asyncio.sleep(0.001)
            if _disconnect_during_trial:
                print("[INFO] Trial discarded due to disconnect. No data saved.")
                try:
                    _wrist_thread_stop.set()
                    _wrist_thread.join(timeout=1.0)
                except Exception:
                    pass
                break
            # ==== STOP WRIST THREAD ====
            _wrist_thread_stop.set()
            _wrist_thread.join(timeout=1.0)
            # final drain
            while True:
                try:
                    ts,ax,ay,az,gx,gy,gz = _wrist_queue.get_nowait()
                except Empty:
                    break
                _current_wrist_container['timestamp'].append(ts)
                _current_wrist_container['acc_x'].append(ax)
                _current_wrist_container['acc_y'].append(ay)
                _current_wrist_container['acc_z'].append(az)
                _current_wrist_container['gyro_x'].append(gx)
                _current_wrist_container['gyro_y'].append(gy)
                _current_wrist_container['gyro_z'].append(gz)


            if _disconnect_during_trial:
                print("[INFO] Trial discarded due to disconnect. No data saved.")
                break

            # ==== SAVE ALL ====
            # ==== VALIDATION ====
            emg_count = len(_current_emg_container['timestamp'])

            if emg_count < 550:
                print(f"[SKIP] Trial {trial} discarded (EMG={emg_count} < 550)")
                print("[INFO] Retrying same trial number...")
                continue

            # ==== CREATE FOLDER ONLY NOW ====
            trial_folder = os.path.join(BASE_FOLDER, f"trial_{trial:02d}")
            os.makedirs(trial_folder, exist_ok=True)

            # ==== SAVE ALL ====
            save_to_csv(_current_eeg_container, "eeg_data.csv", trial)
            save_to_csv(_current_emg_container, "emg_data.csv", trial)
            save_to_csv(_current_imu_container, "imu_data.csv", trial)
            save_to_csv(_current_wrist_container, WRIST_FILENAME, trial)

            valid_trials_done += 1
            trial += 1

            # REST
            if valid_trials_done < TRIAL_COUNT:
                print(f"Resting {REST_BETWEEN_TRIALS}s...")
                await asyncio.sleep(REST_BETWEEN_TRIALS)

        except Exception as e:
            print(f"[ERROR] During trial {trial}: {e}")
            traceback.print_exc()
            
            break

    if _abort_all:
        print("\n[EXIT] Program terminated due to Myo disconnect during trial.")
        return

    # ==== CLEANUP ====
    try:
        await stop_notifications_safe(client, _started_notifications)
    except Exception:
        pass

    if _processor_stop_event is not None:
        _processor_stop_event.set()
    await asyncio.sleep(0.05)
    try:
        processor.cancel()
    except Exception:
        pass

    if _observed_imu_packet_sizes:
        print("[INFO] Observed Myo IMU packet sizes:")
        for u,sizes in _observed_imu_packet_sizes.items():
            print(f"  {u} -> {sorted(list(sizes))}")

    print("Trials completed.")
# -----------------------
# ENTRY
# -----------------------
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")