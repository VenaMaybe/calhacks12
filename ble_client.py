import asyncio
from bleak import BleakScanner, BleakClient
import sys
import time
import pandas as pd

# --- Constants from your Arduino Sketch ---
DEVICE_NAME = "CalHacks Nano ESP32"

# The "Service" UUID
SERVICE_UUID = "026420e3-d509-4be8-bada-eda31acbd7cd"

# The Characteristic UUIDs
LED_UUID = "bf9a865a-ecf1-492c-aadc-063a2cde086a"
SENSOR_1_UUID = "bf9a865a-ecf1-492c-aadc-063a2cde086b"
SENSOR_2_UUID = "bf9a865a-ecf1-492c-aadc-063a2cde086c"

# The amount of samples per second
HZ = 2
SAMPLE_RATE = 1.0 / HZ

# ------------------------------------------

"""
This function makes sure that the BLE data communication is working by testing the reading and writing with the Arduino LED.
A successful test will have the onboard LED turn on for 3 seconds, and then turn off. 
"""


async def test_ble_communication(client):
    try:
        # Set LED to HIGH
        await client.write_gatt_char(LED_UUID, b"1")  # bytes
        # Try reading LED state
        led_state_bytes1 = await client.read_gatt_char(LED_UUID)
        led_state1 = led_state_bytes1.decode('utf-8')
        # Make sure the value of LED state read is HIGH
        assert led_state1 == "1", "The LED should be on. Reading or writing to/from the LED using BLE is not working."

        await asyncio.sleep(3)

        # Set LED to LOW
        await client.write_gatt_char(LED_UUID, b"0")
        # Try reading LED state
        led_state_bytes = await client.read_gatt_char(LED_UUID)
        led_state = led_state_bytes.decode('utf-8')
        # Make sure the value of LED state read is LOW
        assert led_state == "0", "The LED should be off. Reading or writing to/from the LED using BLE is not working."

    except Exception as e:

        # If it fails LED BLE Test, there is something wrong with the BLE and we just want to exit and fix it.
        print(f"Failed LED BLE test; Exception: {e}")
        sys.exit()

""" 
This is the main async function
"""


async def main(data_list):
    print(f"Scanning for '{DEVICE_NAME}'...")

    # Scan for the device by name
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)

    if device is None:
        print(f"Could not find device with name '{DEVICE_NAME}'")
        print("Make sure your Arduino is on and running the sketch.")
        return

    print(f"Found device: {device.name} ({device.address})")
    print("Connecting...")

    # Connect to the device
    try:
        async with BleakClient(device) as client:
            print(f"Connected to {client.address}")

            print("--- Verifying LED BLE communication ---")

            # This line of code makes sure the BLE data Communication is working as expected
            await test_ble_communication(client)

            print("--- Successfully verified BLE communication ---")

            print(f"--- Starting Data Collection (Sample Rate: {SAMPLE_RATE}s) ---")
            print("Stop running the program to stop collection and save data.")

            while True:

                # Get a timestamp
                current_time = time.time()

                # Read Sensor 1
                s1_bytes = await client.read_gatt_char(SENSOR_1_UUID)
                # Convert the raw string to an integer (0-4095)
                s1_val = int(s1_bytes.decode('utf-8'))

                # Read Sensor 2
                s2_bytes = await client.read_gatt_char(SENSOR_2_UUID)
                # Convert the raw string to an integer (0-4095)
                s2_val = int(s2_bytes.decode('utf-8'))

                sample = {
                    "timestamp": current_time,
                    "sensor_1": s1_val,
                    "sensor_2": s2_val
                }

                # Add the sample to our main data list
                data_list.append(sample)

                # Print to screen so the user sees data flowing
                print(f"Sampled: S1={s1_val}, S2={s2_val}")

                # Wait for the next sample
                await asyncio.sleep(SAMPLE_RATE)

    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nDisconnected.")


if __name__ == "__main__":
    collected_data = []

    try:
        # Pass the list to 'main' so it can be filled.
        asyncio.run(main(collected_data))
        # TODO: do some ML stuff??
    except KeyboardInterrupt:
        # If a keyboard (or IDE stop command) interrupts it, we stop collecting data and save it in finally block
        print("\nProgram stopped by user.")
    finally:

        # TODO: Data processing stuff
        print("\n--- Processing Collected Data ---")
        if not collected_data:
            print("No data was collected.")
        else:
            print(f"Collected {len(collected_data)} samples.")

            # Convert our list of dictionaries into a Pandas DataFrame
            df = pd.DataFrame(collected_data)

            # Convert UNIX timestamp to a readable datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert('America/Los_Angeles')

            # Reorder columns: timestamp, datetime, sensor_1, sensor_2
            df = df[['timestamp', 'datetime', 'sensor_1', 'sensor_2']]

            # Save the ML-ready data to a CSV file
            output_file = "sensor_data.csv"
            df.to_csv(output_file, index=False)

            print(f"\nSuccessfully saved data to {output_file}")
            print("\nDataFrame Head:")
            print(df.head())
