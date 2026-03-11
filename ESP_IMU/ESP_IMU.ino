#include <Wire.h>
#include <BluetoothSerial.h>
#include "MPU6050.h"

BluetoothSerial SerialBT;
MPU6050 mpu;

// ---------------- Configuration ----------------
constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000 / SAMPLE_RATE_HZ;

// ------------------------------------------------

uint64_t last_sample_time = 0;

void setup() {
  Serial.begin(115200);          // For debugging only
  Wire.begin();

  // --- MPU6050 init ---
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while (1);
  }

  // Optional: configure ranges
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);

  // --- Bluetooth SPP ---
  if (!SerialBT.begin("ESP32_IMU_WRIST")) {
    Serial.println("Bluetooth failed");
    while (1);
  }

  Serial.println("ESP32 IMU ready. Waiting for BT connection...");
}

void loop() {
  uint64_t now = esp_timer_get_time();

  if (now - last_sample_time < SAMPLE_PERIOD_US) return;
  last_sample_time = now;

  // -------- Read MPU6050 --------
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert to physical units
  float acc_x = ax / 8192.0f;    // ±4g
  float acc_y = ay / 8192.0f;
  float acc_z = az / 8192.0f;

  float gyro_x = gx / 65.5f;     // ±500 dps
  float gyro_y = gy / 65.5f;
  float gyro_z = gz / 65.5f;

  // ------------------------------------------------
  // Packet format:
  // $IMU <t_us> ax ay az gx gy gz
  // ------------------------------------------------
  SerialBT.printf(
    "$IMU %llu %.5f %.5f %.5f %.5f %.5f %.5f\n",
    now,
    acc_x, acc_y, acc_z,
    gyro_x, gyro_y, gyro_z
  );
  Serial.printf(
    "$IMU %llu %.5f %.5f %.5f %.5f %.5f %.5f\n",
    now,
    acc_x, acc_y, acc_z,
    gyro_x, gyro_y, gyro_z
  );
}