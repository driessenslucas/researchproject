#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;

int mean_ax, mean_ay, mean_az, mean_gx, mean_gy, mean_gz;
int ax_offset, ay_offset, az_offset, gx_offset, gy_offset, gz_offset;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    mpu.initialize();
    Serial.println("Testing device connections...");
    Serial.println(mpu.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");
    delay(1000);
    // Assume sensor is flat and collect initial data
    computeOffsets();
    Serial.println("Offsets computed");
    Serial.print("ax_offset: "); Serial.println(ax_offset);
    Serial.print("ay_offset: "); Serial.println(ay_offset);
    Serial.print("az_offset: "); Serial.println(az_offset);
    Serial.print("gx_offset: "); Serial.println(gx_offset);
    Serial.print("gy_offset: "); Serial.println(gy_offset);
    Serial.print("gz_offset: "); Serial.println(gz_offset);
}

void loop() {
    // Your loop code
}

void computeOffsets() {
    const int num_readings = 1000;
    long ax_total = 0, ay_total = 0, az_total = 0;
    long gx_total = 0, gy_total = 0, gz_total = 0;

    for (int i = 0; i < num_readings; i++) {
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        ax_total += ax;
        ay_total += ay;
        az_total += az;
        gx_total += gx;
        gy_total += gy;
        gz_total += gz;
        delay(10);
    }

    mean_ax = ax_total / num_readings;
    mean_ay = ay_total / num_readings;
    mean_az = az_total / num_readings;
    mean_gx = gx_total / num_readings;
    mean_gy = gy_total / num_readings;
    mean_gz = gz_total / num_readings;

    // Assuming sensor is flat, z should be 16384 (1g), x and y should be 0
    ax_offset = -mean_ax;
    ay_offset = -mean_ay;
    az_offset = -mean_az;

    // Gyro offsets are simply the mean values
    gx_offset = -mean_gx;
    gy_offset = -mean_gy;
    gz_offset = -mean_gz;
}
