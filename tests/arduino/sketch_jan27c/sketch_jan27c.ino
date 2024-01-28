#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

MPU6050 mpu;
float angleZ = 0;
float initialAngleZ = 0.0;
bool initialAngleSet = false;
unsigned long lastTime = 0;
void setup() {
    Serial.begin(9600);
    Wire.begin();
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed");
        while (1);
    }

    // Apply the offsets
    mpu.setXAccelOffset(-517);
    mpu.setYAccelOffset(-144);
    mpu.setZAccelOffset(37);
    mpu.setXGyroOffset(0);
    mpu.setYGyroOffset(0);
    mpu.setZGyroOffset(-4);

    Serial.println("Calibration is done.");
}

void loop() {
    // Read raw gyro values
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert gyroscope values to degrees/sec
    float gyroZ = gz / 131.0;

    // Current time in milliseconds
    unsigned long currentTime = millis();
    if (lastTime == 0) {
        lastTime = currentTime; // Initialize lastTime
    }

    // Time difference in seconds
    float deltaTime = (currentTime - lastTime) / 1000.0;
    lastTime = currentTime;

    // Integrate the gyroscope data
    angleZ += gyroZ * deltaTime;

    // Set the initial angle when a certain condition is met (e.g., a button press)
    if (!initialAngleSet) {
        initialAngleZ = angleZ;
        initialAngleSet = true;
        Serial.print("Initial angle set: ");
        Serial.println(initialAngleZ);
    }

    // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
    float angleDifference = angleZ - initialAngleZ;
    Serial.print("Current Angle Difference: ");
    Serial.println(angleDifference);
    angleDifference = abs(angleDifference);

    if (initialAngleSet && (angleDifference > 88 && angleDifference < 93)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
    }

    delay(100);
}
