/*
    MPU6050 Triple Axis Gyroscope & Accelerometer. Simple Gyroscope Example.
    Read more: http://www.jarzebski.pl/arduino/czujniki-i-sensory/3-osiowy-zyroskop-i-akcelerometr-mpu6050.html
    GIT: https://github.com/jarzebski/Arduino-MPU6050
    Web: http://www.jarzebski.pl
    (c) 2014 by Korneliusz Jarzebski
*/

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// Global variables to keep track of the angle and the initial angle
float angleZ = 0;
float initialAngleZ = 0.0;
bool initialAngleSet = false;
unsigned long lastTime = 0;

void setup() 
{
  Serial.begin(9600);

  

  // Initialize MPU6050
  Serial.println("Initialize MPU6050");
  while(!mpu.begin(MPU6050_SCALE_2000DPS, MPU6050_RANGE_2G))
  {
    Serial.println("Could not find a valid MPU6050 sensor, check wiring!");
    delay(500);
  }
  
  // If you want, you can set gyroscope offsets
  // mpu.setGyroOffsetX(155);
  // mpu.setGyroOffsetY(15);
  // mpu.setGyroOffsetZ(15);
  
  // Calibrate gyroscope. The calibration must be at rest.
  // If you don't want calibrate, comment this line.
  mpu.calibrateGyro();

  // Set threshold sensivty. Default 3.
  // If you don't want use threshold, comment this line or set 0.
  mpu.setThreshold(3);
  
  // Check settings
  checkSettings();

//
//  ssd1306_setFixedFont(ssd1306xled_font6x8);
//  ssd1306_128x64_i2c_init();
//  //  ssd1306_128x64_spi_init(22, 5, 21); // Use this line for ESP32 (VSPI)  (gpio22=RST, gpio5=CE for VSPI, gpio21=D/C)
//  ssd1306_clearScreen();
//  ssd1306_printFixed(0,  8, "ESP IP address:", STYLE_NORMAL);
}

void checkSettings()
{
  Serial.println();
  
  Serial.print(" * Sleep Mode:        ");
  Serial.println(mpu.getSleepEnabled() ? "Enabled" : "Disabled");
  
  Serial.print(" * Clock Source:      ");
  switch(mpu.getClockSource())
  {
    case MPU6050_CLOCK_KEEP_RESET:     Serial.println("Stops the clock and keeps the timing generator in reset"); break;
    case MPU6050_CLOCK_EXTERNAL_19MHZ: Serial.println("PLL with external 19.2MHz reference"); break;
    case MPU6050_CLOCK_EXTERNAL_32KHZ: Serial.println("PLL with external 32.768kHz reference"); break;
    case MPU6050_CLOCK_PLL_ZGYRO:      Serial.println("PLL with Z axis gyroscope reference"); break;
    case MPU6050_CLOCK_PLL_YGYRO:      Serial.println("PLL with Y axis gyroscope reference"); break;
    case MPU6050_CLOCK_PLL_XGYRO:      Serial.println("PLL with X axis gyroscope reference"); break;
    case MPU6050_CLOCK_INTERNAL_8MHZ:  Serial.println("Internal 8MHz oscillator"); break;
  }
  
  Serial.print(" * Gyroscope:         ");
  switch(mpu.getScale())
  {
    case MPU6050_SCALE_2000DPS:        Serial.println("2000 dps"); break;
    case MPU6050_SCALE_1000DPS:        Serial.println("1000 dps"); break;
    case MPU6050_SCALE_500DPS:         Serial.println("500 dps"); break;
    case MPU6050_SCALE_250DPS:         Serial.println("250 dps"); break;
  } 
  
  Serial.print(" * Gyroscope offsets: ");
  Serial.print(mpu.getGyroOffsetX());
  Serial.print(" / ");
  Serial.print(mpu.getGyroOffsetY());
  Serial.print(" / ");
  Serial.println(mpu.getGyroOffsetZ());
  
  Serial.println();
}

void loop()
{
    // Read normalized gyro values
    Vector normGyro = mpu.readNormalizeGyro();

    // Current time in milliseconds
    unsigned long currentTime = millis();
    if (lastTime == 0) {
        lastTime = currentTime; // Initialize lastTime
    }

    // Time difference in seconds
    float deltaTime = (currentTime - lastTime) / 1000.0;
    lastTime = currentTime;

    // Integrate the gyroscope data
    angleZ += normGyro.ZAxis * deltaTime;

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

    if (initialAngleSet && (angleDifference == 90)) {
        Serial.println("Rotated approximately 90 degrees from the initial position");
        // Reset the initial angle to start measuring again
        initialAngleSet = false;
    }

    delay(100);
}
