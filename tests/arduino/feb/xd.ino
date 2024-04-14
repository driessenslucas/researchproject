#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"
#include <ssd1306.h>
#include <WiFi.h>
#include <ArduinoOTA.h>

// WiFi credentials and server configuration
const char *ssid = "telenet-799DCED"; // Enter your WiFi SSID
const char *password = "";            // Enter your WiFi password
WiFiServer server(80);

// Pin definitions
const int sensorTrigPins[] = {27, 33, 25};
const int sensorEchoPins[] = {26, 32, 35};
const int motorPins[] = {17, 4};       // m1, m2
const int motorEnablePins[] = {2, 19}; // E1, E2

// Constants
constexpr float SOUND_SPEED_CM_US = 0.034;
constexpr int TURN_DURATION = 245;
constexpr int MAX_DISTANCE_CM = 100;
volatile bool shouldStop = false;
bool isTurning = false;

int initialSpeed = 125; // Set a higher initial speed
int minSpeed = 40;      // Set a minimum speed
int speed = initialSpeed;

// MPU6050
MPU6050 mpu;
// Global variables to keep track of the angle and the initial angle
float angleZ = 0;
float initialAngleZ = 0.0;
bool initialAngleSet = false;
unsigned long lastTime = 0;

// Function declarations
void setupWifiAndOTA();
void setupSensorsAndDisplay();
void handleClientRequests(WiFiClient &client);
String readSensor(int trigPin, int echoPin);
void move_forward();
void move_left();
void move_right();
void stop_moving();
void setupMPU();
void setupMotors();

void calibrateSensors()
{
    long gyroZAccum = 0;
    Serial.println("Calibrating...");
    for (int i = 0; i < 100; i++)
    {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        gyroZAccum += gz;
        delay(20);
    }
    mpu.setZGyroOffset(-gyroZAccum / 13100); // Adjust based on 100 readings
    Serial.println("Calibration Complete");
}

void setup()
{
    Serial.begin(9600);
    setupWifiAndOTA();
    Wire.begin();
    setupSensorsAndDisplay();
    setupMPU();
    setupMotors();
}

void loop()
{
    ArduinoOTA.handle();
    WiFiClient client = server.available();
    if (client)
    {
        handleClientRequests(client);
        client.stop();
    }
}

void setupWifiAndOTA()
{
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected. IP address: " + WiFi.localIP().toString());

    // OTA setup omitted for brevity
    ArduinoOTA.onStart([]()
                       {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH)
      type = "sketch";
    else  // U_SPIFFS
      type = "filesystem";

    // NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
    Serial.println("Start updating " + type); });
    ArduinoOTA.onEnd([]()
                     { Serial.println("\nEnd"); });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total)
                          { Serial.printf("Progress: %u%%\r", (progress / (total / 100))); });
    ArduinoOTA.onError([](ota_error_t error)
                       {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed"); });
    ArduinoOTA.begin();
    server.begin();
}

void setupSensorsAndDisplay()
{
    for (int i = 0; i < sizeof(sensorTrigPins) / sizeof(sensorTrigPins[0]); ++i)
    {
        pinMode(sensorTrigPins[i], OUTPUT);
        pinMode(sensorEchoPins[i], INPUT);
    }

    // SSD1306 display initialization omitted for brevity
    ssd1306_setFixedFont(ssd1306xled_font6x8);
    ssd1306_128x64_i2c_init();
    ssd1306_clearScreen();
    ssd1306_printFixed(0, 8, "ESP IP address:", STYLE_NORMAL);
    String ip = WiFi.localIP().toString();
    ssd1306_printFixed(0, 16, ip.c_str(), STYLE_BOLD);
}

void setupMPU()
{
    mpu.initialize();
    if (!mpu.testConnection())
    {
        Serial.println("MPU6050 connection failed");
        while (1)
            ;
    }
    // mpu.setZGyroOffset(-18);
    calibrateSensors();
}

void setupMotors()
{
    for (int pin : motorEnablePins)
    {
        pinMode(pin, OUTPUT);
        analogWrite(pin, 0);
    }
    for (int pin : motorPins)
    {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, HIGH);
    }
}

void handleClientRequests(WiFiClient &client)
{
    String currentLine = "";
    while (client.connected())
    {
        if (client.available())
        {
            char c = client.read();
            if (c == '\n')
            {
                if (currentLine.length() == 0)
                {
                    // Send a standard HTTP response header
                    client.println("HTTP/1.1 200 OK");
                    client.println("Content-type:text/html");
                    client.println("Connection: close");
                    client.println();
                    // Send the HTML content
                    client.println("<!DOCTYPE html>");
                    client.println("<html>");
                    client.println("<body>");
                    client.println("<h2>ESP32 Motor Control</h2>");
                    client.println("<p><a href=\"/forward\"><button class=\"button\">Forward</button></a></p>");
                    client.println("<p><a href=\"/left\"><button class=\"button\">Left</button></a></p>");
                    client.println("<p><a href=\"/right\"><button class=\"button\">Right</button></a></p>");
                    client.println("<p><a href=\"/stop\"><button class=\"button button2\">Stop</button></a></p>");
                    client.println("</body>");
                    client.println("</html>");

                    break;
                }
                else
                {
                    currentLine = "";
                }
            }
            else if (c != '\r')
            {
                currentLine += c;
            }

            // Check the request route
            if (currentLine.endsWith("GET /forward"))
            {
                Serial.println("moving forward");
                move_forward();
            }
            else if (currentLine.endsWith("GET /left"))
            {
                Serial.println("moving left");
                move_left();
            }
            else if (currentLine.endsWith("GET /right"))
            {
                Serial.println("moving right");
                move_right();
            }
            else if (currentLine.endsWith("GET /stop"))
            {
                stop_moving();
            }
            else if (currentLine.endsWith("GET /sensors"))
            {

                client.println("HTTP/1.1 200 OK");
                client.println("Content-type:text/plain");
                client.println("Connection: close");
                client.println();
                client.println(createJsonResponse());
                client.println("got sensors");
            }
        }
    }
}

String createJsonResponse()
{
    // Initialize a JSON string
    String jsonResponse = "{";

    // Append sensor values to the JSON string
    jsonResponse += "\"right\": " + readSensor(sensorTrigPins[0], sensorEchoPins[0]) + ",";
    jsonResponse += "\"left\": " + readSensor(sensorTrigPins[1], sensorEchoPins[1]) + ",";
    jsonResponse += "\"front\": " + readSensor(sensorTrigPins[2], sensorEchoPins[2]);

    // Close the JSON string
    jsonResponse += "}";

    return jsonResponse;
}

String readSensor(int trigPin, int echoPin)
{
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    float distance = pulseIn(echoPin, HIGH) * SOUND_SPEED_CM_US / 2.0;
    distance = min(distance, MAX_DISTANCE_CM);
    return String(distance);
}

void move_forward()
{
    if (shouldStop)
    {                       // Check if should stop before starting
        shouldStop = false; // Reset the flag
        return;
    }

    // Forward movement code
    analogWrite(motorEnablePins[0], 255); // Set speed
    analogWrite(motorEnablePins[1], 255); // Set speed
    digitalWrite(motorPins[0], LOW);      // Move forward
    digitalWrite(motorPins[1], HIGH);     // Move forward

    for (int i = 0; i < 7; ++i)
    {
        delay(100); // Break long delay into smaller parts
        if (shouldStop)
        {
            break; // Exit early if stop requested
        }
    }

    if (shouldStop)
    {                       // Additional check if stop was requested during delay
        shouldStop = false; // Reset the flag
        // Stop motors after turning with offset in mind
        analogWrite(motorEnablePins[1], 0);
        delay(15);
        analogWrite(motorEnablePins[0], 0);

        digitalWrite(motorPins[0], LOW);
        digitalWrite(motorPins[1], HIGH);
        return;
    }

    // Stop motors after turning with offset in mind
    analogWrite(motorEnablePins[1], 0);
    delay(15);
    analogWrite(motorEnablePins[0], 0);

    digitalWrite(motorPins[0], LOW);
    digitalWrite(motorPins[1], HIGH);
}

void move_left()
{
    calibrateSensors();
    isTurning = true;
    int speedIncrement = 0; // Set initial speed increment

    // Right turn code for 90-degree stationary turn
    while (isTurning && !shouldStop)
    {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        float gyroZ = gz / 131.0;
        unsigned long currentTime = millis();
        if (lastTime == 0)
            lastTime = currentTime; // Initialize lastTime

        float deltaTime = (currentTime - lastTime) / 1000.0;
        lastTime = currentTime;
        angleZ += gyroZ * deltaTime;

        // Check shouldStop flag periodically
        if (shouldStop)
            break;

        // Set the initial angle when a certain condition is met (e.g., a button press)
        if (!initialAngleSet)
        {
            initialAngleZ = angleZ;
            initialAngleSet = true;
            Serial.print("Initial angle set: ");
            Serial.println(initialAngleZ);
        }

        // Calculate the angle difference
        float angleDifference = abs(angleZ - initialAngleZ);

        // Dynamically adjust speed based on angle difference
        speed = initialSpeed - (int)((angleDifference / 90) * (initialSpeed - minSpeed));
        speed = max(speed, minSpeed); // Ensure speed doesn't fall below minimum

        if (angleDifference >= 80 && angleDifference < 88)
        {
            speed += 5 + speedIncrement; // Increase the speed by the base amount plus any increment
            speedIncrement += 5;         // Increase the speed increment for the next iteration
            speed = min(speed, 200);     // Ensure speed doesn't exceed initial speed
        }
        else
        {
            speedIncrement = 0; // Reset speedIncrement if not within the last 4 degrees
        }

        // Set the motor speeds
        analogWrite(motorEnablePins[0], speed); // Set speed for motor 1
        analogWrite(motorEnablePins[1], speed); // Set speed for motor 2
        digitalWrite(motorPins[0], HIGH);       // Run motor 1 backward
        digitalWrite(motorPins[1], HIGH);       // Run motor 2 forward

        Serial.print("Current Angle Difference: ");
        Serial.println(angleDifference);

        // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
        if (initialAngleSet && angleDifference >= 88)
        {
            Serial.println("Rotated approximately 90 degrees from the initial position");
            // Reset the initial angle to start measuring again
            initialAngleSet = false;
            stop_moving();
            isTurning = false;
        }
        delay(100);
    }
    if (shouldStop)
    {                       // If stop was requested, immediately stop the motors and reset the flag
        shouldStop = false; // Reset the flag
        stop_moving();      // Call the stop function to halt the motors
    }
}

void move_right()
{
    calibrateSensors();
    isTurning = true;
    // int initialSpeed = 100; // Set a higher initial speed
    // int minSpeed = 60;      // Set a minimum speed
    // int speed = initialSpeed;
    int speedIncrement = 0; // Set initial speed increment

    // Right turn code for 90-degree stationary turn
    while (isTurning && !shouldStop)
    {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        float gyroZ = gz / 131.0;
        unsigned long currentTime = millis();
        if (lastTime == 0)
            lastTime = currentTime; // Initialize lastTime

        float deltaTime = (currentTime - lastTime) / 1000.0;
        lastTime = currentTime;
        angleZ += gyroZ * deltaTime;

        // Check shouldStop flag periodically
        if (shouldStop)
            break;

        // Set the initial angle when a certain condition is met (e.g., a button press)
        if (!initialAngleSet)
        {
            initialAngleZ = angleZ;
            initialAngleSet = true;
            Serial.print("Initial angle set: ");
            Serial.println(initialAngleZ);
        }

        // Calculate the angle difference
        float angleDifference = abs(angleZ - initialAngleZ);

        // Dynamically adjust speed based on angle difference
        speed = initialSpeed - (int)((angleDifference / 90) * (initialSpeed - minSpeed));
        speed = max(speed, minSpeed); // Ensure speed doesn't fall below minimum

        // Increase speed slightly if within 4 degrees of target angle
        if (angleDifference >= 80 && angleDifference < 87)
        {
            speed += 5 + speedIncrement; // Increase the speed by the base amount plus any increment
            speedIncrement += 5;         // Increase the speed increment for the next iteration
            speed = min(speed, 200);     // Ensure speed doesn't exceed initial speed
        }
        else
        {
            speedIncrement = 0; // Reset speedIncrement if not within the last 4 degrees
        }

        // Set the motor speeds
        analogWrite(motorEnablePins[0], speed); // Set speed for motor 1
        analogWrite(motorEnablePins[1], speed); // Set speed for motor 2
        digitalWrite(motorPins[0], LOW);        // Run motor 1 backward
        digitalWrite(motorPins[1], LOW);        // Run motor 2 forward

        Serial.print("Current Angle Difference: ");
        Serial.println(angleDifference);

        // Check if the rotation around Z-axis is close to 90 degrees from the initial angle
        if (initialAngleSet && angleDifference >= 88)
        {
            Serial.println("Rotated approximately 90 degrees from the initial position");
            // Reset the initial angle to start measuring again
            initialAngleSet = false;
            stop_moving();
            isTurning = false;
        }
        delay(100);
    }
    if (shouldStop)
    {                       // If stop was requested, immediately stop the motors and reset the flag
        shouldStop = false; // Reset the flag
        stop_moving();      // Call the stop function to halt the motors
    }
}

void stop_moving()
{
    shouldStop = true; // Set the stop flag
    isTurning = false;
    analogWrite(motorEnablePins[0], 0); // Stop one motor for turning
    analogWrite(motorEnablePins[1], 0); // Set speed
    digitalWrite(motorPins[0], LOW);
    digitalWrite(motorPins[1], HIGH);
    shouldStop = false;
}