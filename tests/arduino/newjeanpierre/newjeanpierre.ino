#include <ssd1306.h>

#include <WiFi.h>
#include <ArduinoOTA.h>
// or for ESP32: #include <WiFi.h>

const char* ssid = "telenet-799DCED";
const char* password = "m7cnypsHjxhp";

WiFiServer server(80);

// Variable to store the HTTP request
String header;

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

int turnDuration = 300;


const byte encoder0pinA = 32; // 25 or 33
const byte encoder0pinB = 12; // 26 or 32
const byte encoder1pinA = 26;
const byte encoder1pinB = 34; // 26 or 32

int stepsWheel1 =0;
int stepsWheel2 =0;

int steps;

bool stepsInit = false;

int lastStateA1 = 0;
int lastStateB1 = 0;
int lastStateA2 = 0;
int lastStateB2 = 0;


void setup() {
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);
  analogWrite(E2, 0); // Set speed
  digitalWrite(M2, HIGH); // Move forward

  Serial.begin(9600);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  ssd1306_setFixedFont(ssd1306xled_font6x8);
  ssd1306_128x64_i2c_init();
  //  ssd1306_128x64_spi_init(22, 5, 21); // Use this line for ESP32 (VSPI)  (gpio22=RST, gpio5=CE for VSPI, gpio21=D/C)
  ssd1306_clearScreen();
  ssd1306_printFixed(0,  8, "ESP IP address:", STYLE_NORMAL);
  String ip = WiFi.localIP().toString();
  ssd1306_printFixed(0, 16, ip.c_str(), STYLE_BOLD);
  ssd1306_printFixed(0, 24, "Current action:", STYLE_NORMAL);

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();

    ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH)
      type = "sketch";
    else // U_SPIFFS
      type = "filesystem";

    // NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
    Serial.println("Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nEnd");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });
  ArduinoOTA.begin();

  pinMode(encoder0pinA,INPUT);
  pinMode(encoder0pinB,INPUT);
  pinMode(encoder1pinA,INPUT);
  pinMode(encoder1pinB,INPUT);
}

void loop() {

  ArduinoOTA.handle();
  WiFiClient client = server.available(); // Listen for incoming clients

  if (client) {
    String currentLine = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (currentLine.length() == 0) {
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
          } else {
            currentLine = "";
          }
        } else if (c != '\r') {
          currentLine += c;
        }
        

        // Check the request route
        if (currentLine.endsWith("GET /forward")) {
          Serial.println("moving forward");
//          ssd1306_printFixed(0, 32, "moving forward", STYLE_NORMAL);
          stepsInit = true;
          move_forward();
          
        } else if (currentLine.endsWith("GET /left")) {
          Serial.println("moving left");
//          ssd1306_printFixed(0, 32, "moving left", STYLE_NORMAL);
          stepsInit = true;
          move_left();
          
        } else if (currentLine.endsWith("GET /right")) {
          Serial.println("moving right");
//          ssd1306_printFixed(0, 32, "moving right", STYLE_NORMAL);
          stepsInit = true;
          move_right();
        }
        else{
          stop_moving(); 
        }
      }
    }
    client.stop();
  }
}

void get_steps() {
  int a1 = digitalRead(encoder0pinA);
  int b1 = digitalRead(encoder0pinB);
  int a2 = digitalRead(encoder1pinA);
  int b2 = digitalRead(encoder1pinB);

  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
  // if it changes states 
  if (a1 != lastStateA1 || b1 != lastStateB1) {
    if (a1 == 1 && b1 == 0) {
      stepsWheel1--;
    }
    if (a1 == 0 && b1 == 1) {
      stepsWheel1--;
    }
  }
  if (a2 != lastStateA2 || b2 != lastStateB2) {
    if (a2 == 1 && b2 == 0) {
      stepsWheel2--;
    }
    if (a2 == 0 && b2 == 1) {
      stepsWheel2--;
    }
  }
  lastStateA1 = a1;
  lastStateB1 = b1;
  lastStateA2 = a2;
  lastStateB2 = b2;
  Serial.print(stepsWheel1);
  Serial.print(" - ");
  Serial.print(stepsWheel2);
  Serial.println();
}

//void move_forward() {
//  steps = 2; // Set the desired number of steps
//  stepsInit = true; // Enable step counting
//
//  while (stepsInit) {
//    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
//      stop_moving();
//      stepsInit = false; // Reset the flag after stopping motors
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//      break; // Exit the loop once the desired steps are reached
//    } else {
//      analogWrite(E1, 255); // Set speed
//      analogWrite(E2, 255); // Set speed
//      digitalWrite(M1, HIGH); // Move forward
//      digitalWrite(M2, LOW); // Move forward
//    }
//    get_steps();
//  }
//
//  // Additional stop command to ensure motors are stopped
//  stop_moving();
//  stepsInit = false; // Reset the flag after stopping motors
//  stepsWheel1 = 0;
//  stepsWheel2 = 0;
//  delay(10);
//}

void move_forward() {
  steps = 2; // Set the desired number of steps
  stepsInit = true; // Enable step counting

  while (stepsInit) {
    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
      stop_moving();
      stepsInit = false; // Reset the flag after stopping motors
      stepsWheel1 = 0;
      stepsWheel2 = 0;
      break; // Exit the loop once the desired steps are reached
    } else {
      analogWrite(E1, 255); // Set speed
      analogWrite(E2, 255); // Set speed
      digitalWrite(M1, HIGH); // Move forward
      digitalWrite(M2, LOW); // Move forward
    }
  get_steps();
  }

  // Additional stop command to ensure motors are stopped
  stop_moving();
  stepsInit = false; // Reset the flag after stopping motors
  stepsWheel1 = 0;
  stepsWheel2 = 0;
  delay(10);
}


//void move_left() {
//  steps = 1; // Adjust the step count as needed
//  stepsInit = true;
//
//  while (stepsInit) {
//    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
//      stop_moving();
//      stepsInit = false;
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//      break;
//    } else {
//      analogWrite(E1, 200); // Set speed for left motor
//      analogWrite(E2, 200); // Set speed for right motor
//      digitalWrite(M1, HIGH); // Move left motor backward
//      digitalWrite(M2, HIGH); // Move right motor forward
//    }
//    get_steps();
//  }
//
//  stop_moving();
//  stepsInit = false;
//  stepsWheel1 = 0;
//  stepsWheel2 = 0;
//}
//
//void move_right() {
//  steps = 1; // Adjust the step count as needed
//  stepsInit = true;
//
//  while (stepsInit) {
//    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
//      stop_moving();
//      stepsInit = false;
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//      break;
//    } else {
//      analogWrite(E1, 200); // Set speed for left motor
//      analogWrite(E2, 200); // Set speed for right motor
//      digitalWrite(M1, LOW); // Move left motor forward
//      digitalWrite(M2, LOW); // Move right motor backward
//    }
//    get_steps();
//  }
//
//  stop_moving();
//  stepsInit = false;
//  stepsWheel1 = 0;
//  stepsWheel2 = 0;
//}

void move_left() {
  // Left turn code for 90-degree stationary turn
  analogWrite(E1, 200); // Set speed for motor 1
  analogWrite(E2, 200); // Set speed for motor 2
  digitalWrite(M1, HIGH); // Run motor 1 forward
  digitalWrite(M2, HIGH); // Run motor 2 backward

  delay(turnDuration); // Wait for the duration of the turn

  // Stop motors after turning
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}

void move_right() {
  // Right turn code for 90-degree stationary turn
  analogWrite(E1, 255); // Set speed for motor 1
  analogWrite(E2, 255); // Set speed for motor 2
  digitalWrite(M1, LOW); // Run motor 1 backward
  digitalWrite(M2, LOW); // Run motor 2 forward

  delay(turnDuration); // Wait for the duration of the turn

  // Stop motors after turning
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}

void stop_moving() {
  analogWrite(E1, 0); // Stop one motor for turning
  analogWrite(E2, 0); // Set speed
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}
