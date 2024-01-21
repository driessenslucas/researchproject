#include <ssd1306.h>

#include <WiFi.h>
#include <ArduinoOTA.h>
// or for ESP32: #include <WiFi.h>

const char* ssid = "telenet-799DCED";
const char* password = "";

WiFiServer server(80);

// Variable to store the HTTP request
String header;

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

int turnDuration = 300;

int carX = 1;
int carY = 1;

int path[][2] = {
  {1, 1},
  {2, 1},
  {3, 1},
  {4, 1},
  {5, 1},
  {6, 1},
  {7, 1},
  {8, 1},
  {8, 1},
  {8, 2},
  {8, 3},
  {8, 3},
  {9, 3},
  {10, 3},
  {10, 3},
  {10, 4},
  {10, 5},
  {10, 5},
  {9, 5},
  {8, 5},
  {7, 5},
  {6, 5},
  {6, 5},
  {6, 6},
  {6, 7},
  {6, 8},
  {6, 8},
  {7, 8},
  {8, 8},
  {9, 8},
  {10, 8},
  {10, 8},
  {10, 9},
  {10, 10}
};

// Variable to keep track of the current step in the path
int currentStep = 0;

// Define the layout of the maze
int layout[12][12] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1},
    {1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
    {1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1},
    {1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};

//}
void drawMaze() {
    ssd1306_clearScreen();
    ssd1306_printFixed(0,  8, "ESP IP:", STYLE_NORMAL);
    String ip = WiFi.localIP().toString();
    ssd1306_printFixed(0, 16, ip.c_str(), STYLE_BOLD);
    int viewSize = 8; // Define the size of the zoomed-in view

    // Calculate the top-left corner of the view
    int startX = max(0, min(carX - viewSize / 2, 12 - viewSize));
    int startY = max(0, min(carY - viewSize / 2, 12 - viewSize));

    // Calculate the starting position of the maze on the screen
    int mazeStartX = max(0, 128 - (viewSize * 6)); // Adjusted to the right side

    for (int y = startY; y < startY + viewSize; y++) {
        for (int x = startX; x < startX + viewSize; x++) {
            int screenX = mazeStartX + (x - startX) * 6; // Adjusted for maze position
            int screenY = (y - startY) * 8;

            if (layout[y][x] == 1) {
                // Draw wall as '#'
                ssd1306_printFixed(screenX, screenY, "#", STYLE_NORMAL);
            } else {
                // Draw empty space
                ssd1306_printFixed(screenX, screenY, " ", STYLE_NORMAL);
            }
        }
    }

    // Draw the car at its current position
    int carScreenX = mazeStartX + (carX - startX) * 6; // Adjusted for maze position
    int carScreenY = (carY - startY) * 8;
    ssd1306_printFixed(carScreenX, carScreenY, "o", STYLE_NORMAL);
}
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
  drawMaze();
  currentStep = 0; // Start from the beginning of the path
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
          move_forward();
          currentStep++;
          carX = path[currentStep][0];
          carY = path[currentStep][1];
          drawMaze();
          
        } else if (currentLine.endsWith("GET /left")) {
          Serial.println("moving left");
//          ssd1306_printFixed(0, 32, "moving left", STYLE_NORMAL);
          move_left();
          currentStep++;
          carX = path[currentStep][0];
          carY = path[currentStep][1];
          drawMaze();
          
        } else if (currentLine.endsWith("GET /right")) {
          Serial.println("moving right");
//          ssd1306_printFixed(0, 32, "moving right", STYLE_NORMAL);
          move_right();
          currentStep++;
          carX = path[currentStep][0];
          carY = path[currentStep][1];
          drawMaze();
          
        }
        else{
          stop_moving(); 
        }
      }
    }
    client.stop();
  }
}

void move_forward() {
  // Forward movement code
  analogWrite(E1, 255); // Set speed
  analogWrite(E2, 255); // Set speed
  digitalWrite(M1, HIGH); // Move forward
  digitalWrite(M2, LOW); // Move forward

  delay(1000);

  // Stop motors after turning
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
}

void move_left() {
  // Left turn code for 90-degree stationary turn
  analogWrite(E1, 255); // Set speed for motor 1
  analogWrite(E2, 255); // Set speed for motor 2
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