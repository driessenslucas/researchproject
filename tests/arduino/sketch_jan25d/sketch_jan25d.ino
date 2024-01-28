//
//Wiring method:
//black: motor power "-"
//Red: Motor Power "+"
//Brown: Encode Power "+"
//Green: Encode Power "-"
//Blue: Encode Output A phase
//White: encode phase B of output
//(Note: Encoder input voltage: 3.3-12v)



// Encoder pins
const int encoderPinA = 7; // Encoder output A
const int encoderPinB = 6; // Encoder output B

volatile long encoderCount = 0;
boolean encoderALast;

void setup() {
  Serial.begin(9600); // Start the serial communication

  // Set encoder pins as inputs
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);

  // Initialize the encoder reading
  encoderALast = digitalRead(encoderPinA);

  // Attach an interrupt to encoder Pin A
  attachInterrupt(digitalPinToInterrupt(encoderPinA), readEncoder, CHANGE);
}

void loop() {
  // Print the encoder count every second
  static unsigned long lastPrintTime = 0;
  if (millis() - lastPrintTime >= 1000) {
    Serial.print("Encoder Count: ");
    Serial.println(encoderCount);
    lastPrintTime = millis();
  }
}

void readEncoder() {
  boolean encoderA = digitalRead(encoderPinA);
  boolean encoderB = digitalRead(encoderPinB);

  if (encoderA != encoderALast) { // If the A signal has changed
    // Determine the rotation direction
    if (encoderA != encoderB) {
      encoderCount++;
    } else {
      encoderCount--;
    }
  }
  encoderALast = encoderA;
}
