////const int right =32; // Connect to the digital output (DO) of the sensor 25
////const int left = 26; // Connect to the digital output (DO) of the sensor  26
////
////void setup() {
////  pinMode(right, INPUT);
////  pinMode(left, INPUT);
////  
////  Serial.begin(9600);
////}
////
////void loop() {
////  int leftValue = digitalRead(left); // Read the digital value (interruption rate)
////  int rightValue = digitalRead(left); // Read the digital value (interruption rate)
////
////  // Print the values to the Serial Monitor
////  Serial.print("Digital Value: ");
////  Serial.print(digitalValue);
////
////}
//
////
////const byte encoder0pin = 32; // 25  or 2
////const byte encoder1pin = 26; // 26 or 3
//// 
////int stepsWheel1 =0;
////int stepsWheel2 =0;
////
////
////int lastStateA1 = 0;
////int lastStateB1 = 0;
////int lastStateA2 = 0;
////int lastStateB2 = 0;
////
////void setup() {
////  Serial.begin(9600);
////  pinMode(encoder0pin,INPUT);
////  pinMode(encoder1pin,INPUT);
////}
////
////void loop() {
////  int a1 = digitalRead(encoder0pin);
////  int a2 = digitalRead(encoder1pin);
////
////
////  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
////  // if it changes states 
////  if (a1 != lastStateA1) {
////    if (a1 == 1 ) {
////      stepsWheel1++;
////    }
////    if (a1 == 0) {
////      stepsWheel1--;
////    }
////  }
////  if (a2 != lastStateA2 ) {
////    if (a2 == 1 ) {
////      stepsWheel2++;
////    }
////    if (a2 == 0 ) {
////      stepsWheel2--;
////    }
////  }
////  lastStateA1 = a1;
////  lastStateA2 = a2;
////  Serial.print(stepsWheel1);
////  Serial.print(" - ");
////  Serial.print(stepsWheel2);
////  Serial.println();
////  delay(100);
////
////}
//
//
//int E1 = 2;
//int M1 = 17;
//int E2 = 19;
//int M2 = 4;
//
//const byte encoder0pinA = 32; // 25 or 33
//const byte encoder0pinB = 12; // 26 or 32
//const byte encoder1pinA = 26;
//const byte encoder1pinB = 34; // 26 or 32
//
//int stepsWheel1 =0;
//int stepsWheel2 =0;
//
//int steps;
//
//bool stepsInit = false;
//
//int lastStateA1 = 0;
//int lastStateB1 = 0;
//int lastStateA2 = 0;
//int lastStateB2 = 0;
//
//void setup() {
//  Serial.begin(9600);
//  pinMode(encoder0pinA,INPUT);
//  pinMode(encoder0pinB,INPUT);
//  pinMode(encoder1pinA,INPUT);
//  pinMode(encoder1pinB,INPUT);
//
//  pinMode(M1, OUTPUT);
//  pinMode(M2, OUTPUT);
//  analogWrite(E2, 0); // Set speed
//  digitalWrite(M2, HIGH); // Move forward
//}
//
//void loop() {
//  if (Serial.available() > 0 && !stepsInit) {
//    steps = Serial.parseInt();
//    stepsInit = true;
//  }
//
//  if (stepsInit) {
//    Serial.print("steps wanted: ");
//    Serial.println(steps);
//
//    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
//      stopMotors();
//      stepsInit = false; // Reset the flag after stopping motors
//      stepsWheel1 = 0;
//      stepsWheel2 = 0;
//    } else {
//      moveMotors();
//    }
//    get_steps();
//  }
//}
//
//void get_steps() {
//  int a1 = digitalRead(encoder0pinA);
//  int b1 = digitalRead(encoder0pinB);
//  int a2 = digitalRead(encoder1pinA);
//  int b2 = digitalRead(encoder1pinB);
//
//  // if a1 or b1 changes from 0 to 1, that means the encoder is count it as a step
//  // if it changes states 
//  if (a1 != lastStateA1 || b1 != lastStateB1) {
//    if (a1 == 1 && b1 == 0) {
//      stepsWheel1++;
//    }
//    if (a1 == 0 && b1 == 1) {
//      stepsWheel1--;
//    }
//  }
//  if (a2 != lastStateA2 || b2 != lastStateB2) {
//    if (a2 == 1 && b2 == 0) {
//      stepsWheel2++;
//    }
//    if (a2 == 0 && b2 == 1) {
//      stepsWheel2--;
//    }
//  }
//  lastStateA1 = a1;
//  lastStateB1 = b1;
//  lastStateA2 = a2;
//  lastStateB2 = b2;
//  Serial.print(stepsWheel1);
//  Serial.print(" - ");
//  Serial.print(stepsWheel2);
//  Serial.println();
//}
//
//
//void stopMotors(){
//    analogWrite(E1, 0); // Gradually increase speed
//    analogWrite(E2, 0);
//    digitalWrite(M1, LOW); // Move forward
//    digitalWrite(M2, HIGH);
//}
//void moveMotors(){
//    analogWrite(E1, 255); // Gradually increase speed
//    analogWrite(E2, 255); 
//    digitalWrite(M1, HIGH); // Move forward
//    digitalWrite(M2, LOW);    
//  
//}

int E1 = 2;
int M1 = 17;
int E2 = 19;
int M2 = 4;

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
  Serial.begin(9600);
  pinMode(encoder0pinA,INPUT);
  pinMode(encoder0pinB,INPUT);
  pinMode(encoder1pinA,INPUT);
  pinMode(encoder1pinB,INPUT);

  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);
  analogWrite(E2, 0); // Set speed
  digitalWrite(M2, HIGH); // Move forward
}



void loop() {
  if (Serial.available() > 0 && !stepsInit) {
    steps = Serial.parseInt();
    stepsInit = true;
  }

  if (stepsInit) {
    Serial.print("steps wanted: ");
    Serial.println(steps);

    if (abs(stepsWheel1) >= steps && abs(stepsWheel2) >= steps) {
      stopMotors();
      stepsInit = false; // Reset the flag after stopping motors
      stepsWheel1 = 0;
      stepsWheel2 = 0;
    } else {
      moveMotors();
    }
    get_steps();
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


void stopMotors(){
    analogWrite(E1, 0); // Gradually increase speed
    analogWrite(E2, 0);
    digitalWrite(M1, LOW); // Move forward
    digitalWrite(M2, HIGH);
}
void moveMotors(){
    analogWrite(E1, 255); // Gradually increase speed
    analogWrite(E2, 255); 
    digitalWrite(M1, HIGH); // Move forward
    digitalWrite(M2, LOW);    
  
}
