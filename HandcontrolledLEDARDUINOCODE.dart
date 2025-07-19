const int ledPin = 9;  
int brightness = 0;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    brightness = Serial.parseInt();
    brightness = constrain(brightness, 0, 255);
    analogWrite(ledPin, brightness);
  }
}
