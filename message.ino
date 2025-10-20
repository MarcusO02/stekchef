#include <Arduino.h>
#include <Servo.h>

// ===== Pins =====
const int PIN_BODY     = 8;   // base_yaw
const int PIN_SHOULDER = 9;   // shoulder_pitch
const int PIN_ELBOW    = 10;  // elbow_pitch
const int PIN_GRIP     = 6;   // gripper
const int PIN_WRIST    = 11;  // wrist pitch/roll

// ===== Safe limits (µs) =====
const int US_MIN_BODY      =  560, US_MAX_BODY      = 2310;
const int US_MIN_SHOULDER  = 1550, US_MAX_SHOULDER  = 2180;
const int US_MIN_ELBOW     = 1160, US_MAX_ELBOW     = 2110;
const int US_MIN_WRIST     = 100, US_MAX_WRIST     = 2500;  

// Gripper presets (tune!)
const int US_GRIP_OPEN  = 100;
const int US_GRIP_CLOSE = 2500;

const bool TEST_MODE = false;

// ===== Joint structure =====
struct Joint {
  Servo s;
  int us_min, us_max;
  int pos_us;
  int target_us;
};

Joint J_body, J_shoul, J_elbow, J_wrist;
Servo S_grip;

// ===== Utilities =====
static inline int clampi(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }
static inline int isign(int v){ return (v > 0) - (v < 0); }

bool nearTarget(const Joint& J, int eps_us = 8) {
  return abs(J.target_us - J.pos_us) <= eps_us;
}

// READY applies only to the arm CSV command (base+shoulder+elbow)
bool readyPending = false;

// ===== CSV parser (base,shoulder,elbow) =====
bool parse_csv_three_ints(const char* line, int &b, int &s, int &e){
  char tmp[96];
  strncpy(tmp, line, sizeof(tmp)-1);
  tmp[sizeof(tmp)-1] = '\0';

  char *p1 = strtok(tmp, ",");
  char *p2 = strtok(NULL, ",");
  char *p3 = strtok(NULL, ",");

  if (!p1 || !p2 || !p3) return false;

  b = clampi(atoi(p1), US_MIN_BODY,     US_MAX_BODY);
  s = clampi(atoi(p2), US_MIN_SHOULDER, US_MAX_SHOULDER);
  e = clampi(atoi(p3), US_MIN_ELBOW,    US_MAX_ELBOW);
  return true;
}

void setup() {
  Serial.begin(115200);

  // Init joints
  J_body.us_min   = US_MIN_BODY;     J_body.us_max   = US_MAX_BODY;     J_body.pos_us = US_MIN_BODY;     J_body.target_us = J_body.pos_us;
  J_shoul.us_min  = US_MIN_SHOULDER; J_shoul.us_max  = US_MAX_SHOULDER; J_shoul.pos_us = US_MIN_SHOULDER; J_shoul.target_us = J_shoul.pos_us;
  J_elbow.us_min  = US_MIN_ELBOW;    J_elbow.us_max  = US_MAX_ELBOW;    J_elbow.pos_us = US_MIN_ELBOW;    J_elbow.target_us = J_elbow.pos_us;
  J_wrist.us_min  = US_MIN_WRIST;    J_wrist.us_max  = US_MAX_WRIST;    J_wrist.pos_us = (US_MIN_WRIST+US_MAX_WRIST)/2; J_wrist.target_us = J_wrist.pos_us;

  if (!TEST_MODE) {
    J_body.s.attach(PIN_BODY, US_MIN_BODY, US_MAX_BODY);
    J_shoul.s.attach(PIN_SHOULDER, US_MIN_SHOULDER, US_MAX_SHOULDER);
    J_elbow.s.attach(PIN_ELBOW, US_MIN_ELBOW, US_MAX_ELBOW);
    J_wrist.s.attach(PIN_WRIST, US_MIN_WRIST, US_MAX_WRIST);
    S_grip.attach(PIN_GRIP); // gripper usually just uses full range; adjust if needed

    // Write initial holds
    J_body.s.writeMicroseconds(J_body.pos_us);
    J_shoul.s.writeMicroseconds(J_shoul.pos_us);
    J_elbow.s.writeMicroseconds(J_elbow.pos_us);
    J_wrist.s.writeMicroseconds(J_wrist.pos_us);
    S_grip.writeMicroseconds(US_GRIP_OPEN); // power-on open for safety
  }

  Serial.println(F("ROBOT READY. CSV: base,shoulder,elbow  |  GRIP,OPEN/CLOSE  |  WRIST,<us>"));
}

// ===== Incremental motion (non-blocking) =====
void updateJoint(Joint &J, int delta) {
  int diff = J.target_us - J.pos_us;
  if (abs(diff) > delta) {
    J.pos_us += delta * isign(diff);
    J.pos_us = clampi(J.pos_us, J.us_min, J.us_max);
    if (!TEST_MODE) J.s.writeMicroseconds(J.pos_us);
  } else {
    J.pos_us = J.target_us;
    if (!TEST_MODE) J.s.writeMicroseconds(J.pos_us);
  }
}

void loop() {
  // === 1) Serial command handling ===
  static char buf[96];
  static uint8_t len = 0;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[len] = '\0';
      if (len > 0) {

        // 1a) GRIP command
        if (strncmp(buf, "GRIP,", 5) == 0) {
          const char* p = buf + 5;
          if (strstr(p, "OPEN")) {
            if (!TEST_MODE) S_grip.writeMicroseconds(US_GRIP_OPEN);
            Serial.println(F("OK: GRIP OPEN"));
          } else if (strstr(p, "CLOSE")) {
            if (!TEST_MODE) S_grip.writeMicroseconds(US_GRIP_CLOSE);
            Serial.println(F("OK: GRIP CLOSE"));
          } else {
            Serial.println(F("ERR: GRIP arg"));
          }

        // 1b) WRIST command (absolute µs)
        } else if (strncmp(buf, "WRIST,", 6) == 0) {
          int u = atoi(buf + 6);
          u = clampi(u, US_MIN_WRIST, US_MAX_WRIST);
          J_wrist.target_us = u;   // smooth move via updateJoint
          Serial.print(F("OK: WRIST ")); Serial.println(u);

        // 1c) CSV for arm (base,shoulder,elbow)
        } else {
          int ub, us, ue;
          if (parse_csv_three_ints(buf, ub, us, ue)) {
            J_body.target_us  = ub;
            J_shoul.target_us = us;
            J_elbow.target_us = ue;
            readyPending = true; // sender waits for READY
            Serial.print(F("OK: "));
            Serial.print(ub); Serial.print(',');
            Serial.print(us); Serial.print(',');
            Serial.println(ue);
          } else {
            Serial.println(F("ERR: bad CSV"));
          }
        }
      }
      len = 0;
    } else if (len < sizeof(buf)-1) {
      buf[len++] = c;
    }
  }

  // === 2) Smoothly move joints toward targets ===
  const int delta_arm   = 5;  // µs per loop for arm joints
  const int delta_wrist = 20;  // can be slower/faster if you like
  updateJoint(J_body,  delta_arm);
  updateJoint(J_shoul, delta_arm);
  updateJoint(J_elbow, delta_arm);
  updateJoint(J_wrist, delta_wrist);

  // === 3) READY gate for arm motion ===
  if (readyPending &&
      nearTarget(J_body) && nearTarget(J_shoul) && nearTarget(J_elbow))
  {
    Serial.println(F("READY"));
    readyPending = false;
  }

  delay(20);

}
