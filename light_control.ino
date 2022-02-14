/**************
 *  +---------------+-------------------------+-------------------------+
 *  | Uno           | D2 (NOT CHANGABLE)      | D0-D1, D3-D20           |
 *  +---------------+-------------------------+-------------------------+
 */
#include <RBDdimmer.h>
#include <SoftwareSerial.h>

/*  RBDDimmer cfg  */
#define USE_SERIAL Serial
#define outputPin 3
#define zerocross 2           // for boards with CHANGEBLE input pins
dimmerLamp dimmer(outputPin); // initialase port for dimmer for MEGA, Leonardo, UNO, Arduino M0, Arduino Zero

/*  BT cfg  */
#define RX 9
#define TX 8
SoftwareSerial BTserial(10, 11);

char BTbuff;
int outVal = 0;

void setup()
{
    /*  RBDDimer  */
    USE_SERIAL.begin(9600);
    dimmer.begin(NORMAL_MODE, ON); // dimmer initialisation: name.begin(MODE, STATE)
    USE_SERIAL.println("Dimmer Program is starting...");
    USE_SERIAL.println("Set value");

    /*  BT  */
    BTserial.begin(9600);
    BTserial.println("Started!!");
    pinMode(LED_BUILTIN, OUTPUT);
}

void printSpace(int val)
{
    if ((val / 100) == 0)
        USE_SERIAL.print(" ");
    if ((val / 10) == 0)
        USE_SERIAL.print(" ");
}

void loop()
{
    int preVal = outVal;

    if (BTserial.available())
    {
        USE_SERIAL.println("Incomming data...");
        BTbuff = BTserial.read();
        if (String(BTbuff) == "0")
        {
            outVal = preVal - 5;
        }
        if (String(BTbuff) == "1")
        {
            outVal = preVal + 5;
        }

        if (outVal < 30)
        {
            outVal = 30;
        }
        if (outVal > 85)
        {
            outVal = 85;
        }
    }

    if (USE_SERIAL.available())
    {
        USE_SERIAL.println("Incomming data...");
        BTbuff = USE_SERIAL.read();
        if (String(BTbuff) == "0")
        {
            outVal = preVal - 5;
        }
        if (String(BTbuff) == "1")
        {
            outVal = preVal + 5;
        }

        if (outVal < 30)
        {
            outVal = 30;
        }
        if (outVal > 80)
        {
            outVal = 80;
        }
    }

    dimmer.setPower(outVal); // setPower(0-100%);

    if (preVal != outVal)
    {
        USE_SERIAL.print("lampValue -> ");
        printSpace(dimmer.getPower());
        USE_SERIAL.print(dimmer.getPower());
        USE_SERIAL.println("%");
    }
    delay(50);
}