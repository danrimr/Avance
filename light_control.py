import time
import serial


class LightControl:
    """Docs"""

    def __init__(self, port: str = "COM3", bauds: int = 9600) -> None:
        self.serial_port = serial.Serial(
            port=port,
            baudrate=bauds,
            bytesize=8,
            timeout=1,
            stopbits=serial.STOPBITS_ONE,
        )
        self.data: int
        time.sleep(1)

    def send_data(self, data) -> None:
        """Send a data stream throught the serial port."""
        if data < 1.0:
            self.serial_port.write("1".encode("ascii"))
        elif data > 1.3:
            self.serial_port.write("0".encode("ascii"))
        # self.serial_port.write(f"{data}".encode("ascii"))
        # print(data)
