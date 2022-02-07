"""Docs"""

import serial


class LightControl:
    """Docs"""

    def __init__(self) -> None:
        # define serial parameters
        serial_port = serial.Serial(
            port="COM4",
            baudrate=9600,
            bytesize=8,
            timeout=2,
            stopbits=serial.STOPBITS_ONE,
        )

    def send_data(self) -> None:
        """Send data by the specified serial port."""
        ...

    def automatic_control(self, light_status: int) -> None:
        """Docs"""
        if light_status == 1:
            # ... # reduce error
            while error > 0.3:
                ...  # increase light

        else:
            if light_status == 0:
                ...  # increase light
            else:
                ...  # decrease light

    def get_light_status(self) -> str:
        """Docs"""
        ...

    def get_variable_data(self) -> str:
        """Docs"""
        ...
