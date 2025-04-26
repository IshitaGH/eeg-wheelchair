import pigpio
import keyboard
import time
from enum import Enum

pi = pigpio.pi()

INA1 = 17
INB1 = 27
PWM1 = 18

INA2 = 22
INB2 = 23
PWM2 = 13

class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5

# TODO: calibrate these values for the wheelchair
class Speed(Enum):
    FORWARD_SPEED = 80
    BACKWARD_SPEED = 80
    RIGHT_SPEED = 80
    LEFT_SPEED = 80


def drive_motor(direction):
    if direction == Direction.FORWARD.value:
        pi.write(INA1, 1)
        pi.write(INB1, 0)
        pi.write(INA2, 1)
        pi.write(INB2, 0)
        pi.set_PWM_dutycycle(PWM1, abs(Speed.FORWARD_SPEED.value))
        pi.set_PWM_dutycycle(PWM2, abs(Speed.FORWARD_SPEED.value))
    elif direction == Direction.BACKWARD.value:
        pi.write(INA1, 0)
        pi.write(INB1, 1)
        pi.write(INA2, 0)
        pi.write(INB2, 1)
        pi.set_PWM_dutycycle(PWM1, abs(Speed.BACKWARD_SPEED.value))
        pi.set_PWM_dutycycle(PWM2, abs(Speed.BACKWARD_SPEED.value))
    elif direction == Direction.RIGHT.value:
        pi.write(INA1, 1)
        pi.write(INB1, 0)
        pi.write(INA2, 0)
        pi.write(INB2, 1)
        pi.set_PWM_dutycycle(PWM1, abs(Speed.RIGHT_SPEED.value))
        pi.set_PWM_dutycycle(PWM2, abs(Speed.RIGHT_SPEED.value))
    elif direction == Direction.LEFT.value:
        pi.write(INA1, 0)
        pi.write(INB1, 1)
        pi.write(INA2, 1)
        pi.write(INB2, 0)
        pi.set_PWM_dutycycle(PWM1, abs(Speed.LEFT_SPEED.value))
        pi.set_PWM_dutycycle(PWM2, abs(Speed.LEFT_SPEED.value))
    elif direction == Direction.STOP.value:
        pi.write(INA1, 0)
        pi.write(INB1, 0)
        pi.write(INA2, 0)
        pi.write(INB2, 0)
        pi.set_PWM_dutycycle(PWM1, 0)
        pi.set_PWM_dutycycle(PWM2, 0)
    

if __name__ == '__main__':
    for pin in [INA1, INB1, INA2, INB2, PWM1, PWM2]:
        pi.set_mode(pin, pigpio.OUTPUT)

    try:
        while True:
            val = input("Enter your value: ")
            if val == 'w':
                drive_motor(Direction.FORWARD.value)
            elif val == 's':
                drive_motor(Direction.BACKWARD.value)
            elif val == 'a':
                drive_motor(Direction.LEFT.value)
            elif val == 'd':
                drive_motor(Direction.RIGHT.value)
            elif keyboard.is_pressed('space'):
                drive_motor(Direction.STOP.value)
    except KeyboardInterrupt:
        print("Stopping")
        drive_motor(Direction.STOP.value)
        pi.stop()