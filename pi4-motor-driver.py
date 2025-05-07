import pigpio
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
    FORWARD = "w"
    BACKWARD = "s"
    LEFT = "a"
    RIGHT = "d"
    STOP = " "


# calibrate these values based on how fast you want to go
# range is 0-255
DIRECTION_SPEED = {
    Direction.FORWARD.value: 70,
    Direction.BACKWARD.value: 55,
    Direction.RIGHT.value: 40,
    Direction.LEFT.value: 40,
}

DIRECTION_IN = {
    Direction.FORWARD.value: [1, 0, 0, 1],
    Direction.BACKWARD.value: [0, 1, 1, 0],
    Direction.RIGHT.value: [0, 1, 0, 1],
    Direction.LEFT.value: [1, 0, 1, 0],
    Direction.STOP.value: [0, 0, 0, 0],
}

# controls how fast the wheelchair changes directions
# I wouldn't make it much faster than this unless you
# end up machining the aluminum keys because the sheer
# force will break the plastic keys
TIME_DELAY = 0.020


def drive_motors(dir):
    global curr_direction
    if curr_direction == dir:
        return

    for IN, d in zip([INA1, INB1, INA2, INB2], DIRECTION_IN.get(dir)):
        pi.write(IN, d)

    if dir == Direction.STOP.value:
        for i in range(DIRECTION_SPEED.get(curr_direction), 0, -1):
            pi.set_PWM_dutycycle(PWM1, i)
            pi.set_PWM_dutycycle(PWM2, i)
            time.sleep(TIME_DELAY)
    else:
        curr_direction = dir
        drive_motors(Direction.STOP.value)

        for i in range(DIRECTION_SPEED.get(dir)):
            pi.set_PWM_dutycycle(PWM1, i)
            pi.set_PWM_dutycycle(PWM2, i)
            time.sleep(TIME_DELAY)

    curr_direction = dir


if __name__ == "__main__":
    for pin in [INA1, INB1, INA2, INB2, PWM1, PWM2]:
        pi.set_mode(pin, pigpio.OUTPUT)

    global curr_direction
    curr_direction = None
    try:
        while True:
            val = input("Enter your direction: ")
            if val in Direction._value2member_map_:
                drive_motors(val)
    except KeyboardInterrupt:
        print("Stopping")
        drive_motors(Direction.STOP.value)
        pi.stop()
