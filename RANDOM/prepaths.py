import RPi.GPIO as GPIO
import time

# Define GPIO pins
LEFT_MOTOR_FORWARD = 17
LEFT_MOTOR_BACKWARD = 18
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 23

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [LEFT_MOTOR_FORWARD, LEFT_MOTOR_BACKWARD, RIGHT_MOTOR_FORWARD, RIGHT_MOTOR_BACKWARD]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Movement functions
def stop():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

def move_forward(duration=1):
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    time.sleep(duration)
    stop()

def move_backward(duration=1):
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.HIGH)
    time.sleep(duration)
    stop()

def turn_left(duration=0.5):
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    time.sleep(duration)
    stop()

def turn_right(duration=0.5):
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.HIGH)
    time.sleep(duration)
    stop()

# Path 1: Right-dominant with one backward step
path_1 = [
    "Forward", "Forward", "Right", "Forward", "Forward",
    "Right", "Backward", "Left", "Forward", "Right",
    "Forward", "Forward", "Right", "Forward", "Forward"
]

# Path 2: Left-dominant with a unique route and backward step
path_2 = [
    "Forward", "Left", "Forward", "Forward", "Left",
    "Forward", "Forward", "Right", "Forward", "Left",
    "Backward", "Left", "Forward", "Forward", "Forward"
]

def move(direction):
    print(f"Moving {direction}")
    if direction == "Forward":
        move_forward()
    elif direction == "Backward":
        move_backward()
    elif direction == "Left":
        turn_left()
    elif direction == "Right":
        turn_right()

def execute_path(path, label):
    print(f"\n--- Executing {label} ---")
    for step in path:
        move(step)
    print(f"--- {label} Complete ---\n")

def main():
    try:
        print("1 - Path 1")
        print("2 - Path 2")
        
        choice = input("Enter 1 or 2: ").strip()
        
        if choice == "1":
            execute_path(path_1, "Path 1")
        elif choice == "2":
            execute_path(path_2, "Path 2")
        else:
            print("Invalid input. Exiting.")
    finally:
        stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()