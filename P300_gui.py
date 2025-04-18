import pygame
import time
import os
import csv
import random
from datetime import datetime

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("EEG Data Collection - P300")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)  # Not-too-bright red for direction text

# Directions
DIRECTIONS = ["Forward", "Backward", "Left", "Right"]

# Experiment Parameters
TRIAL_DURATION = 10
BASELINE_DURATION = 10
PHASE_ONE_TRIALS_PER_DIRECTION = 5
PHASE_TWO_TOTAL_TRIALS = 20

# Image loading
IMAGE_PATH = "images"
ARROW_PATH = "images_arrows"
direction_images = {}
arrow_images = {}
for direction in DIRECTIONS:
    dir_img_path = os.path.join(IMAGE_PATH, f"{direction.lower()}.png")
    arrow_img_path = os.path.join(ARROW_PATH, f"{direction.lower()}_arrow.png")
    direction_images[direction] = pygame.image.load(dir_img_path)
    arrow_images[direction] = pygame.image.load(arrow_img_path)

# Timestamps and logging
stopwatch_start = None
timestamp_log = []

# Output folder
date_str = datetime.now().strftime("%Y-%m-%d")
folder_name = f"Data Collection - P300 {date_str}"
os.makedirs(folder_name, exist_ok=True)

def log_event(event):
    global stopwatch_start
    now = time.time()
    if stopwatch_start is None:
        stopwatch_start = now
    elapsed = now - stopwatch_start
    timestamp_log.append([now, f"{elapsed:.3f}", event])

def save_log(filename):
    with open(os.path.join(folder_name, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Real Time", "Elapsed Time", "Event"])
        writer.writerows(timestamp_log)

def wait_for_space(text):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 48)
    rendered = font.render(text, True, WHITE)
    rect = rendered.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(rendered, rect)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

def countdown():
    font = pygame.font.Font(None, 72)
    for i in range(5, 0, -1):
        screen.fill(BLACK)
        rendered = font.render(str(i), True, WHITE)
        rect = rendered.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(rendered, rect)
        pygame.display.flip()
        time.sleep(1)

def flash_sequence(direction):
    img_w, img_h = 150, 150
    positions = {
        "Forward":  (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT // 6 - img_h // 2),
        "Backward": (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT * 5 // 6 - img_h // 2),
        "Left":     (SCREEN_WIDTH // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2),
        "Right":    (SCREEN_WIDTH * 5 // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2)
    }

    flash_order = [("Forward", "Left"), ("Top", "Right"), ("Left", "Right"), ("Forward", "Backward")]
    flash_order = [random.sample(DIRECTIONS, 2) for _ in range(10)]

    log_event(f"Start Flashing - {direction}")
    for combo in flash_order:
        screen.fill(BLACK)
        for dir in DIRECTIONS:
            img = direction_images[dir]
            if dir in combo:
                img = arrow_images[dir]  # flash arrow
            screen.blit(pygame.transform.scale(img, (img_w, img_h)), positions[dir])
        font = pygame.font.Font(None, 36)
        text = font.render(direction, True, RED)
        screen.blit(text, (SCREEN_WIDTH - 200, 20))
        pygame.display.flip()
        log_event(f"Flash {combo}")
        time.sleep(0.3)  # flash duration

    log_event(f"End Flashing - {direction}")
    time.sleep(BASELINE_DURATION)  # baseline black screen
    log_event(f"Baseline - {direction}")

    screen.fill(BLACK)
    pygame.display.flip()

def phase(direction_order, phase_num):
    for i, direction in enumerate(direction_order):
        global stopwatch_start, timestamp_log
        stopwatch_start = None
        timestamp_log = []
        log_event(f"Start Trial {i+1} - {direction}")
        flash_sequence(direction)
        log_event(f"End Trial {i+1} - {direction}")
        save_log(f"phase{phase_num}_trial{i+1}_{direction}.csv")

def main():
    wait_for_space("Data Collection (1/2) - Press SPACE to begin")
    countdown()
    phase([d for d in DIRECTIONS for _ in range(PHASE_ONE_TRIALS_PER_DIRECTION)], phase_num=1)

    wait_for_space("Data Collection (2/2) - Press SPACE to begin")
    countdown()
    random_order = random.sample(DIRECTIONS * 5, PHASE_TWO_TOTAL_TRIALS)
    phase(random_order, phase_num=2)

    wait_for_space("Data Collection Complete - Press SPACE to exit")
    pygame.quit()

if __name__ == "__main__":
    main()
