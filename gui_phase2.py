import pygame
import time
import os
import csv
import random
from datetime import datetime

pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("EEG Data Collection - P300 - Phase 2")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)

DIRECTIONS = ["Forward", "Backward", "Left", "Right"]
TRIAL_DURATION = 10
BASELINE_DURATION = 10
PHASE_TWO_TOTAL_TRIALS = 20
FLASH_DURATION = 0.1  # balanced for visibility and P300 spacing
INTER_FLASH_DELAY = 0.1  # allow reset before next flash

IMAGE_PATH = "images"
ARROW_PATH = "images_arrows"
direction_images = {}
arrow_images = {}
for direction in DIRECTIONS:
    direction_images[direction] = pygame.image.load(os.path.join(IMAGE_PATH, f"{direction.lower()}.png"))
    arrow_images[direction] = pygame.image.load(os.path.join(ARROW_PATH, f"{direction.lower()}_arrow.png"))

timestamp_log = []
stopwatch_start = None

date_str = datetime.now().strftime("%Y-%m-%d")
folder_name = f"Data Collection - P300 {date_str} S1 P2"
os.makedirs(folder_name, exist_ok=True)

def log_event(event):
    global stopwatch_start
    now = time.time()
    if stopwatch_start is None:
        stopwatch_start = now
    timestamp_log.append([now, f"{now - stopwatch_start:.3f}", event])

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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

def countdown():
    font = pygame.font.Font(None, 72)
    for i in range(5, 0, -1):
        screen.fill(BLACK)
        rendered = font.render(str(i), True, WHITE)
        rect = rendered.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(rendered, rect)
        pygame.display.flip()
        time.sleep(1)

def show_direction_prompt(direction):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 36)
    text = font.render(direction, True, RED)
    screen.blit(text, (SCREEN_WIDTH - 200, 20))
    img_w, img_h = 150, 150
    positions = get_positions(img_w, img_h)
    for dir in DIRECTIONS:
        arrow = arrow_images[dir]
        screen.blit(pygame.transform.scale(arrow, (img_w, img_h)), positions[dir])
    pygame.display.flip()
    time.sleep(2)

def get_positions(img_w, img_h):
    return {
        "Forward":  (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT // 6 - img_h // 2),
        "Backward": (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT * 5 // 6 - img_h // 2),
        "Left":     (SCREEN_WIDTH // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2),
        "Right":    (SCREEN_WIDTH * 5 // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2)
    }

def flash_sequence(direction):
    img_w, img_h = 150, 150
    positions = get_positions(img_w, img_h)
    start_time = time.time()
    log_event(f"Start Flashing - {direction}")
    while time.time() - start_time < TRIAL_DURATION:
        single_target = random.choice(DIRECTIONS)
        screen.fill(BLACK)
        for dir in DIRECTIONS:
            arrow = arrow_images[dir]
            screen.blit(pygame.transform.scale(arrow, (img_w, img_h)), positions[dir])
        # Overlay just one image (the single flashing target)
        overlay = direction_images[single_target]
        screen.blit(pygame.transform.scale(overlay, (img_w, img_h)), positions[single_target])
        pygame.display.flip()
        log_event(f"Flash [{single_target}]")
        time.sleep(FLASH_DURATION)
        screen.fill(BLACK)
        for dir in DIRECTIONS:
            arrow = arrow_images[dir]
            screen.blit(pygame.transform.scale(arrow, (img_w, img_h)), positions[dir])
        pygame.display.flip()
        time.sleep(INTER_FLASH_DELAY)
    log_event(f"End Flashing - {direction}")
    screen.fill(BLACK)
    pygame.display.flip()
    time.sleep(BASELINE_DURATION)
    log_event(f"Baseline - {direction}")

def phase():
    random_order = random.sample(DIRECTIONS * 5, PHASE_TWO_TOTAL_TRIALS)
    for i, direction in enumerate(random_order):
        global stopwatch_start, timestamp_log
        stopwatch_start = None
        timestamp_log = []
        log_event(f"Start Trial {i+1} - {direction}")
        show_direction_prompt(direction)
        flash_sequence(direction)
        log_event(f"End Trial {i+1} - {direction}")
        save_log(f"phase2_trial{i+1}_{direction}.csv")

def main():
    wait_for_space("Data Collection (2/2) - Press SPACE to begin")
    countdown()
    phase()
    wait_for_space("Phase 2 Complete - Press SPACE to exit")
    pygame.quit()

if __name__ == "__main__":
    main()
