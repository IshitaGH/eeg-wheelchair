#!/usr/bin/env python3

import pygame
import time
import os
import csv
from datetime import datetime

# Initialize Pygame
pygame.init()
# start windowed at 800×600, but we’ll query the real size each frame
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("EEG Data Collection - P300 - Phase 1")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Constants
FLASH_COUNT       = 3     # Number of flashes per trial
INTER_FLASH_DELAY = 3     # Delay between flashes
FLASH_DURATION    = 3     # Duration to show the image
BASELINE_DURATION = 10    # Baseline period after each trial
NUM_TRIALS        = 3     # Number of trials

IMAGE_PATH  = "images"
HOUSE_IMAGE = "red.png"
house_image = pygame.image.load(os.path.join(IMAGE_PATH, HOUSE_IMAGE))

timestamp_log    = []
stopwatch_start  = None
date_str         = datetime.now().strftime("%Y-%m-%d")
folder_name      = f"Data Collection - P300 {date_str} S1 P1"
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
    while True:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_SPACE:
                return
            if evt.type == pygame.VIDEORESIZE:
                # re-create screen at new size
                screen_size = evt.size
                pygame.display.set_mode(screen_size, pygame.RESIZABLE)

        screen.fill(BLACK)
        w, h = screen.get_size()
        font = pygame.font.Font(None, 48)
        rendered = font.render(text, True, WHITE)
        rect = rendered.get_rect(center=(w//2, h//2))
        screen.blit(rendered, rect)
        pygame.display.flip()
        time.sleep(0.01)


def countdown():
    font = pygame.font.Font(None, 72)
    for i in range(5, 0, -1):
        for evt in pygame.event.get():
            if evt.type == pygame.VIDEORESIZE:
                pygame.display.set_mode(evt.size, pygame.RESIZABLE)

        screen.fill(BLACK)
        w, h = screen.get_size()
        rendered = font.render(str(i), True, WHITE)
        rect = rendered.get_rect(center=(w//2, h//2))
        screen.blit(rendered, rect)
        pygame.display.flip()
        time.sleep(1)


def show_image():
    # get current window size
    w, h = screen.get_size()
    screen.fill(BLACK)

    # define image size (you can also scale with w/h if you like)
    img_w, img_h = 200, 200
    # center in left half
    x = w//10 - img_w//2
    y = h//2 - img_h//2

    screen.blit(pygame.transform.scale(house_image, (img_w, img_h)), (x, y))
    pygame.display.flip()


def flash_sequence():
    log_event("Start Flashing - Forward")
    for flash_num in range(1, FLASH_COUNT + 1):
        show_image()
        log_event(f"Flash [Forward] #{flash_num}")
        time.sleep(FLASH_DURATION)

        # blank
        screen.fill(BLACK)
        pygame.display.flip()
        if flash_num != FLASH_COUNT:
            log_event("Blank Screen - 0.5s delay")
            time.sleep(INTER_FLASH_DELAY)

    log_event("End Flashing - Forward")
    # baseline
    screen.fill(BLACK)
    pygame.display.flip()
    time.sleep(BASELINE_DURATION)


def phase():
    global stopwatch_start, timestamp_log
    for i in range(NUM_TRIALS):
        stopwatch_start = None
        timestamp_log   = []
        log_event(f"Start Trial {i+1} - Forward")
        flash_sequence()
        log_event(f"End Trial {i+1} - Forward")
        save_log(f"phase1_trial{i+1}.csv")


def main():
    wait_for_space("Data Collection - Press SPACE to begin")
    countdown()
    phase()
    wait_for_space("Data Collection Complete - Press SPACE to exit")
    pygame.quit()


if __name__ == "__main__":
    main()
