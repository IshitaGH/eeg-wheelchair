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
pygame.display.set_caption("EEG Data Collection - SSVEP (Multi-Target)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Movement Directions and Frequencies (Hz)
DIRECTIONS = {
    "Forward": 8,
    "Backward": 10,
    "Left": 12,
    "Right": 15
}

# Experiment Parameters
NUM_TRIALS = 20
FOCUS_TIME = 5
DIRECTION_PREP_TIME = 5

# Image loading
IMAGE_PATH = "images"
direction_images = {}
for direction in DIRECTIONS:
    path = os.path.join(IMAGE_PATH, f"{direction.lower()}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image for {direction}: {path}")
    direction_images[direction] = pygame.image.load(path)

# Timestamps and logging
stopwatch_start = None
timestamp_log = []
current_direction = None

# Create a folder for today's session
date_str = datetime.now().strftime("%Y-%m-%d")
folder_name = f"Data Collection - SSVEP {date_str}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def get_elapsed_time():
    if stopwatch_start is None:
        return "0.000"
    return f"{(time.time() - stopwatch_start):.3f}"

def save_timestamps():
    if current_direction is None:
        return
    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = os.path.join(folder_name, f"{date_str}_{current_direction}_{timestamp}.csv")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Elapsed Time (s)", "Event"])
        writer.writerows(timestamp_log)

def update_screen_size():
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()

def get_scaled_font(size):
    return pygame.font.Font(None, int(size * (SCREEN_WIDTH / 800)))

def log_event(event_name):
    global stopwatch_start
    real_time = time.time()
    if stopwatch_start is None:
        stopwatch_start = real_time
    elapsed_time = get_elapsed_time()
    timestamp_log.append([real_time, elapsed_time, event_name])

def wait_for_space(text):
    update_screen_size()
    screen.fill(WHITE)
    font = get_scaled_font(36)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

def display_countdown(direction):
    start_time = time.time()
    while time.time() - start_time < DIRECTION_PREP_TIME:
        update_screen_size()
        screen.fill(WHITE)
        font = get_scaled_font(36)
        countdown = DIRECTION_PREP_TIME - int(time.time() - start_time)
        text_surface = font.render(f"Focus on {direction} image", True, BLACK)
        countdown_surface = font.render(f"Starting in {countdown} seconds", True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        countdown_rect = countdown_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
        screen.blit(text_surface, text_rect)
        screen.blit(countdown_surface, countdown_rect)
        pygame.display.flip()
        pygame.time.delay(1000)

def flicker_images(focus_direction, duration):
    start_time = time.time()
    frame_clock = pygame.time.Clock()
    flicker_phases = {dir: 0 for dir in DIRECTIONS}

    image_size = (150, 150)
    spacing = SCREEN_WIDTH // 5
    y_pos = SCREEN_HEIGHT // 2 - image_size[1] // 2
    img_w, img_h = image_size
    positions = {
        "Forward":  (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT // 6 - img_h // 2),
        "Backward": (SCREEN_WIDTH // 2 - img_w // 2, SCREEN_HEIGHT * 5 // 6 - img_h // 2),
        "Left":     (SCREEN_WIDTH // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2),
        "Right":    (SCREEN_WIDTH * 5 // 6 - img_w // 2, SCREEN_HEIGHT // 2 - img_h // 2),
    }


    log_event(f"Flicker Start - {focus_direction}")

    while time.time() - start_time < duration:
        update_screen_size()
        screen.fill(WHITE)
        current_ticks = pygame.time.get_ticks()

        for direction, freq in DIRECTIONS.items():
            interval_ms = 1000 / (2 * freq)  # 2x for full cycle (on/off)
            visible = (current_ticks // interval_ms) % 2 == 0
            if visible:
                img = pygame.transform.scale(direction_images[direction], image_size)
                screen.blit(img, positions[direction])

        pygame.display.flip()
        frame_clock.tick(60)

    log_event(f"Flicker End - {focus_direction}")

def instruction_screen():
    wait_for_space("Welcome to SSVEP Training. Press SPACE to begin.")

def run_trials():
    global stopwatch_start, current_direction, timestamp_log
    directions = list(DIRECTIONS.keys())

    for trial_num in range(NUM_TRIALS):
        random.shuffle(directions)
        for direction in directions:
            current_direction = direction
            stopwatch_start = None
            timestamp_log = []
            wait_for_space(f"Trial {trial_num + 1}: Press SPACE to prepare.")
            display_countdown(direction)
            flicker_images(direction, FOCUS_TIME)
            save_timestamps()

    display_text("Training Complete! Thank you!", 5)
    pygame.quit()

def display_text(text, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        update_screen_size()
        screen.fill(WHITE)
        font = get_scaled_font(36)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        pygame.time.delay(100)

# Main execution
if __name__ == "__main__":
    instruction_screen()
    run_trials()
