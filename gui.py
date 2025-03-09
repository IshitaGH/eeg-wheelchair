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
pygame.display.set_caption("EEG Data Collection - Imagined Movement")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 255)

# Movement Directions
MOVEMENT_DIRECTIONS = ["Forward", "Backward", "Left", "Right"]

# Experiment Parameters
NUM_TRIALS = 20
BASELINE_TIME = 5
IMAGINATION_TIME = 10
REST_TIME = 5
DIRECTION_PREP_TIME = 30

# Variables for stopwatch and timing
stopwatch_start = None
timestamp_log = []
current_direction = None

# Create a folder for today's session
date_str = datetime.now().strftime("%Y-%m-%d")
folder_name = f"Data Collection - {date_str}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def get_elapsed_time():
    """Returns stopwatch time in milliseconds."""
    if stopwatch_start is None:
        return "0.000"
    return f"{(time.time() - stopwatch_start):.3f}"

def save_timestamps():
    """Saves timestamps for the current direction in a CSV file."""
    if current_direction is None:
        return

    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = os.path.join(folder_name, f"{date_str}_{current_direction}_{timestamp}.csv")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Elapsed Time (s)", "Event"])
        writer.writerows(timestamp_log)

def update_screen_size():
    """Update screen size dynamically for resizing."""
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()

def get_scaled_font(size):
    """Return a dynamically scaled font size."""
    return pygame.font.Font(None, int(size * (SCREEN_WIDTH / 800)))

def log_event(event_name):
    """Logs an event with real-world and stopwatch timestamps."""
    global stopwatch_start
    real_time = time.time()
    
    # Start stopwatch on the first baseline event
    if stopwatch_start is None and "Baseline Start" in event_name:
        stopwatch_start = real_time

    elapsed_time = get_elapsed_time()
    timestamp_log.append([real_time, elapsed_time, event_name])

def display_text(text, duration):
    """Display text and progress bar for a given duration."""
    start_time = time.time()
    while time.time() - start_time < duration:
        update_screen_size()
        screen.fill(WHITE)
        font = get_scaled_font(36)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(text_surface, text_rect)

        # Draw progress bar
        elapsed_time = time.time() - start_time
        progress_width = int((elapsed_time / duration) * SCREEN_WIDTH)
        pygame.draw.rect(screen, BLUE, (0, 20, progress_width, 10))

        pygame.display.flip()
        pygame.time.delay(50)

def display_countdown(direction):
    """Display countdown before trials start."""
    start_time = time.time()
    while time.time() - start_time < DIRECTION_PREP_TIME:
        update_screen_size()
        screen.fill(WHITE)
        font = get_scaled_font(36)
        countdown = DIRECTION_PREP_TIME - int(time.time() - start_time)
        text_surface = font.render(f"Prepare for {direction} trials", True, BLACK)
        countdown_surface = font.render(f"Starting in {countdown} seconds", True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        countdown_rect = countdown_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
        screen.blit(text_surface, text_rect)
        screen.blit(countdown_surface, countdown_rect)
        pygame.display.flip()
        pygame.time.delay(1000)

def wait_for_space(text):
    """Displays a message and waits for the space bar to be pressed."""
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

def instruction_screen():
    """Display instructions before starting."""
    wait_for_space("Welcome! Press SPACE to begin.")

def run_trials():
    """Run trials for all movement directions, saving data per direction."""
    global timestamp_log, current_direction, stopwatch_start

    randomized_directions = random.sample(MOVEMENT_DIRECTIONS, len(MOVEMENT_DIRECTIONS))

    for i, direction in enumerate(randomized_directions):
        if i > 0:
            wait_for_space(f"Finished the {randomized_directions[i - 1]} direction. Press SPACE to start the next one.")
        
        current_direction = direction
        timestamp_log = []  # Reset log for each direction
        stopwatch_start = None  # Reset stopwatch
        
        display_countdown(direction)

        for trial in range(1, NUM_TRIALS + 1):
            log_event(f"Baseline Start - {direction}")
            display_text("+ (Baseline)", BASELINE_TIME)

            log_event(f"Imagination Start - {direction}")
            display_text(f"Imagine Moving {direction}", IMAGINATION_TIME)

            log_event(f"Rest Start - {direction}")
            display_text("Relax", REST_TIME)

        save_timestamps()  # Save after each direction

    display_text("Experiment Complete! Thank you!", 5)
    pygame.quit()

# Main Execution
if __name__ == "__main__":
    instruction_screen()
    run_trials()
