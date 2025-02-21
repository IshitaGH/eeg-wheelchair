import pygame
import random
import time
from pylsl import StreamInfo, StreamOutlet

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("EEG Data Collection - Imagined Movement")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Fonts
font = pygame.font.Font(None, 36)
large_font = pygame.font.Font(None, 50)

# Movement Directions
MOVEMENT_DIRECTIONS = ["Forward", "Backward", "Left", "Right"]

# Experiment Parameters
NUM_TRIALS = 50  # Number of trials per direction
BASELINE_TIME = 2  # Baseline period in seconds
IMAGINATION_TIME = 6  # Imagination period in seconds
REST_TIME = 3  # Rest period in seconds

# Initialize LSL Stream for event markers
info = StreamInfo(name='MarkerStream', type='Markers', channel_count=1, nominal_srate=0,
                  channel_format='string', source_id='marker123')
outlet = StreamOutlet(info)

def send_marker(marker):
    """Send event marker via LSL."""
    outlet.push_sample([marker])

def display_text(text, duration=2):
    """Display text on the screen for a given duration."""
    screen.fill(WHITE)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    time.sleep(duration)

def instruction_screen():
    """Display study instructions before starting."""
    screen.fill(WHITE)
    instructions = [
        "Welcome to the EEG Study!",
        "You will be asked to imagine moving in different directions.",
        "Do NOT physically move, only visualize the movement.",
        "Each trial consists of:",
        "1. A baseline period (Relax)",
        "2. An imagination period (Focus on the movement)",
        "3. A rest period (Clear your mind)",
        "Press SPACE to begin the experiment."
    ]

    y_offset = 50
    for line in instructions:
        text_surface = font.render(line, True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(text_surface, text_rect)
        y_offset += 50

    pygame.display.flip()

    # Wait for SPACE key to start
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

def run_trials():
    """Run the full set of trials for all movement directions."""
    randomized_directions = random.sample(MOVEMENT_DIRECTIONS, len(MOVEMENT_DIRECTIONS))

    for direction in randomized_directions:
        display_text(f"Get Ready for {direction} Trials", 3)

        for trial in range(1, NUM_TRIALS + 1):
            # Baseline Period
            display_text("+ (Baseline)", BASELINE_TIME)
            send_marker(f"Baseline Start - {direction}")

            # Imagination Period
            display_text(f"Imagine Moving {direction}", IMAGINATION_TIME)
            send_marker(f"Imagination Start - {direction}")

            # Rest Period
            display_text("Relax", REST_TIME)
            send_marker(f"Rest Start - {direction}")

        # Short break between directions
        display_text("Short Break - Next Direction Soon", 5)

    # End of Experiment
    display_text("Experiment Complete! Thank you!", 5)
    pygame.quit()

# Main Execution
if __name__ == "__main__":
    instruction_screen()
    run_trials()