import pygame
import random
import time
from pylsl import StreamInfo, StreamOutlet

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)  # Allow window resizing
pygame.display.set_caption("EEG Data Collection - Imagined Movement")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 255)

# Movement Directions
MOVEMENT_DIRECTIONS = ["Forward", "Backward", "Left", "Right"]

# Experiment Parameters
NUM_TRIALS = 20  # Number of trials per direction
BASELINE_TIME = 5  # Baseline period in seconds
IMAGINATION_TIME = 10  # Imagination period in seconds
REST_TIME = 5  # Rest period in seconds
DIRECTION_PREP_TIME = 5  # Time before each direction starts

def update_screen_size():
    """Update screen size dynamically for resizing."""
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()

def get_scaled_font(size):
    """Return a dynamically scaled font size."""
    return pygame.font.Font(None, int(size * (SCREEN_WIDTH / 800)))

# Initialize LSL Stream for event markers
info = StreamInfo(name='MarkerStream', type='Markers', channel_count=1, nominal_srate=0,
                  channel_format='string', source_id='marker123')
outlet = StreamOutlet(info)

def send_marker(marker):
    """Send event marker via LSL."""
    outlet.push_sample([marker])

def display_text(text, duration, phase):
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
        progress_width = int((1 - elapsed_time / duration) * SCREEN_WIDTH)
        pygame.draw.rect(screen, BLUE, (0, 20, progress_width, 10))
        
        pygame.display.flip()
        pygame.time.delay(50)  # Small delay to prevent high CPU usage

def display_countdown(direction):
    """Display direction indication with countdown before starting trials."""
    start_time = time.time()
    while time.time() - start_time < DIRECTION_PREP_TIME:
        update_screen_size()
        screen.fill(WHITE)
        font = get_scaled_font(36)
        countdown = DIRECTION_PREP_TIME - int(time.time() - start_time)
        text_surface = font.render(f"We will begin the {direction} direction phase now", True, BLACK)
        countdown_surface = font.render(f"Starting in {countdown} seconds", True, BLACK)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        countdown_rect = countdown_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
        screen.blit(text_surface, text_rect)
        screen.blit(countdown_surface, countdown_rect)
        pygame.display.flip()
        pygame.time.delay(1000)  # Delay for countdown

def instruction_screen():
    """Display study instructions before starting."""
    update_screen_size()
    screen.fill(WHITE)
    font = get_scaled_font(36)
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
        screen.fill(WHITE)
        display_countdown(direction)
        
        for trial in range(1, NUM_TRIALS + 1):
            display_text("+ (Baseline)", BASELINE_TIME, "baseline")
            send_marker(f"Baseline Start - {direction}")
            display_text(f"Imagine Moving {direction}", IMAGINATION_TIME, "imagination")
            send_marker(f"Imagination Start - {direction}")
            display_text("Relax", REST_TIME, "rest")
            send_marker(f"Rest Start - {direction}")

    display_text("Experiment Complete! Thank you!", 5, "info")
    pygame.quit()

# Main Execution
if __name__ == "__main__":
    instruction_screen()
    run_trials()
