import sys
import random
import pygame

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
FPS = 60
BLOCK_SIZE = 20

# Create the game screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define some colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class SnakeGame:
    def __init__(self):
        self.snake_pos = [200, 200]
        self.snake_body = [[200, 200], [220, 200], [240, 200]]
        self.direction = "RIGHT"
        self.score = 0
        self.apple_pos = self.generate_apple_position()

    def generate_apple_position(self):
        return [random.randint(1, WIDTH // BLOCK_SIZE - 2) * BLOCK_SIZE,
                random.randint(1, HEIGHT // BLOCK_SIZE - 2) * BLOCK_SIZE]

    def draw_snake(self):
        for pos in self.snake_body:
            pygame.draw.rect(screen, WHITE, (pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))

    def move_snake(self):
        head = self.snake_pos[:]
        if self.direction == "RIGHT":
            if (head[0] + BLOCK_SIZE) // BLOCK_SIZE < len(self.snake_body):
                head[0] += BLOCK_SIZE
            else:
                return
        elif self.direction == "LEFT":
            if head[0] % BLOCK_SIZE == 0 and head[0] >= BLOCK_SIZE:
                head[0] -= BLOCK_SIZE
            elif (head[0] - BLOCK_SIZE) // BLOCK_SIZE < len(self.snake_body):
                head[0] -= BLOCK_SIZE
        elif self.direction == "UP":
            if (head[1] - BLOCK_SIZE) // BLOCK_SIZE >= 0:
                head[1] -= BLOCK_SIZE
        elif self.direction == "DOWN":
            head[1] += BLOCK_SIZE

        self.snake_pos = head
        self.snake_body.insert(0, self.snake_pos)

    def check_collision(self):
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= WIDTH or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= HEIGHT):
            return True

        if self.snake_pos in [pos for pos in self.snake_body[1:]]:
            return True

    def check_apple_collision(self):
        if self.snake_pos == self.apple_pos:
            self.score += 1
            self.apple_pos = self.generate_apple_position()
            self.snake_body.pop()

def main():
    clock = pygame.time.Clock()
    game = SnakeGame()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and game.direction != "DOWN":
                    game.direction = "UP"
                elif event.key == pygame.K_DOWN and game.direction != "UP":
                    game.direction = "DOWN"
                elif event.key == pygame.K_LEFT and game.direction != "RIGHT":
                    game.direction = "LEFT"
                elif event.key == pygame.K_RIGHT and game.direction != "LEFT":
                    game.direction = "RIGHT"

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, RED, (game.apple_pos[0], game.apple_pos[1], BLOCK_SIZE, BLOCK_SIZE))

        if not game.check_collision():
            game.move_snake()
        else:
            return

        game.check_apple_collision()

        game.draw_snake()

        text = f"Score: {game.score}"
        font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()