import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# Particle class
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.color = (255, 255, 255)
        self.size = random.randint(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.x <= 0 or self.x >= width:
            self.vx *= -1
        if self.y <= 0 or self.y >= height:
            self.vy *= -1

    def draw(self, scr):
        pygame.draw.circle(scr, self.color, (int(self.x), int(self.y)), self.size)

# Create particles
particles = [Particle(random.uniform(0, width), random.uniform(0, height)) for _ in range(100)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for particle in particles:
        particle.update()
        particle.draw(screen)

    pygame.display.flip()
    pygame.time.delay(30)

pygame.quit()
sys.exit()
