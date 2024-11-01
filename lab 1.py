import numpy as np
from collections import deque
import random
import pygame
import time

# Define colors for different elements in the visualization
BLACK = (0, 0, 0)      # Obstacles
WHITE = (255, 255, 255) # Background
RED = (255, 0, 0)      # Path taken
GREEN = (0, 255, 0)    # Target position
BLUE = (0, 0, 255)     # Start position
YELLOW = (255, 255, 0) # Moving agent

class Environment:
    def __init__(self, size=10):
        # Initialize the grid environment with given size
        self.size = size
        self.grid = np.zeros((size, size))  # Create empty grid
        self.place_random_obstacles()        # Add obstacles
        self.start_pos = (0, 0)             # Set start position at top-left
        self.target_pos = (size-1, size-1)  # Set target position at bottom-right
        self.grid[self.target_pos] = 2      # Mark target position on grid

    def place_random_obstacles(self):
        # Place obstacles with 30% probability, avoiding start and target positions
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.3 and (i, j) != (0, 0) and (i, j) != (self.size-1, self.size-1):
                    self.grid[i][j] = 1

class Agent:
    def __init__(self, environment):
        # Initialize agent with reference to environment
        self.env = environment
        self.position = environment.start_pos
        self.path_history = []
        
    def get_valid_moves(self):
        # Check all possible moves in four directions (right, down, left, up)
        valid_moves = []
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in moves:
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            # Validate move: within grid bounds and not an obstacle
            if (0 <= new_x < self.env.size and 
                0 <= new_y < self.env.size and 
                self.env.grid[new_x][new_y] != 1):
                valid_moves.append((new_x, new_y))
                
        return valid_moves

    def bfs_pathfinding(self):
        # Implement Breadth-First Search algorithm to find path to target
        queue = deque([(self.position, [self.position])])
        visited = {self.position}
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Check if target reached
            if current_pos == self.env.target_pos:
                return path
            
            self.position = current_pos
            valid_moves = self.get_valid_moves()
            
            # Explore all possible moves from current position
            for next_pos in valid_moves:
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return None  # Return None if no path found

class Visualization:
    def __init__(self, cell_size=50):
        # Initialize pygame and set cell size for grid
        self.cell_size = cell_size
        pygame.init()
        
    def display_grid(self, env, path=None):
        # Calculate window dimensions
        window_size = env.size * self.cell_size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Intelligent Agent Navigation")
        
        running = True
        current_step = 0  # Track agent's current position in path
        
        while running:
            # Handle window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(WHITE)  # Clear screen with background color
            
            # Draw grid cells and static elements
            for i in range(env.size):
                for j in range(env.size):
                    rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, rect, 1)  # Draw cell borders
                    
                    # Color cells based on their content
                    if (i, j) == env.target_pos:
                        pygame.draw.rect(screen, GREEN, rect)  # Target
                    elif env.grid[i][j] == 1:
                        pygame.draw.rect(screen, BLACK, rect)  # Obstacle
                    elif path and (i, j) in path[:current_step]:
                        pygame.draw.rect(screen, RED, rect)    # Path taken
            
            # Draw moving agent
            if path and current_step < len(path):
                agent_pos = path[current_step]
                agent_rect = pygame.Rect(agent_pos[1] * self.cell_size, 
                                       agent_pos[0] * self.cell_size,
                                       self.cell_size, self.cell_size)
                pygame.draw.rect(screen, YELLOW, agent_rect)
                
                # Move agent to next position after delay
                if current_step < len(path) - 1:
                    current_step += 1
                    time.sleep(0.5)
            
            pygame.display.flip()  # Update display
        
        pygame.quit()

def simulate():
    # Create environment and agent
    env = Environment()
    agent = Agent(env)
    path = agent.bfs_pathfinding()
    
    # Display simulation if path found
    if path:
        print("Path found! Steps:", len(path)-1)
        vis = Visualization()
        vis.display_grid(env, path)
    else:
        print("No path found!")

# Entry point of program
if __name__ == "__main__":
    simulate()