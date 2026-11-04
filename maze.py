
from collections import deque


class Maze:

    def __init__(self, grid=None, start=(0, 0), goal=None):

        # 0 = empty cell, 1 = wall
        self.grid = grid

        self.start = (0, 0)
        self.goal = goal
        self.position = self.start

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    def is_valid(self, row, col):

        # Check boundaries.
        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        # Check walls.
        if self.grid[row][col] == 1:
            return False

        return True

    def get_neighbors(self, position):

        row, col = position

        # UP, DOWN, LEFT, RIGHT
        possible_moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        neighbors = []

        for dr, dc in possible_moves:

            new_row = row + dr
            new_col = col + dc

            if self.is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def step(self, action):

        moves = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }

        if action not in moves:
            raise ValueError("Unknown action")

        dr, dc = moves[action]

        new_row = self.position[0] + dr
        new_col = self.position[1] + dc

        if self.is_valid(new_row, new_col):
            self.position = (new_row, new_col)

        done = self.position == self.goal

        return self.position, done

    def reset(self):
        self.position = self.start
        return self.position

    def display(self, path=None):

        if path is None:
            path = []

        for row in range(self.rows):

            line = ""

            for col in range(self.cols):

                position = (row, col)

                if position == self.start:
                    line += " S "
                elif position == self.goal:
                    line += " G "
                elif position == self.position:
                    line += " A "
                elif self.grid[row][col] == 1:
                    line += " # "
                elif position in path:
                    line += " . "
                else:
                    line += "   "

            print(line)

        print()



def bfs(maze):

    start = maze.start
    goal = maze.goal

    queue = deque([start])

    # The start has no predecessor.
    parent = {start: None}

    while queue:

        current = queue.popleft()

        if current == goal:
            break

        for neighbor in maze.get_neighbors(current):

            # Have we already discovered this cell?
            if neighbor not in parent:

                parent[neighbor] = current
                queue.append(neighbor)

    # If the goal was never discovered, no path exists.
    if goal not in parent:
        return None

    # Reconstruct the path by following predecessors.
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    # Reconstruction went from goal to start.
    path.reverse()

    return path



import random



def generate_maze(size=8, wall_probability=0.25):

    while True:

        grid = []

        for row in range(size):

            current_row = []

            for col in range(size):

                if random.random() < wall_probability:
                    current_row.append(1)
                else:
                    current_row.append(0)

            grid.append(current_row)

        start = (0, 0)
        goal = (size - 1, size - 1)

        # Start and goal must be open.
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0

        maze = Maze(grid, start, goal)

        path = bfs(maze)

        if path is not None:
            return maze, path



