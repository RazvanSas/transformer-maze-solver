
import random

from maze import Maze, bfs


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



SIZE = 8

vocab = {
    "PAD": 0,
    "EMPTY": 1,
    "WALL": 2,
    "START": 3,
    "GOAL": 4,
    "SEP": 5,
    "UP": 6,
    "DOWN": 7,
    "LEFT": 8,
    "RIGHT": 9
}

# Add one token for every possible position.
for row in range(SIZE):
    for col in range(SIZE):

        token = f"POS_{row}_{col}"

        vocab[token] = len(vocab)


def encode_state(maze, position):

    tokens = []

    # Encode the entire maze in row-major order.
    for row in range(maze.rows):
        for col in range(maze.cols):

            cell = (row, col)

            if cell == maze.start:
                tokens.append("START")

            elif cell == maze.goal:
                tokens.append("GOAL")

            elif maze.grid[row][col] == 1:
                tokens.append("WALL")

            else:
                tokens.append("EMPTY")

    tokens.append("SEP")

    row, col = position
    tokens.append(f"POS_{row}_{col}")

    # Convert token strings into integer IDs.
    input_ids = [vocab[token] for token in tokens]

    return input_ids



def get_action(current, next_position):

    row, col = current
    next_row, next_col = next_position

    if next_row == row - 1:
        return "UP"

    elif next_row == row + 1:
        return "DOWN"

    elif next_col == col - 1:
        return "LEFT"

    elif next_col == col + 1:
        return "RIGHT"

    else:
        raise ValueError("Positions are not adjacent")


def create_examples(maze, path, maze_id):

    examples = []

    # The goal itself does not require another action.
    for i in range(len(path) - 1):

        current = path[i]
        next_position = path[i + 1]

        input_ids = encode_state(maze, current)

        action = get_action(current, next_position)
        target_id = vocab[action]

        example = {
            "maze_id": maze_id,
            "input_ids": input_ids,
            "target_id": target_id
        }

        examples.append(example)

    return examples



import json
import os

random.seed(42)

NUM_MAZES = 10000

all_mazes = []

for maze_id in range(NUM_MAZES):

    maze, path = generate_maze(
        size=SIZE,
        wall_probability=0.35
    )

    examples = create_examples(
        maze,
        path,
        maze_id
    )

    all_mazes.append(examples)

    if (maze_id + 1) % 500 == 0:
        print(f"Generated {maze_id + 1} mazes")


# Shuffle maze groups, not individual examples.
random.shuffle(all_mazes)

train_end = 7000
val_end = 8000

train_mazes = all_mazes[:train_end]
val_mazes = all_mazes[train_end:val_end]
test_mazes = all_mazes[val_end:]


def flatten(groups):
    examples = []

    for group in groups:
        examples.extend(group)

    return examples


train_data = flatten(train_mazes)
val_data = flatten(val_mazes)
test_data = flatten(test_mazes)


os.makedirs("data", exist_ok=True)


def save_json(filename, data):

    with open(filename, "w") as file:
        json.dump(data, file)


save_json("data/train.json", train_data)
save_json("data/validation.json", val_data)
save_json("data/test.json", test_data)

save_json("data/vocab.json", vocab)

print("Training examples:", len(train_data))
print("Validation examples:", len(val_data))
print("Test examples:", len(test_data))
print("Vocabulary size:", len(vocab))




