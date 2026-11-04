
import json
import random

import torch

from maze import Maze, bfs
from model import MazeTransformer


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
SIZE = 8
MAX_STEPS = 128


def generate_test_maze(size=SIZE, wall_probability=0.25):
    """Generate a solvable maze that we can test the model on."""
    while True:
        grid = [
            [
                1 if random.random() < wall_probability else 0
                for _ in range(size)
            ]
            for _ in range(size)
        ]

        start = (0, 0)
        goal = (size - 1, size - 1)

        # The start and goal must be open.
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0

        maze = Maze(grid=grid, start=start, goal=goal)

        # Reject mazes that have no path to the goal.
        if bfs(maze):
            return maze


def encode_state(maze, position, vocab):
    """Encode the maze and current position into 66 token IDs."""
    tokens = []

    for row in range(SIZE):
        for col in range(SIZE):
            cell = (row, col)

            if cell == maze.start:
                token = "START"
            elif cell == maze.goal:
                token = "GOAL"
            elif maze.grid[row][col] == 1:
                token = "WALL"
            else:
                token = "EMPTY"

            tokens.append(vocab[token])

    # Add the separator and current-position token.
    tokens.append(vocab["SEP"])
    tokens.append(vocab[f"POS_{position[0]}_{position[1]}"])

    assert len(tokens) == 66

    return torch.tensor(tokens, dtype=torch.long)


def solve_maze(model, maze, vocab, device):
    """Let the model control the maze until it wins or reaches the limit."""
    position = maze.start
    visited = [position]

    for step_number in range(1, MAX_STEPS + 1):
        if position == maze.goal:
            return True, visited

        input_ids = encode_state(maze, position, vocab)
        input_ids = input_ids.unsqueeze(0).to(device)  # (1, 66)

        with torch.no_grad():
            logits = model(input_ids)  # (1, 4)
            action_index = logits.argmax(dim=1).item()

        action = ACTIONS[action_index]

        # Execute the selected action in the environment.
        old_position = position
        position, done = maze.step(action)

        visited.append(position)

        print(
            f"Step {step_number:3d}: "
            f"{old_position} --{action}--> {position}"
        )

        if done or position == maze.goal:
            return True, visited

    return position == maze.goal, visited


def main():
    #random.seed(123)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    with open("data/vocab.json", "r") as f:
        vocab = json.load(f)

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device
    )

    model = MazeTransformer(
        vocab_size=checkpoint["vocab_size"],
        action_token_ids=checkpoint["action_token_ids"],
        max_seq_len=checkpoint["max_seq_len"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    maze = generate_test_maze()

    # BFS gives us a reference solution, not actions for the model.
    optimal_path = bfs(maze)

    print("\nGenerated an unseen, solvable maze.")
    print("BFS shortest-path length:", len(optimal_path) - 1)
    print("\nMaze layout:")
    maze.display()

    success, visited = solve_maze(model, maze, vocab, device)

    print("\nVisited positions:", len(visited))
    print("Unique positions:", len(set(visited)))

    if success:
        print("RESULT: The model reached the goal!")
    else:
        print("RESULT: The model did not reach the goal.")

    print("\nVisited path:")
    maze.display(visited)


if __name__ == "__main__":
    main()
