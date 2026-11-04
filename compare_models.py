
import random
import statistics
import torch
from torch.distributions import Categorical

from model import MazeTransformer
from maze import Maze, generate_maze


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SUPERVISED_CHECKPOINT = "best_model.pt"
RL_CHECKPOINT = "reinforce_latest.pt"
ACTOR_CRITIC_CHECKPOINT = "actor_critic_latest.pt"

NUM_MAZES = 600
MAX_STEPS = 128
GRID_SIZE = 8
WALL_PROBABILITY = 0.25
SEED = 42

# Vocabulary token IDs
EMPTY_ID = 1
WALL_ID = 2
START_ID = 3
GOAL_ID = 4
SEP_ID = 5

# Model output order
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
EXPECTED_ACTION_TOKEN_IDS = [6, 7, 8, 9]


# --------------------------------------------------
# Encode maze state
# --------------------------------------------------

def encode_state(maze, position):
    """
    Encode the maze as:
        64 cell tokens + SEP + current-position token

    Total sequence length: 66.
    """

    tokens = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = (row, col)

            if cell == maze.start:
                token_id = START_ID
            elif cell == maze.goal:
                token_id = GOAL_ID
            elif maze.grid[row][col] == 1:
                token_id = WALL_ID
            else:
                token_id = EMPTY_ID

            tokens.append(token_id)

    tokens.append(SEP_ID)

    row, col = position
    position_token = 10 + row * GRID_SIZE + col
    tokens.append(position_token)

    if len(tokens) != 66:
        raise ValueError(
            f"Expected 66 tokens, but got {len(tokens)}."
        )

    return tokens


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    vocab_size = int(checkpoint["vocab_size"])
    action_token_ids = checkpoint["action_token_ids"]

    if isinstance(action_token_ids, torch.Tensor):
        action_token_ids = action_token_ids.tolist()

    action_token_ids = [int(x) for x in action_token_ids]
    max_seq_len = int(checkpoint.get("max_seq_len", 66))

    if vocab_size != 74:
        raise ValueError(
            f"{checkpoint_path}: expected vocab_size=74, "
            f"got {vocab_size}."
        )

    if max_seq_len != 66:
        raise ValueError(
            f"{checkpoint_path}: expected max_seq_len=66, "
            f"got {max_seq_len}."
        )

    if action_token_ids != EXPECTED_ACTION_TOKEN_IDS:
        raise ValueError(
            f"{checkpoint_path}: unexpected action token IDs: "
            f"{action_token_ids}."
        )

    model = MazeTransformer(
        vocab_size=vocab_size,
        action_token_ids=action_token_ids,
        max_seq_len=max_seq_len
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# --------------------------------------------------
# Create an independent copy of a maze
# --------------------------------------------------

def copy_maze(original_maze):
    """
    Each model must start from the same maze layout and initial position.

    Copy the grid so evaluating one model cannot affect another.
    """

    grid_copy = [row[:] for row in original_maze.grid]

    maze = Maze(
        grid_copy,
        start=original_maze.start,
        goal=original_maze.goal
    )

    maze.reset()

    return maze


# --------------------------------------------------
# Evaluate one maze
# --------------------------------------------------

@torch.no_grad()
def run_one_maze(model, original_maze, device):
    maze = copy_maze(original_maze)

    collisions = 0

    for step_index in range(1, MAX_STEPS + 1):

        # Encode the current maze and agent position.
        state_tokens = encode_state(maze, maze.position)

        state_tensor = torch.tensor(
            [state_tokens],
            dtype=torch.long,
            device=device
        )


        # Model output shape: (1, 4).
        logits = model(state_tensor)

        # Convert logits into a categorical probability distribution.
        distribution = Categorical(logits=logits)

        # Sample an action according to its probability.
        action_index = distribution.sample().item()
        #action_index = logits.argmax(dim=-1).item()

        action_name = ACTION_NAMES[action_index]

        old_position = maze.position

        # Execute the action in the environment.
        _, done = maze.step(action_name)

        # If the position did not change, the move hit a wall
        # or attempted to leave the grid.
        if maze.position == old_position:
            collisions += 1

        if done:
            return {
                "success": True,
                "steps": step_index,
                "collisions": collisions
            }

    # The agent did not reach the goal within MAX_STEPS.
    return {
        "success": False,
        "steps": MAX_STEPS,
        "collisions": collisions
    }


# --------------------------------------------------
# Evaluate a model on all test mazes
# --------------------------------------------------

def evaluate_model(model, mazes, device, model_name):
    results = [
        run_one_maze(model, maze, device)
        for maze in mazes
    ]

    successes = sum(
        result["success"] for result in results
    )

    success_rate = 100.0 * successes / len(results)

    average_steps = statistics.mean(
        result["steps"] for result in results
    )

    successful_steps = [
        result["steps"]
        for result in results
        if result["success"]
    ]

    average_successful_steps = (
        statistics.mean(successful_steps)
        if successful_steps
        else float("nan")
    )

    average_collisions = statistics.mean(
        result["collisions"]
        for result in results
    )

    print(f"\n{'=' * 60}")
    print(model_name)
    print("=" * 60)

    print(f"Mazes evaluated:             {len(results)}")
    print(f"Successful mazes:            {successes}/{len(results)}")
    print(f"Success rate:                {success_rate:.2f}%")
    print(f"Average steps (all mazes):   {average_steps:.2f}")
    print(
        f"Average steps (successes):   "
        f"{average_successful_steps:.2f}"
    )
    print(
        f"Average collisions per maze: "
        f"{average_collisions:.2f}"
    )

    return {
        "name": model_name,
        "successes": successes,
        "success_rate": success_rate,
        "average_steps": average_steps,
        "average_successful_steps": average_successful_steps,
        "average_collisions": average_collisions,
        "results": results
    }


# --------------------------------------------------
# Compare two models on identical mazes
# --------------------------------------------------

def compare_pair(results_a, results_b):
    a_only = 0
    b_only = 0
    both = 0
    neither = 0

    for a, b in zip(
        results_a["results"],
        results_b["results"]
    ):
        if a["success"] and b["success"]:
            both += 1
        elif a["success"] and not b["success"]:
            a_only += 1
        elif not a["success"] and b["success"]:
            b_only += 1
        else:
            neither += 1

    print(f"\n{results_a['name']} vs. {results_b['name']}")
    print("-" * 60)
    print(f"{results_a['name']} only: {a_only}")
    print(f"{results_b['name']} only: {b_only}")
    print(f"Both succeed: {both}")
    print(f"Neither succeeds: {neither}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    #random.seed(SEED)
    #torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Generating {NUM_MAZES} solvable mazes...")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Wall probability: {WALL_PROBABILITY}")
    print(f"Maximum steps per maze: {MAX_STEPS}")

    # Generate the test mazes only once.
    # All three models will be evaluated on these same layouts.
    mazes = [
        generate_maze(
            size=GRID_SIZE,
            wall_probability=WALL_PROBABILITY
        )[0]
        for _ in range(NUM_MAZES)
    ]

    print("\nLoading models...")

    supervised_model = load_model(
        SUPERVISED_CHECKPOINT,
        device
    )

    rl_model = load_model(
        RL_CHECKPOINT,
        device
    )

    actor_critic_model = load_model(
        ACTOR_CRITIC_CHECKPOINT,
        device
    )

    print("\nEvaluating supervised model...")

    supervised_results = evaluate_model(
        supervised_model,
        mazes,
        device,
        "Supervised model (best_model.pt)"
    )

    print("\nEvaluating REINFORCE model...")

    rl_results = evaluate_model(
        rl_model,
        mazes,
        device,
        "REINFORCE model (reinforce_latest.pt)"
    )

    print("\nEvaluating Actor-Critic model...")

    actor_critic_results = evaluate_model(
        actor_critic_model,
        mazes,
        device,
        "Actor-Critic model (actor_critic_best.pt)"
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    all_results = [
        supervised_results,
        rl_results,
        actor_critic_results
    ]

    print(f"\n{'=' * 85}")
    print("FINAL COMPARISON")
    print("=" * 85)

    print(
        f"{'Model':<38}"
        f"{'Success %':>11}"
        f"{'Avg Steps':>12}"
        f"{'Success Steps':>15}"
        f"{'Collisions':>12}"
    )

    for result in all_results:
        print(
            f"{result['name']:<38}"
            f"{result['success_rate']:>10.2f}%"
            f"{result['average_steps']:>12.2f}"
            f"{result['average_successful_steps']:>15.2f}"
            f"{result['average_collisions']:>12.2f}"
        )

    # --------------------------------------------------
    # Paired comparisons
    # --------------------------------------------------

    print("\nPAIRED COMPARISONS")

    compare_pair(
        supervised_results,
        rl_results
    )

    compare_pair(
        supervised_results,
        actor_critic_results
    )

    compare_pair(
        rl_results,
        actor_critic_results
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
