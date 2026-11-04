
import json
import time

import streamlit as st
import torch

from maze import Maze, bfs
from model import MazeTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SIZE = 8
MAX_STEPS = 128
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

st.set_page_config(
    page_title="Transformer Maze Solver",
    page_icon="🧩",
    layout="wide"
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

def initialize_state():
    if "grid" not in st.session_state:
        st.session_state.grid = [
            [0 for _ in range(SIZE)]
            for _ in range(SIZE)
        ]

    if "start" not in st.session_state:
        st.session_state.start = (0, 0)

    if "goal" not in st.session_state:
        st.session_state.goal = (SIZE - 1, SIZE - 1)

    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = "Wall"

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


initialize_state()


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open("data/vocab.json", "r") as f:
        vocab = json.load(f)

    checkpoint = torch.load(
        "actor_critic_best.pt",
        map_location=device,
        weights_only=True
    )

    model = MazeTransformer(
        vocab_size=checkpoint["vocab_size"],
        action_token_ids=checkpoint["action_token_ids"],
        max_seq_len=checkpoint["max_seq_len"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, device


# --------------------------------------------------
# Encode the current maze state
# --------------------------------------------------

def encode_state(grid, start, goal, position, vocab):
    tokens = []

    for row in range(SIZE):
        for col in range(SIZE):
            cell = (row, col)

            if cell == start:
                token = "START"
            elif cell == goal:
                token = "GOAL"
            elif grid[row][col] == 1:
                token = "WALL"
            else:
                token = "EMPTY"

            tokens.append(vocab[token])

    tokens.append(vocab["SEP"])
    tokens.append(
        vocab[f"POS_{position[0]}_{position[1]}"]
    )

    assert len(tokens) == 66

    return torch.tensor(tokens, dtype=torch.long)


# --------------------------------------------------
# Environment transition
# --------------------------------------------------

def execute_action(grid, position, action):
    """
    Apply the action to the maze.

    A wall or boundary prevents movement.
    Returns the resulting position.
    """

    row, col = position

    moves = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }

    dr, dc = moves[action]

    new_row = row + dr
    new_col = col + dc

    # Check boundaries.
    if not (
        0 <= new_row < SIZE
        and 0 <= new_col < SIZE
    ):
        return position

    # Check walls.
    if grid[new_row][new_col] == 1:
        return position

    return (new_row, new_col)


# --------------------------------------------------
# Draw the maze
# --------------------------------------------------

def render_maze(
    grid,
    start,
    goal,
    position=None,
    path=None
):
    path = set(path or [])

    html = """
    <style>
        .maze-grid {
            display: grid;
            grid-template-columns: repeat(8, minmax(0, 1fr));
            gap: 4px;
            width: 100%;
            max-width: 560px;
        }

        .maze-cell {
            aspect-ratio: 1 / 1;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 5px;
            font-size: clamp(12px, 2.5vw, 22px);
            font-weight: bold;
            border: 1px solid #c9cdd3;
        }

        .empty {
            background: #f4f6f8;
            color: #64748b;
        }

        .wall {
            background: #374151;
            border-color: #374151;
        }

        .start {
            background: #15965b;
            color: white;
        }

        .goal {
            background: #2675d8;
            color: white;
        }

        .agent {
            background: #e9a23b;
            color: #111827;
        }

        .path {
            background: #fef3c7;
            color: #92400e;
        }
    </style>
    <div class="maze-grid">
    """

    for row in range(SIZE):
        for col in range(SIZE):
            cell = (row, col)

            if cell == position:
                css_class = "agent"
                symbol = "🤖"
            elif cell == start:
                css_class = "start"
                symbol = "S"
            elif cell == goal:
                css_class = "goal"
                symbol = "G"
            elif grid[row][col] == 1:
                css_class = "wall"
                symbol = ""
            elif cell in path:
                css_class = "path"
                symbol = "•"
            else:
                css_class = "empty"
                symbol = ""

            html += (
                f'<div class="maze-cell {css_class}">'
                f'{symbol}</div>'
            )

    html += "</div>"

    return html


# --------------------------------------------------
# Model inference
# --------------------------------------------------


def predict_action(
    model, vocab, device, grid, start, goal, position,
    forbidden_action=None
):
    input_ids = encode_state(
        grid, start, goal, position, vocab
    )

    input_ids = input_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids)

        # Prevent immediately repeating the action
        # that hit a wall on the previous step.
        if forbidden_action is not None:
            forbidden_index = ACTIONS.index(forbidden_action)
            logits[0, forbidden_index] = float("-inf")

        # Convert logits to probabilities.
        probabilities = torch.softmax(logits, dim=1)

        # Sample an allowed action.
        action_index = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

    return ACTIONS[action_index]



# --------------------------------------------------
# Solve the maze
# --------------------------------------------------

def solve_maze(model, vocab, device, grid, start, goal, delay):
    # Build an independent environment for BFS reference.
    maze = Maze(
        grid=[row[:] for row in grid],
        start=start,
        goal=goal
    )

    optimal_path = bfs(maze)

    if not optimal_path:
        return {
            "success": False,
            "reason": "The maze has no valid path to the goal.",
            "path": [start],
            "optimal_length": None,
            "steps": 0
        }

    position = start
    visited = [position]
    forbidden_action = None
    grid_display = st.empty()

    grid_display.markdown(
        render_maze(
            grid, start, goal, position, visited
        ),
        unsafe_allow_html=True
    )

    progress = st.progress(0.0)
    status = st.empty()
    action_log = st.empty()

    for step in range(1, MAX_STEPS + 1):
        if position == goal:
            break

        action = predict_action(
            model, vocab, device,
            grid, start, goal, position,
            forbidden_action=forbidden_action
        )

        old_position = position

        position = execute_action(
            grid, position, action
        )

        if position == old_position:
            forbidden_action = action
        else:
            forbidden_action = None

        visited.append(position)

        grid_display.markdown(
            render_maze(
                grid, start, goal,
                position, visited
            ),
            unsafe_allow_html=True
        )

        status.write(
            f"Step {step}/{MAX_STEPS} — "
            f"Action: **{action}** — "
            f"Position: **{position}**"
        )

        progress.progress(
            min(step / MAX_STEPS, 1.0)
        )

        action_log.code(
            f"{old_position} --{action}--> {position}"
        )

        if position == goal:
            break

        if delay > 0:
            time.sleep(delay)

    success = position == goal

    return {
        "success": success,
        "reason": (
            "Goal reached!"
            if success
            else f"Goal not reached within {MAX_STEPS} steps."
        ),
        "path": visited,
        "optimal_length": len(optimal_path) - 1,
        "steps": len(visited) - 1
    }


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("🧩 Transformer Maze Solver")

st.write(
    "Design a maze, then watch your trained "
    "decoder-only Transformer try to solve it."
)

st.caption(
    "Green = Start | Blue = Goal | Dark = Wall | "
    "Yellow = Visited path | Orange = Agent"
)

# Sidebar controls.
with st.sidebar:
    st.header("Maze settings")

    st.radio(
        "Click mode",
        ["Wall", "Start", "Goal", "Erase"],
        key="edit_mode"
    )

    if st.button("Clear maze", use_container_width=True):
        st.session_state.grid = [
            [0 for _ in range(SIZE)]
            for _ in range(SIZE)
        ]
        st.session_state.start = (0, 0)
        st.session_state.goal = (SIZE - 1, SIZE - 1)
        st.session_state.last_result = None
        st.rerun()

    if st.button("Random maze", use_container_width=True):
        import random

        while True:
            grid = [
                [
                    1 if random.random() < 0.25 else 0
                    for _ in range(SIZE)
                ]
                for _ in range(SIZE)
            ]

            grid[0][0] = 0
            grid[SIZE - 1][SIZE - 1] = 0

            maze = Maze(
                grid=grid,
                start=(0, 0),
                goal=(SIZE - 1, SIZE - 1)
            )

            if bfs(maze):
                st.session_state.grid = grid
                st.session_state.start = (0, 0)
                st.session_state.goal = (SIZE - 1, SIZE - 1)
                st.session_state.last_result = None
                break

        st.rerun()

    delay = st.slider(
        "Animation delay (seconds)",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05
    )

# Draw editable grid.
st.subheader("Maze editor")

for row in range(SIZE):
    columns = st.columns(SIZE)

    for col in range(SIZE):
        cell = (row, col)

        if cell == st.session_state.start:
            symbol = "🟢"
        elif cell == st.session_state.goal:
            symbol = "🔵"
        elif st.session_state.grid[row][col] == 1:
            symbol = "⬛"
        else:
            symbol = "⬜"

        with columns[col]:
            if st.button(
                symbol,
                key=f"cell_{row}_{col}",
                help=f"Cell ({row}, {col})",
                use_container_width=True
            ):
                mode = st.session_state.edit_mode

                if mode == "Start":
                    if cell != st.session_state.goal:
                        st.session_state.start = cell
                        st.session_state.grid[row][col] = 0

                elif mode == "Goal":
                    if cell != st.session_state.start:
                        st.session_state.goal = cell
                        st.session_state.grid[row][col] = 0

                elif mode == "Wall":
                    if cell not in (
                        st.session_state.start,
                        st.session_state.goal
                    ):
                        st.session_state.grid[row][col] = 1

                elif mode == "Erase":
                    if cell not in (
                        st.session_state.start,
                        st.session_state.goal
                    ):
                        st.session_state.grid[row][col] = 0

                st.session_state.last_result = None
                st.rerun()

st.caption(
    "Select a click mode in the sidebar, then click cells to edit them."
)

# Load model and solve.
st.divider()

if st.button(
    "▶ Solve with Transformer",
    type="primary",
    use_container_width=True
):
    try:
        model, vocab, device = load_model()

        st.write(f"Running inference on: **{device}**")

        result = solve_maze(
            model=model,
            vocab=vocab,
            device=device,
            grid=[
                row[:] for row in st.session_state.grid
            ],
            start=st.session_state.start,
            goal=st.session_state.goal,
            delay=delay
        )

        st.session_state.last_result = result

    except FileNotFoundError as error:
        st.error(
            f"Could not find a required file: {error.filename}. "
            "Check that best_model.pt and data/vocab.json exist."
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")
        st.exception(error)

# Display results after execution.
result = st.session_state.last_result

if result is not None:
    st.subheader("Results")

    if result["success"]:
        st.success("The Transformer reached the goal!")
    else:
        st.warning(result["reason"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Model steps", result["steps"])

    if result["optimal_length"] is not None:
        col2.metric(
            "BFS shortest path",
            result["optimal_length"]
        )

        if result["optimal_length"] > 0:
            ratio = (
                result["steps"] / result["optimal_length"]
            )
            col3.metric("Steps / optimal", f"{ratio:.2f}×")
        else:
            col3.metric(
                "Steps / optimal",
                "—"
            )
    else:
        col2.metric("BFS shortest path", "Unavailable")
        col3.metric("Steps / optimal", "—")

    st.write("**Model's visited path:**")
    st.code(" → ".join(map(str, result["path"])))
