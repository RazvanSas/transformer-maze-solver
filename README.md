# Transformer Maze Solver

A PyTorch project exploring supervised learning and reinforcement learning for navigation in procedurally generated 8×8 mazes. A custom decoder-only Transformer predicts the next movement from the maze layout and the agent's current position.

The project includes BFS-generated supervision, REINFORCE fine-tuning, two actor–critic variants, checkpoint comparison, and an interactive Streamlit maze editor.

## Model and data

| Component | Implementation |
| --- | --- |
| State | 64 grid-cell tokens + separator + current-position token |
| Vocabulary | 74 tokens |
| Output | UP, DOWN, LEFT, RIGHT |
| Transformer | 2 blocks, 4 attention heads, embedding dimension 64 |
| Feed-forward layers | 64 → 256 → 64 with GELU |
| Regularization | Dropout 0.1, pre-layer normalization, residual connections |
| Attention | Causal mask; final position supplies action logits |
| Supervised objective | Cross-entropy with AdamW |

`generate_dataset.py` generates 10,000 solvable mazes with wall probability 0.35 and BFS shortest-path action labels. It splits **by maze**, before flattening state/action examples: 7,000 training mazes, 1,000 validation mazes and 2,000 test mazes. States from one maze therefore stay in one partition.

## Quick start

Run commands from this repository's root. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The included `data/vocab.json` and four supplied checkpoints allow inference without regenerating the training set. PyTorch selects CUDA when available; otherwise the scripts use CPU.

In the app, draw walls, set start and goal positions, then run the model. The app loads `actor_critic_best.pt`, samples actions, and prevents immediately repeating an action that just caused a collision. Training uses fixed opposite-corner endpoints; arbitrary endpoints in the editor may generalize less reliably.

For a terminal demonstration using the supervised checkpoint:

```bash
python solve_maze.py
```

## Training

```bash
python generate_dataset.py
python train.py
python train_reinforce.py
```

Supervised training defaults to 30 epochs, batch size 64 and learning rate 3e-4. The best validation action-accuracy checkpoint is saved as `best_model.pt`. REINFORCE starts from that checkpoint, uses discounted batch-normalized returns and saves `reinforce_latest.pt`.

Choose one actor–critic experiment:

```bash
# Monte Carlo returns with a learned value baseline; starts from reinforce_latest.pt
python train_actor_critic.py

# One-step TD actor–critic; starts from best_model.pt
python train_actor_critic_td.py
```

Both actor–critic scripts save to `actor_critic_latest.pt` and `actor_critic_best.pt`; running either overwrites those files. Copy checkpoints to a separate experiment directory if you want to retain multiple runs. Their configuration constants are near the top of each script.

The Monte Carlo variant uses `G_t - V(s_t)` as its advantage. The TD variant uses `r_t + gamma * V(s_next) - V(s_t)`, with zero bootstrap at terminal states. Both use a critic loss, normalized actor advantages, entropy regularization and gradient clipping. Reward shaping uses BFS distance changes, alongside goal rewards and penalties for steps, collisions and timeouts. BFS therefore supplies supervision and reward shaping; it is not a learned component.

## Evaluation

```bash
python compare_models.py
```

The current comparison uses 600 generated solvable mazes, wall probability 0.25 and a 128-step limit. All models receive the same maze layouts. It reports success rate, average steps, average successful-path length, collisions and paired success outcomes.

Important interpretation details:

- Comparison actions are sampled; the seed-setting lines in `main()` are currently commented out. Enable both Python and PyTorch seeds when recording reproducible runs.
- Comparison loads `actor_critic_latest.pt`, despite one printed label saying `actor_critic_best.pt`. Change `ACTOR_CRITIC_CHECKPOINT` if evaluating the best checkpoint.
- Actor–critic checkpoint selection uses greedy actions on 200 fixed validation mazes, while comparison and the app use sampling. Report the decoding mode alongside results.
- Comparison generates fresh mazes rather than reading `data/test.json`; its wall probability also differs from supervised data generation.
- No benchmark scores are claimed here: the supplied archive has checkpoints but no saved evaluation report. RL fine-tuning is an experiment, not a guarantee of improved success.

## Files

| File | Purpose |
| --- | --- |
| `maze.py` | Environment, BFS and solvable-maze generation |
| `generate_dataset.py`, `dataset.py` | Supervised data generation and PyTorch dataset |
| `model.py` | Decoder-only policy Transformer |
| `actor_critic_model.py` | Transformer policy with value head |
| `train.py` | Supervised training |
| `train_reinforce.py` | REINFORCE fine-tuning |
| `train_actor_critic.py` | Monte Carlo actor–critic experiment |
| `train_actor_critic_td.py` | One-step TD actor–critic experiment |
| `compare_models.py` | Evaluation of three policy checkpoints |
| `solve_maze.py`, `app.py` | Terminal and interactive demonstrations |
| `data/vocab.json` | Token mapping matching the supplied encoding |

## Scope and next steps

This is an educational research project with a fixed 8×8 representation. Useful extensions include repeated seeded evaluations, separate reporting for greedy and sampled policies, consistent checkpoint naming, and evaluation across maze densities and endpoint locations. Training scripts overwrite checkpoints; `generate_dataset.py` also generates data when imported, so run it as a script.

The uploaded source is preserved; `train_actor_critic (1).py` was renamed to `train_actor_critic_td.py` for clarity. Some original comments describe older settings; the descriptions above follow executable code. Dependency versions are not pinned, and the full training/UI workflow has not been rerun for this packaging pass.

