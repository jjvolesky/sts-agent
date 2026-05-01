from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

TRAINING = True
SAVE_PATH = "sts2_agent/rl/model.pth"

STATE_DIM = 111
ACTION_DIM = 11

GAMMA = 0.9


class RLModel(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, state: torch.Tensor):
        return self.net(state)

    # https://docs.pytorch.org/docs/2.11/distributions.html

    def select_action_training(self, state: torch.Tensor):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def select_action(self, state: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
        return action


model = None
optimizer = None

episode_log_probs = []
episode_rewards = []
last_hp = -1


def on_combat_enter(state: dict):
    global model, optimizer, last_hp

    model = RLModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    path = Path(SAVE_PATH)
    if path.exists():
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}.")
    else:
        print(f"No checkpoint at {path}. Starting fresh.")

    last_hp = state["player"]["hp"]


def record_reward(state: dict):
    decision = state.get("decision", "")

    """
    My thought process here (very simple starting, we can expand):
    - Winning or losing is the same as winning the last combat
    - Winning a combat is the best reward and losing is the worst
    - Taking damage is not great but not nearly as bad as losing
    """

    match decision:
        case "game_over":
            victory = state.get("victory", False)
            reward = 1.0 if victory else -1.0
        case "combat_play":
            hp = state["player"]["hp"]
            global last_hp
            if hp < last_hp:
                reward = -0.1
            else:
                reward = 0.1
            last_hp = hp
        case _:
            reward = 1.0

    episode_rewards.append(reward)


def build_state_tensor(state: dict) -> torch.Tensor:
    if TRAINING and state["round"] > 1:
        record_reward(state)

    # TODO turn the state into a feature vector

    # HP

    # Energy

    # Hand (card types, costs, etc.)

    # Enemies (HP, intents, etc.)

    return torch.zeros(STATE_DIM, dtype=torch.float32)


def run_inference(state: dict) -> str:
    state_tensor = build_state_tensor(state)
    state_tensor = state_tensor.to(DEVICE)

    if TRAINING:
        model.train()
        action, log_prob = model.select_action_training(state_tensor)
        episode_log_probs.append(log_prob)
    else:
        model.eval()
        action = model.select_action(state_tensor)

    return action.item()


def on_combat_end(state: dict):
    global episode_log_probs, episode_rewards, last_hp

    if TRAINING:
        _ = build_state_tensor(state)

        # Trying to do this: https://en.wikipedia.org/wiki/Policy_gradient_method
        # REINFORCE

        returns = []
        G = 0.0
        for reward in reversed(episode_rewards):
            G = reward + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(episode_log_probs)
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Training step done. Loss: {loss.item():.4f}")

        path = Path(SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

        episode_log_probs = []
        episode_rewards = []
        last_hp = -1
