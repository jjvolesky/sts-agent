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

STATE_DIM = 35
ACTION_DIM = 11

GAMMA = 0.9

HAND_SIZE = 10


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

    def select_action_training(self, state: torch.Tensor, valid_actions: torch.Tensor):
        logits = self.forward(state)
        masked_logits = (
            logits.clone()
        )  # don't want to mess with the computation graph in training
        masked_logits[~valid_actions] = float("-inf")

        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def select_action(self, state: torch.Tensor, valid_actions: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(state)
            logits[~valid_actions] = float("-inf")

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


def build_valid_actions(state: dict) -> torch.Tensor:
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)

    hand = state.get("hand", [])
    energy = state.get("energy", 0)

    for i, card in enumerate(hand):
        playable = card.get("can_play", False) and (card.get("cost", 0) <= energy)
        mask[i] = playable

    mask[10] = True  # end turn is always valid

    return mask


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

    # player info

    hp = state["player"]["hp"]
    max_hp = state["player"]["max_hp"]

    energy = state.get("energy", 0)
    max_energy = state.get("max_energy", 0)

    # hand info

    hand = state.get("hand", [])

    playable = [0.0] * HAND_SIZE
    skills = [0.0] * HAND_SIZE
    attacks = [0.0] * HAND_SIZE

    for i, card in enumerate(hand):
        playable[i] = (
            1.0
            if (card.get("can_play", False) and card.get("cost", 0) <= energy)
            else 0.0
        )
        skills[i] = 1.0 if card.get("type") == "Skill" else 0.0
        attacks[i] = 1.0 if card.get("type") == "Attack" else 0.0

    # enemy info

    enemies = state.get("enemies", [])

    enemy_count = len(enemies)
    enemy_hp_ratios = []
    for enemy in enemies:
        enemy_hp = enemy.get("hp", 0)
        enemy_max_hp = enemy.get("max_hp", 1)
        enemy_hp_ratios.append(enemy_hp / enemy_max_hp if enemy_max_hp else 0.0)
    enemy_hp_avg = sum(enemy_hp_ratios) / enemy_count if enemy_count else 0.0
    incoming_damage = sum(e.get("intent", {}).get("damage", 0) for e in enemies)

    state_features = torch.tensor(
        [hp / max_hp, energy / max_energy if max_energy else 0.0]
        + playable
        + skills
        + attacks
        + [
            enemy_count,
            enemy_hp_avg,
            incoming_damage / enemy_count if enemy_count else 0.0,
        ],
        dtype=torch.float32,
    )

    return state_features


def run_inference(state: dict) -> str:
    state_tensor = build_state_tensor(state)
    state_tensor = state_tensor.to(DEVICE)

    valid_actions = build_valid_actions(state)
    valid_actions = valid_actions.to(DEVICE)

    if TRAINING:
        model.train()
        action, log_prob = model.select_action_training(state_tensor, valid_actions)
        episode_log_probs.append(log_prob)
    else:
        model.eval()
        action = model.select_action(state_tensor, valid_actions)

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


if __name__ == "__main__":
    import json

    with open("sts_agent/rl/example-state.json", "r") as f:
        example_state = json.load(f)

    state_tensor = build_state_tensor(example_state)
    print(state_tensor.shape)
    print(state_tensor)
