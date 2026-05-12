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

CHECKPOINT_PATH = "sts_agent/rl/model_checkpoint.pt"
LOSS_PATH = "sts_agent/rl/loss_log.txt"

STATE_DIM = 66
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
        )

        self.policy_head = nn.Linear(hidden_dim, ACTION_DIM)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        return self.policy_head(out), self.value_head(out).squeeze(-1)

    # https://docs.pytorch.org/docs/2.11/distributions.html

    def select_action_training(self, state: torch.Tensor, valid_actions: torch.Tensor):
        logits, value = self.forward(state)
        masked_logits = (
            logits.clone()
        )  # don't want to mess with the computation graph in training
        masked_logits[~valid_actions] = float("-inf")

        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def select_action(self, state: torch.Tensor, valid_actions: torch.Tensor):
        with torch.no_grad():
            logits, _ = self.forward(state)
            logits[~valid_actions] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            action = dist.sample()
        return action


model = None
optimizer = None

episode_log_probs = []
episode_rewards = []
episode_values = []

last_hp = -1
last_enemy_hp = 0


def on_combat_enter(state: dict):
    global model, optimizer, last_hp, last_enemy_hp

    model = RLModel().to(DEVICE)
    # Karpathy constant lol
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    path = Path(CHECKPOINT_PATH)
    if path.exists():
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}.")
    else:
        print(f"No checkpoint at {path}. Starting fresh.")

    last_hp = state["player"]["hp"]
    last_enemy_hp = sum(enemy.get("hp", 0) for enemy in state.get("enemies", []))


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
    global last_hp, last_enemy_hp

    hp = state["player"]["hp"]
    max_hp = state["player"]["max_hp"]
    block = state["player"]["block"]

    enemy_hp = 0
    incoming_damage = 0
    for enemy in state.get("enemies", []):
        enemy_hp += enemy.get("hp", 0)
        incoming_damage += enemy.get("intent", {}).get("damage", 0)

    damage_reward = (last_enemy_hp - enemy_hp) / max_hp
    block_reward = min(block, incoming_damage) / max(incoming_damage, 1)
    hp_penalty = (hp - last_hp) / max_hp

    decision = state.get("decision", "")
    if decision != "combat_play" and decision != "game_over":
        # this means combat ended and we survived
        end_of_combat_bonus = 5 * (hp / max_hp)
    else:
        end_of_combat_bonus = 0

    reward = damage_reward + block_reward + end_of_combat_bonus + hp_penalty
    episode_rewards.append(reward)

    last_hp = hp
    last_enemy_hp = enemy_hp


def build_state_tensor(state: dict, training: bool) -> torch.Tensor:
    if training:
        record_reward(state)

    # player info

    hp = state["player"]["hp"]
    max_hp = state["player"]["max_hp"]
    block = state["player"]["block"]

    energy = state.get("energy", 0)
    max_energy = state.get("max_energy", 0)

    # hand info

    hand = state.get("hand", [])

    playable = [0.0] * HAND_SIZE
    costs = [0.0] * HAND_SIZE
    skills = [0.0] * HAND_SIZE
    attacks = [0.0] * HAND_SIZE
    block_percents = [0.0] * HAND_SIZE
    damage_percents = [0.0] * HAND_SIZE

    for i, card in enumerate(hand):
        playable[i] = (
            1.0
            if (card.get("can_play", False) and card.get("cost", 0) <= energy)
            else 0.0
        )

        costs[i] = card.get("cost", 0) / max_energy if max_energy else 0.0

        card_type = card.get("type", "")
        stats = card.get("stats") or {}

        if card_type == "Skill":
            skills[i] = 1.0
            block_percents[i] = stats.get("block", 0.0) / max_hp

        if card_type == "Attack":
            attacks[i] = 1.0
            damage_percents[i] = stats.get("damage", 0.0) / max_hp

    # enemy info

    enemies = state.get("enemies", [])

    enemy_count = len(enemies)

    enemy_hp_ratios = []
    incoming_damage = 0
    attacking_count = 0

    for enemy in enemies:
        enemy_hp = enemy.get("hp", 0)
        enemy_max_hp = enemy.get("max_hp", 1)
        enemy_hp_ratios.append(enemy_hp / enemy_max_hp if enemy_max_hp else 0.0)

        intent = enemy.get("intent", {})
        attacking_count += 1 if intent.get("type") == "Attack" else 0
        incoming_damage += intent.get("damage", 0)

    enemy_hp_avg = sum(enemy_hp_ratios) / enemy_count if enemy_count else 0.0

    # bring together all features

    state_features = torch.tensor(
        [hp / max_hp, block / max_hp, energy / max_energy if max_energy else 0.0]
        + playable
        + costs
        + skills
        + attacks
        + block_percents
        + damage_percents
        + [
            attacking_count / enemy_count if enemy_count else 0.0,
            enemy_hp_avg,
            incoming_damage / max_hp,
        ],
        dtype=torch.float32,
    )

    return state_features


def run_inference(state: dict, training: bool) -> str:
    state_tensor = build_state_tensor(state, training)
    state_tensor = state_tensor.to(DEVICE)

    valid_actions = build_valid_actions(state)
    valid_actions = valid_actions.to(DEVICE)

    if training:
        model.train()
        action, log_prob, value = model.select_action_training(
            state_tensor, valid_actions
        )
        episode_log_probs.append(log_prob)
        episode_values.append(value)
    else:
        model.eval()
        action = model.select_action(state_tensor, valid_actions)

    return action.item()


def on_combat_end(state: dict, training: bool):
    global episode_log_probs, episode_rewards, episode_values, last_hp, last_enemy_hp

    if training:
        _ = build_state_tensor(state, training)

        # drop the first reward since it was for entering combat
        episode_rewards = episode_rewards[1:]

        # Trying to do this: https://en.wikipedia.org/wiki/Policy_gradient_method
        # REINFORCE / Monte Carlo policy gradient

        returns = []
        G = 0.0
        for reward in reversed(episode_rewards):
            G = reward + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        values = torch.stack(episode_values)

        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(episode_log_probs)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open(LOSS_PATH, "a") as f:
            f.write(f"{policy_loss.item()},{value_loss.item()}\n")

        print(f"Training step done. Loss: {loss.item():.4f}")

        path = Path(CHECKPOINT_PATH)
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
        episode_values = []

        last_hp = -1
        last_enemy_hp = 0


if __name__ == "__main__":
    import json

    with open("sts_agent/rl/example-state.json", "r") as f:
        example_state = json.load(f)

    state_tensor = build_state_tensor(example_state, training=False)

    print(state_tensor.shape)
    print(state_tensor)
