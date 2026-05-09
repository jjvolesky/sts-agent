import argparse
import json
import os
import random
import subprocess
from time import sleep

from sts_agent.input_cleaner import clean_input
from sts_agent.rl.model import on_combat_enter, on_combat_end, run_inference

CLI_DIR = os.path.join(os.path.dirname(__file__), "../sts2-cli")
TRAINING_LOG_PATH = "sts_agent/rl/training_log.txt"

START_CMD = {
    "cmd": "start_run",
    "character": "Ironclad",
    "seed": None,
    "ascension": 0,
}

TRAINING_GAMES = 250
random.seed(42)


def main(training: bool):
    if training:
        for i in range(TRAINING_GAMES):
            START_CMD["seed"] = str(i)
            game_process = start_game()
            game_loop(game_process, training)
    else:
        START_CMD["seed"] = "cs540_test_seed"
        game_process = start_game()
        game_loop(game_process, training)


def start_game():
    game_process = subprocess.Popen(
        ["dotnet", "run", "--project", "src/Sts2Headless/Sts2Headless.csproj"],
        cwd=CLI_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        # stderr was verbose and for debugging, maybe we can store it into a file in case
        # we run into bugs in the sts2-cli stuff but that's not our focus
        text=True,
        bufsize=1,
    )

    sleep(0.5)

    print(f"Starting game with seed: {START_CMD['seed']}")
    game_process.stdin.write(json.dumps(START_CMD) + "\n")
    game_process.stdin.flush()

    sleep(0.5)

    return game_process


def game_loop(game_process: subprocess.Popen[str], training: bool):
    in_combat = False
    combats = 0

    try:
        while game_process.poll() is None:
            line = clean_input(game_process.stdout.readline())
            state = json.loads(line)

            state_type = state["type"]
            print(f"{state_type=}")

            if state_type == "error":
                if random.random() < 0.5:
                    action = {"cmd": "action", "action": "proceed"}
                else:
                    action = {"cmd": "action", "action": "leave_room"}

                game_process.stdin.write(json.dumps(action) + "\n")
                game_process.stdin.flush()

                sleep(0.5)
                continue

            if "decision" not in state:
                sleep(0.5)
                continue

            decision = state["decision"]
            print(f"{decision=}")

            if in_combat and decision != "combat_play" and decision != "card_select":
                in_combat = False
                on_combat_end(state, training)

            match decision:
                case "bundle_select":
                    action = {
                        "cmd": "action",
                        "action": "select_bundle",
                        "args": {"bundle_index": 0},
                    }
                case "card_reward":
                    action = card_reward(state)
                case "card_select":
                    action = card_select(state)
                case "combat_play":
                    current_round = state["round"]

                    if not in_combat and current_round == 1:
                        in_combat = True
                        on_combat_enter(state)
                        combats += 1

                    action = combat_play_rl(state, training)
                case "event_choice":
                    action = event_choice(state)
                case "game_over":
                    victory = state.get("victory", False)
                    player = state.get("player", {})
                    context = state.get("context", {})

                    print(
                        f"\n{'VICTORY' if victory else 'DEFEAT'} at act {context.get('act')}, "
                        f"floor {context.get('floor')} "
                        f"(HP: {player.get('hp')}/{player.get('max_hp')}, "
                        f"Gold: {player.get('gold')}, "
                        f"Deck: {player.get('deck_size')} cards)\n"
                    )

                    if training:
                        with open(TRAINING_LOG_PATH, "a") as f:
                            act = context["act"]
                            floor = context["floor"]
                            f.write(f"{act},{floor},{combats}\n")

                    break
                case "map_select":
                    action = map_select(state)
                case "rest_site":
                    action = rest_site(state)
                case "shop":
                    action = {"cmd": "action", "action": "leave_room"}
                case _:
                    action = {"cmd": "action", "action": "proceed"}

            print(f"{action=}")
            game_process.stdin.write(json.dumps(action) + "\n")
            game_process.stdin.flush()

            sleep(0.5)
    finally:
        game_process.kill()


def card_reward(state: dict):
    cards = state.get("cards", [])
    if cards:
        action = {
            "cmd": "action",
            "action": "select_card_reward",
            "args": {"card_index": 0},
        }
    else:
        action = {"cmd": "action", "action": "skip_card_reward"}
    return action


def card_select(state: dict):
    cards = state.get("cards", [])
    if cards:
        action = {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": "0"},
        }
    else:
        action = {"cmd": "action", "action": "skip_select"}
    return action


def combat_play(state: dict):
    hand = state.get("hand", [])
    energy = state.get("energy", 0)
    enemies = state.get("enemies", [])

    playable = [
        c for c in hand if c.get("can_play", False) and (c.get("cost", 0) <= energy)
    ]

    if playable:
        card = playable[0]
        args = {"card_index": card["index"]}
        if card.get("target_type") == "AnyEnemy" and enemies:
            args["target_index"] = 0
        action = {"cmd": "action", "action": "play_card", "args": args}
    else:
        action = {"cmd": "action", "action": "end_turn"}
    return action


def combat_play_rl(state: dict, training: bool):
    action = run_inference(state, training)

    if action == 10:
        action_dict = {"cmd": "action", "action": "end_turn"}
    else:
        hand = state["hand"]
        enemies = state["enemies"]

        card = hand[action]
        args = {"card_index": card["index"]}
        if card.get("target_type") == "AnyEnemy" and enemies:
            args["target_index"] = 0
        action_dict = {"cmd": "action", "action": "play_card", "args": args}

    return action_dict


def event_choice(state: dict):
    options = state.get("options", [])
    if options:
        choice = next((o for o in options if not o.get("is_locked")), options[0])
        action = {
            "cmd": "action",
            "action": "choose_option",
            "args": {"option_index": choice["index"]},
        }
    else:
        action = {"cmd": "action", "action": "leave_room"}
    return action


def map_select(state: dict):
    choices = state.get("choices", [])
    choice = random.choice(choices)
    action = {
        "cmd": "action",
        "action": "select_map_node",
        "args": {"col": choice["col"], "row": choice["row"]},
    }
    return action


def rest_site(state: dict):
    options = state.get("options", [])
    enabled = [o for o in options if o.get("is_enabled", True)]
    heal = next((o for o in enabled if o.get("option_id") == "HEAL"), None)
    choice = heal or (enabled[0] if enabled else None)
    if choice:
        action = {
            "cmd": "action",
            "action": "choose_option",
            "args": {"option_index": choice["index"]},
        }
    else:
        action = {"cmd": "action", "action": "leave_room"}
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    main(args.training)
