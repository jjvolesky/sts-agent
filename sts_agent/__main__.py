import argparse
import json
import os
import random
import subprocess
from time import sleep

import numpy as np

from sts_agent.input_cleaner import clean_input
from sts_agent.rl.model import on_combat_enter, on_combat_end, run_inference
from sts_agent.pathing import get_best_path

CLI_DIR = os.path.join(os.path.dirname(__file__), "../sts2-cli")
TRAINING_LOG_PATH = "sts_agent/rl/training_log.txt"

TRAINING_GAMES = 500
TESTING_GAMES = 50

TRAINING_SEED = ""
TESTING_SEED = "cs540_test_seed_"

random.seed(42)

PREV_STATE = None  # necessary for shop removal


def main(training: bool, rl: bool, pathing: bool):
    if training:
        games = TRAINING_GAMES
        base_seed = TRAINING_SEED

        # overrides for when we are training
        rl = True
        pathing = False
    else:
        games = TESTING_GAMES
        base_seed = TESTING_SEED

    acts, floors, combats = [], [], []
    for i in range(games):
        seed = f"{base_seed}{i}"
        game_process = start_game(seed)

        act, floor, combat_count = game_loop(game_process, training, rl, pathing)
        acts.append(act)
        floors.append(floor)
        combats.append(combat_count)

    if not training:
        acts = np.asarray(acts)
        floors = np.asarray(floors)
        combats = np.asarray(combats)

        print(f"\nAct: {acts.mean():.2f} +/- {acts.std():.2f}")
        print(f"Floor: {floors.mean():.2f} +/- {floors.std():.2f}")
        print(f"Combats: {combats.mean():.2f} +/- {combats.std():.2f}")


def start_game(seed: str):
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

    print(f"Starting game with seed: {seed}")
    start_command = {
        "cmd": "start_run",
        "character": "Ironclad",
        "seed": seed,
        "ascension": 0,
    }

    game_process.stdin.write(json.dumps(start_command) + "\n")
    game_process.stdin.flush()

    sleep(0.5)
    return game_process


def game_loop(game_process: subprocess.Popen[str], training: bool, rl: bool, pathing: bool):
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

                if rl:
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
                        combats += 1

                        if rl:
                            on_combat_enter(state)

                    if rl:
                        action = combat_play_rl(state, training)
                    else:
                        action = combat_play(state)
                case "event_choice":
                    action = event_choice(state)
                case "game_over":
                    victory = state.get("victory", False)
                    player = state.get("player", {})
                    context = state.get("context", {})

                    act = context["act"]
                    floor = context["floor"]

                    print(
                        f"\n{'VICTORY' if victory else 'DEFEAT'} at act {context.get('act')}, "
                        f"floor {context.get('floor')} "
                        f"(HP: {player.get('hp')}/{player.get('max_hp')}, "
                        f"Gold: {player.get('gold')}, "
                        f"Deck: {player.get('deck_size')} cards)\n"
                    )

                    if training:
                        with open(TRAINING_LOG_PATH, "a") as f:
                            f.write(f"{act},{floor},{combats}\n")

                    return act, floor, combats
                case "map_select":
                    if pathing:
                        action = smart_map_select(state)
                    else:
                        action = map_select(state)
                case "rest_site":
                    action = rest_site(state)
                case "shop":
                    action = shop_select(state)
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
            "args": {"card_index": random.randint(0, len(cards)-1)}, # select card reward at random
        }
    else:
        action = {"cmd": "action", "action": "skip_card_reward"}
    return action


def card_select(state: dict):
    cards = state.get("cards", [])
    if PREV_STATE and PREV_STATE["decision"] == "shop" and not index_of_strike(cards) == -1:
        action = {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": f"{index_of_strike(cards)}"}
        }
    elif cards:
        action = {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": f"{random.randint(0, len(cards)-1)}"}, # select card at random
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
    choices = [o for o in options if not o.get("is_locked")]
    if len(choices) > 0:
        action = {
            "cmd": "action",
            "action": "choose_option",
            "args": {"option_index": random.randint(0, len(choices)-1)}, # randomly select an option if able to
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


def smart_map_select(state: dict):
    choice = get_best_path(state)[0]
    action = {
        'cmd': 'action',
        'action': 'select_map_node',
        'args': {'col': choice['col'], 'row': choice['row']}
    }
    return action


def shop_select(state: dict):
    gold = state["player"]["gold"]
    relics = state.get("relics", [])
    action = None
    if gold >= state["card_removal_cost"] and has_strike(state["player"]):
        global PREV_STATE
        PREV_STATE = state
        action = {
            "cmd": "action",
            "action": "remove_card"
        }
    if relics: # try to buy relics if have enough gold
        relic_tuples = [(relic["cost"], relic["index"]) for relic in relics if relic['is_stocked']]
        relic_tuples.sort(key=lambda x: x[0])
        for relic in relic_tuples:
            if gold >= relic[0]:
                action = {
                    "cmd": "action",
                    "action": "buy_relic",
                    "args": {"relic_index": int(relic[1])}
                }
    return action if action else {"cmd": "action", "action": "leave_room"}


def has_strike(player_state: dict):
    for card in player_state["deck"]:
        if card["name"] == "Strike": return True
    return False


def index_of_strike(cards):
    for card in cards:
        if card["name"] == "Strike": return card["index"]
    return -1


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
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--pathing", action="store_true")

    args = parser.parse_args()

    main(args.training, args.rl, args.pathing)

    # with open("example-json/example-shop.json", 'r') as f:
    #     shop_state = json.load(f)
    #     action = shop_select(shop_state)
    #     print(action)
