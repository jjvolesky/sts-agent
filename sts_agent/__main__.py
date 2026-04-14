import json
import os
import random
import subprocess
from time import sleep

CLI_DIR = os.path.join(os.path.dirname(__file__), "../sts2-cli")

START_CMD = {
    "cmd": "start_run",
    "character": "Ironclad",
    "seed": "cs540_test_seed",
    "ascension": 0,
}


def main():
    game_process = start_game()
    game_loop(game_process)


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

    game_process.stdin.write(json.dumps(START_CMD) + "\n")
    game_process.stdin.flush()

    return game_process


def game_loop(game_process):
    last_action = None

    try:
        while game_process.poll() is None:
            line = game_process.stdout.readline()
            state = json.loads(line)

            type = state["type"]
            print(f"{type=}")

            if type == "error":
                print(f"{last_action=}")

                if last_action == "end_turn":
                    action = {"cmd": "action", "action": "proceed"}
                else:
                    action = {"cmd": "action", "action": "leave_room"}

                print(f"{action=}")
                last_action = action["action"]

                game_process.stdin.write(json.dumps(action) + "\n")
                game_process.stdin.flush()

                sleep(1)
                continue

            if not "decision" in state:
                sleep(1)
                continue

            decision = state["decision"]
            print(f"{decision=}")

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
                    action = combat_play(state)
                case "event_choice":
                    action = event_choice(state)
                case "game_over":
                    victory = state.get("victory", False)
                    player = state.get("player", {})
                    print(
                        f"\n{'VICTORY' if victory else 'DEFEAT'} at act {state.get('act')}, "
                        f"floor {state.get('floor')} "
                        f"(HP: {player.get('hp')}/{player.get('max_hp')}, "
                        f"Gold: {player.get('gold')}, "
                        f"Deck: {player.get('deck_size')} cards)"
                    )
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

            last_action = action["action"]
            sleep(1)
    finally:
        game_process.kill()


def card_reward(state):
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


def card_select(state):
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


def combat_play(state):
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


def event_choice(state):
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


def map_select(state):
    choices = state.get("choices", [])
    choice = random.choice(choices)
    action = {
        "cmd": "action",
        "action": "select_map_node",
        "args": {"col": choice["col"], "row": choice["row"]},
    }
    return action


def rest_site(state):
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
    main()
