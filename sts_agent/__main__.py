import json
import os
from pprint import pprint
import subprocess
import sys

CLI_DIR = os.path.join(os.path.dirname(__file__), "../sts2-cli")


def main():
    game_process = subprocess.Popen(
        ["dotnet", "run", "--project", "src/Sts2Headless/Sts2Headless.csproj"],
        cwd=CLI_DIR,
        stdin=sys.stdin,
        # I pointed this to sys stdin so that I could test from the command line, but we probably
        # want to make this a pipe as well and then create a dict for actions, turn it into a string (jsondumps),
        # and flush it into the stdin pipe
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        # stderr was verbose and for debugging, maybe we can store it into a file in case
        # we run into bugs in the sts2-cli stuff but that's not our focus
        text=True,
        bufsize=1,
    )

    try:
        while game_process.poll() is None:
            line = game_process.stdout.readline()

            state = json.loads(line)
            pprint(state)

            # Depending on what type of action we have to make (from the state), we can choose which agent to interact with
            # I'm envisioning a combat agent (RL) and a simple A* pathing agent to start
            # The RL agent can be loaded in training mode for the duration of the encounter and then saved at the end
    finally:
        game_process.kill()


if __name__ == "__main__":
    main()
