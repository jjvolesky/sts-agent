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
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    try:
        while game_process.poll() is None:
            line = game_process.stdout.readline()
            state = json.loads(line)
            pprint(state)
    finally:
        game_process.kill()


if __name__ == "__main__":
    main()
