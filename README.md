# sts-agent

Agent to assist in slaying the Spire.

## Requirements

A major requirement is owning a copy of Slay the Spire 2. An environment variable needs to be defined pointing to where the game is located:

```sh
export STS2_GAME_DIR="..."
```

Another requirement is having a .NET SDK/runtime available. Without this, things will not build in the next step that uses `dotnet run`.

## Setup

When first cloned, run `build.sh`.

This will add the git submodule of the project we forked, build that (which will pull files from the game directory), and then install Pip dependencies from the `requirements.txt` file.

Note that the forked project is a Git submodule, which has been added as an SSH submodule. This will not work if SSH cloning is not set up.

After that, `run.sh` should work, which will play a seeded run using our methodology and trained model. To train instead of running the seeded test run, use `run.sh --training`.
