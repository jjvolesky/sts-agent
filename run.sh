#!/bin/bash

if [[ -z "$STS2_GAME_DIR" ]]
then
    echo "STS2_GAME_DIR environment variable not set. Please set it to the path of your Slay the Spire 2 installation."
    exit 1
fi

if [[ ! -d "venv" ]]
then
    echo "Virtual environment not found. Please run build.sh."
    exit 1
fi

source ./venv/bin/activate
python3 -m sts_agent
