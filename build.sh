#!/bin/bash

if [[ -z "$STS2_GAME_DIR" ]]
then
    echo "STS2_GAME_DIR environment variable not set. Please set it to the path of your Slay the Spire 2 installation."
    exit 1
fi

git submodule init
git submodule update

pushd sts2-cli
./setup.sh
popd

python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
