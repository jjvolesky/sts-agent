#!/bin/bash

git submodule init
git submodule update

cd sts2-cli
./setup.sh

python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
