#!/bin/bash

git submodule init
git submodule update

#pushd sts2-cli
#./setup.sh
#popd

python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
