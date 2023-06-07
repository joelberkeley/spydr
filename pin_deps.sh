#!/bin/bash

set -e

VENV_DIR=$(mktemp -d)

pin_deps () {
  python3.11 -m venv $VENV_DIR/$1
  source $VENV_DIR/$1/bin/activate
  pip install --upgrade pip
  if [ "$2" = true ]; then
    pip install -e .
  fi
  pip install -r $1/requirements.txt
  pip freeze --exclude-editable spydr > $1/constraints.txt
  deactivate
}

pin_deps deps/format false
pin_deps deps/docs true
pin_deps deps/types true
pin_deps deps/test true

rm -rf $VENV_DIR