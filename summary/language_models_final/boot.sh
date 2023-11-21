#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

while getopts "r" o &>/dev/null; do
  case $o in
    "r") EXTRA="--reload-dir $SCRIPT_DIR --reload" ;;
  esac
done

shift "$(($OPTIND - 1))"

LANGUAGE_MODEL=$1

if [[ -z $LANGUAGE_MODEL ]]; then
  cd $SCRIPT_DIR/models
  for value in $(ls *.py); do
    if [[ "$value" != "_"* && "$value" == *".py" ]]; then
      MODELS="$MODELS $(basename $value .py)"
    fi
  done
  MODELS=$(echo $MODELS)
  echo "no model name was provided"
  echo "has to be one of '${MODELS}'"
  exit 1
fi

if [[ -d "$SCRIPT_DIR/venv" ]]; then
  source $SCRIPT_DIR/venv/bin/activate || exit 1
fi

PORT=${PORT-5000}
LANGUAGE_MODEL=$LANGUAGE_MODEL uvicorn --app-dir $SCRIPT_DIR app:app --host 0.0.0.0 --port $PORT $EXTRA
