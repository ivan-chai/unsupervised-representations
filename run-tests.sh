#!/usr/bin/env bash
set -e

SOURCE_DIR=`dirname "${BASH_SOURCE[0]}"`
ROOT=`realpath "$SOURCE_DIR"`
TESTS_ROOT="$ROOT/urep-tests"
echo $TESTS_ROOT

echo "Test comparator"
python3 "$TESTS_ROOT/comparator-test.py"

echo "Test loss"
python3 "$TESTS_ROOT/loss-test.py"

echo "=========================="
echo "      ALL TESTS OK!"
echo "=========================="
