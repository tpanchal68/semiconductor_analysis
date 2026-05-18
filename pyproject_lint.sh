#!/bin/bash

# Target folder
TARGET_DIR="src tests"

echo "🧹 Cleaning cache files..."
pyclean .  # Runs on project root to clear all __pycache__

echo "🎨 Running Black (Formatting)..."
black "$TARGET_DIR"

echo "📝 Checking Docstrings (Pydocstyle)..."
pydocstyle "$TARGET_DIR"

echo "🔍 Running Flake8 (Style)..."
flake8 "$TARGET_DIR"

echo "🛡️ Running Pylint (Logic)..."
pylint --rcfile=pyproject.toml "$TARGET_DIR"

echo -e "\n✨ All checks finished for '$TARGET_DIR'!"
