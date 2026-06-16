#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${1:-}"
if [ -z "$REMOTE_URL" ]; then
  echo "Usage: bash scripts/setup_git_repo.sh <github-remote-url>"
  exit 1
fi

if [ ! -d ".git" ]; then
  git init
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

MODEL_PATH="models/pigdetect_best.pt"
if [ -f "$MODEL_PATH" ]; then
  MODEL_SIZE=$(wc -c < "$MODEL_PATH")
  if [ "$MODEL_SIZE" -gt 100000000 ]; then
    echo "Warning: $MODEL_PATH is larger than 100MB. Use Git LFS before pushing to GitHub."
  fi
fi

git add .
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Initial pig count project"
fi

git branch -M main
git push -u origin main
echo "Repository pushed to main."
