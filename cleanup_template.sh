#!/usr/bin/env bash
# cleanup_template.sh
#
# Cleans up repos created from the same bad template that bootstrapFlywheel used.
# Run from the root of the repo. Review the output before committing.
#
# Usage:
#   cd /path/to/repo
#   bash cleanup_template.sh

set -euo pipefail

echo "=== Template Cleanup Script ==="
echo "Working directory: $(pwd)"
echo ""

# Safety check
if [ ! -d .git ]; then
    echo "ERROR: Not a git repository. Run this from the repo root."
    exit 1
fi

# --- Remove broken/redundant workflow files ---
echo "Removing broken/redundant workflows..."
rm -f .github/workflows/cos_integration.yml
rm -f .github/workflows/docs.yml
rm -f .github/workflows/docs_test.yml
rm -f .github/workflows/code_quality_control.yml
rm -f .github/workflows/lints.yml
rm -f .github/workflows/pylint.yml
rm -f .github/workflows/pull-request-links.yml
rm -f .github/workflows/label.yml
rm -f .github/workflows/stale.yml
rm -f .github/workflows/pr_request_checks.yml
rm -f .github/workflows/run_test.yml
rm -f .github/workflows/quality.yml
rm -f .github/workflows/test.yml
rm -f .github/workflows/testing.yml
rm -f .github/workflows/welcome.yml
rm -f .github/workflows/python-publish.yml

# --- Remove template config files ---
echo "Removing obsolete config files..."
rm -f Dockerfile
rm -f Makefile
rm -f .pre-commit-config.yaml
rm -f .readthedocs.yml
rm -f mkdocs.yml
rm -f poetry_check.sh
rm -f requirements.txt
rm -f example.py

# --- Remove template GitHub files ---
echo "Removing template GitHub files..."
rm -rf .github/ISSUE_TEMPLATE/
rm -f .github/PULL_REQUEST_TEMPLATE.yml
rm -f .github/FUNDING.yml
rm -f .github/labeler.yml

# --- Remove dead code ---
echo "Removing dead code..."
rm -rf src/package/
rm -f src/main.py
rm -f src/project_evaluator.py

# --- Remove empty doc stubs ---
echo "Removing docs directory..."
rm -rf docs/

# --- Remove empty test dirs ---
echo "Removing empty test directories..."
rm -rf tests/performance/

# --- Remove pycache ---
echo "Removing __pycache__ directories..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# --- Fix test __init__.py files (remove broken 'from package import main') ---
echo "Fixing test __init__.py files..."
for f in tests/__init__.py tests/unit/__init__.py; do
    if [ -f "$f" ] && grep -q "from package import main" "$f"; then
        # Keep the docstring line, remove the broken import
        sed -i '/from package import main/d' "$f"
        echo "  Fixed: $f"
    fi
done

# --- Add .DS_Store to .gitignore if not present ---
if ! grep -q "\.DS_Store" .gitignore 2>/dev/null; then
    echo "Adding .DS_Store to .gitignore..."
    sed -i '1i# macOS\n.DS_Store\n' .gitignore
fi

# --- Fix ruff errors ---
echo "Auto-fixing ruff errors..."
if command -v uv &>/dev/null; then
    uv tool run ruff check --fix . 2>/dev/null || true
elif command -v ruff &>/dev/null; then
    ruff check --fix . 2>/dev/null || true
else
    echo "  WARNING: ruff not found, skipping auto-fix"
fi

echo ""
echo "=== Cleanup complete ==="
echo ""
echo "Review changes with: git status && git diff"
echo ""
echo "You still need to manually:"
echo "  1. Update pyproject.toml (fix package name, author, switch to PEP 621 + hatchling)"
echo "  2. Update/create unit-test.yml workflow to use uv"
echo "  3. Update/create ruff.yml workflow with astral-sh/ruff-action@v3"
echo "  4. Update any cron workflows to use uv"
echo "  5. Update README.md (remove references to make commands, old tools)"
