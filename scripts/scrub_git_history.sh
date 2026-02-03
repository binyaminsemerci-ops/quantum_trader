#!/bin/bash
# Git History Secret Scrubbing Script
# Removes leaked API keys from entire git history

set -e

echo "🧹 Git History Secret Scrubbing"
echo "================================"
echo ""
echo "⚠️  WARNING: This will rewrite git history!"
echo "⚠️  All contributors must re-clone after force-push"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "❌ git-filter-repo not found"
    echo ""
    echo "Install with:"
    echo "  pip install git-filter-repo"
    echo ""
    echo "Or on Linux:"
    echo "  sudo apt install git-filter-repo"
    exit 1
fi

# Create replacements file
REPLACEMENTS_FILE="$(pwd)/.git-secret-replacements.txt"
cat > "$REPLACEMENTS_FILE" << 'EOF'
# Old leaked testnet keys (now rotated)
e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD==>REDACTED_TESTNET_API_KEY
ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja==>REDACTED_TESTNET_API_SECRET
EOF

echo "📝 Created replacements file: $REPLACEMENTS_FILE"
echo ""

# Show what will be replaced
echo "Will replace in all history:"
cat "$REPLACEMENTS_FILE"
echo ""

# Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Make sure working tree is clean
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Working tree has uncommitted changes"
    echo "Please commit or stash changes first"
    exit 1
fi

# Create backup tag
BACKUP_TAG="backup-before-secret-scrub-$(date +%Y%m%d-%H%M%S)"
git tag "$BACKUP_TAG"
echo "✅ Created backup tag: $BACKUP_TAG"
echo "   (To restore: git reset --hard $BACKUP_TAG)"
echo ""

# Run filter-repo
echo "🔄 Running git-filter-repo (this may take a minute)..."
git filter-repo --replace-text "$REPLACEMENTS_FILE" --force

echo ""
echo "✅ History rewritten locally"
echo ""

# Add remote back (filter-repo removes it)
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
if [ -n "$REMOTE_URL" ]; then
    git remote add origin "$REMOTE_URL"
    echo "✅ Re-added remote: origin"
else
    echo "⚠️  Could not determine remote URL"
    echo "Manually add with: git remote add origin <URL>"
fi

echo ""
echo "📤 Next step: Force push"
echo ""
echo "  git push --force --all"
echo "  git push --force --tags"
echo ""
echo "⚠️  After force push, all team members MUST:"
echo "  1. Delete their local repo"
echo "  2. Fresh clone from GitHub"
echo ""
echo "Cleanup:"
rm -f "$REPLACEMENTS_FILE"
echo "  ✅ Deleted replacements file"
echo ""
echo "🎯 Backup tag preserved: $BACKUP_TAG"
echo "   List all backups: git tag | grep backup-before"
echo "   Delete backup: git tag -d $BACKUP_TAG"
