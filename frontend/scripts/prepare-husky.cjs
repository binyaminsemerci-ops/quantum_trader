#!/usr/bin/env node
const { execSync } = require('child_process');

function gitAvailable() {
  try {
    execSync('git rev-parse --git-dir', { stdio: 'ignore' });
    return true;
  } catch (err) {
    return false;
  }
}

function gitDotFolderPresent() {
  const fs = require('fs');
  const path = require('path');
  try {
    const repoRoot = path.resolve(__dirname, '..', '..');
    return fs.existsSync(path.join(repoRoot, '.git'));
  } catch (e) {
    return false;
  }
}

function runHuskyInstall() {
  try {
    execSync('npx husky install', { stdio: 'inherit' });
    console.log('husky install succeeded');
    return 0;
  } catch (err) {
    console.error('husky install failed:', err && err.message ? err.message : err);
    return 1;
  }
}

// Be extra defensive: if git isn't available OR the .git folder is missing, skip husky in this environment.
if (!gitAvailable() || !gitDotFolderPresent()) {
  console.warn('\n[prepare-husky] Git repository not detected or git not available.');
  console.warn('[prepare-husky] Skipping husky install. To install locally:');
  console.warn('  - Ensure git is installed and this is a git clone (not a zip).');
  console.warn("  - From repo root run: cd frontend && npm run prepare\n");
  process.exit(0);
}

process.exit(runHuskyInstall());
