#!/usr/bin/env node
const { execSync } = require('child_process');

function gitAvailable() {
  try {
    // This will fail if not a git repo or git not available
    execSync('git rev-parse --git-dir', { stdio: 'ignore' });
    return true;
  } catch (err) {
    return false;
  }
}

function runHuskyInstall() {
  try {
    // Prefer local husky binary via npx so PATH issues are reduced
    execSync('npx husky install', { stdio: 'inherit' });
    console.log('husky install succeeded');
    return 0;
  } catch (err) {
    console.error('husky install failed:', err && err.message ? err.message : err);
    return 1;
  }
}

if (!gitAvailable()) {
  console.warn('\n[prepare-husky] Git repository not detected or git not available.');
  console.warn('[prepare-husky] Skipping husky install. To install locally:');
  console.warn('  - Ensure git is installed and this is a git clone (not a zip).');
  console.warn("  - From repo root run: cd frontend && npm run prepare\n");
  // Do not fail the install; this keeps CI/sandboxes quiet.
  process.exit(0);
}

process.exit(runHuskyInstall());
