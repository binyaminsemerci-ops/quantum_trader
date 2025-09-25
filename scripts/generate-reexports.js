#!/usr/bin/env node
// Conservative re-export generator
// Usage: node scripts/generate-reexports.js [--yes] [--root <path>]
// By default the script performs a dry-run and prints planned changes.

const fs = require('fs');
const path = require('path');

const argv = process.argv.slice(2);
const apply = argv.includes('--yes');
const rootArgIndex = argv.indexOf('--root');
const root = rootArgIndex >= 0 && argv[rootArgIndex + 1] ? argv[rootArgIndex + 1] : path.join(__dirname, '..', 'frontend', 'src');

function walk(dir) {
  const results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const full = path.join(dir, file);
    const stat = fs.statSync(full);
    if (stat && stat.isDirectory()) {
      results.push(...walk(full));
    } else {
      results.push(full);
    }
  }
  return results;
}

function makeStub(targetRelPath) {
  const importPath = './' + path.basename(targetRelPath);
  return `// Auto-generated re-export stub\nexport { default } from '${importPath}';\n`;
}

function run() {
  if (!fs.existsSync(root)) {
    console.error('Root path does not exist:', root);
    process.exit(1);
  }

  const all = walk(root).filter(f => f.endsWith('.ts') || f.endsWith('.tsx'));
  const planned = [];

  for (const target of all) {
    const dir = path.dirname(target);
    const base = path.basename(target, path.extname(target));
    const legacyJs = path.join(dir, base + '.js');
    const legacyJsx = path.join(dir, base + '.jsx');
    if (fs.existsSync(legacyJs)) {
      planned.push({ legacy: legacyJs, target, relLegacy: path.relative(root, legacyJs), relTarget: path.relative(root, target) });
    } else if (fs.existsSync(legacyJsx)) {
      planned.push({ legacy: legacyJsx, target, relLegacy: path.relative(root, legacyJsx), relTarget: path.relative(root, target) });
    }
  }

  if (planned.length === 0) {
    console.log('No matching legacy .js/.jsx files found for existing .ts/.tsx implementations in', root);
    return;
  }

  console.log('Found', planned.length, 'files to replace.');
  for (const p of planned) {
    console.log('-', p.relLegacy, '-> re-export', p.relTarget);
  }

  if (!apply) {
    console.log('\nDry-run mode (no files changed). Re-run with --yes to apply changes.');
    return;
  }

  for (const p of planned) {
    const bak = p.legacy + '.bak';
    if (!fs.existsSync(bak)) {
      fs.copyFileSync(p.legacy, bak);
    }
    const stub = makeStub(p.relTarget);
    fs.writeFileSync(p.legacy, stub, { encoding: 'utf8' });
    console.log('Replaced', p.relLegacy, '-> re-export', p.relTarget, '(backup:', path.basename(bak) + ')');
  }
}

run();
