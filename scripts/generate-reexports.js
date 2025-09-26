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

function makeStub(tsxRelPath, jsxRelPath) {
  const importPath = './' + path.basename(tsxRelPath);
  return `// Auto-generated re-export stub\nexport { default } from '${importPath}';\n`;
}

function run() {
  if (!fs.existsSync(root)) {
    console.error('Root path does not exist:', root);
    process.exit(1);
  }

  const all = walk(root).filter(f => f.endsWith('.tsx'));
  const planned = [];

  for (const tsx of all) {
    const dir = path.dirname(tsx);
    const base = path.basename(tsx, '.tsx');
    const jsx = path.join(dir, base + '.jsx');
    if (fs.existsSync(jsx)) {
      const relTsx = path.relative(root, tsx);
      const relJsx = path.relative(root, jsx);
      planned.push({ jsx, tsx, relJsx, relTsx });
    }
  }

  if (planned.length === 0) {
    console.log('No matching .jsx files found for existing .tsx files in', root);
    return;
  }

  console.log('Found', planned.length, 'files to replace.');
  for (const p of planned) {
    console.log('-', p.relJsx, '-> re-export', p.relTsx);
  }

  if (!apply) {
    console.log('\nDry-run mode (no files changed). Re-run with --yes to apply changes.');
    return;
  }

  for (const p of planned) {
    const bak = p.jsx + '.bak';
    if (!fs.existsSync(bak)) {
      fs.copyFileSync(p.jsx, bak);
    }
    const stub = makeStub(p.relTsx, p.relJsx);
    fs.writeFileSync(p.jsx, stub, { encoding: 'utf8' });
    console.log('Replaced', p.relJsx, '(backup created at', path.basename(bak) + ')');
  }
}

run();
