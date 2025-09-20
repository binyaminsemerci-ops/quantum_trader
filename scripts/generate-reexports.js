#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..', 'frontend', 'src');
const backupDir = path.resolve(__dirname, '..', 'frontend', 'backups');

if (!fs.existsSync(backupDir)) fs.mkdirSync(backupDir, { recursive: true });

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      walk(full);
    } else if (e.isFile() && full.endsWith('.jsx')) {
      const base = full.slice(0, -4); // drop .jsx
      const tsx = `${base}.tsx`;
      const ts = `${base}.ts`;
      if (fs.existsSync(tsx) || fs.existsSync(ts)) {
        const rel = path.relative(root, full);
        const backupPath = path.join(backupDir, rel.replace(/[\\/]/g, '--'));
        fs.mkdirSync(path.dirname(backupPath), { recursive: true });
        fs.copyFileSync(full, backupPath);
        const exportTarget = `./${path.basename(base)}.${fs.existsSync(tsx) ? 'tsx' : 'ts'}`;
        const stub = `// Re-export canonical implementation\nexport { default } from '${exportTarget}';\n`;
        fs.writeFileSync(full, stub, 'utf8');
        console.log(`Replaced ${rel} -> ${exportTarget} (backup: ${path.relative(process.cwd(), backupPath)})`);
      }
    }
  }
}

walk(root);
console.log('Done.');
