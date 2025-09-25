generate-reexports (TypeScript)

Purpose
- Conservative re-export generator to replace legacy `.js`/`.jsx` files with small stubs that re-export the converted `.ts`/`.tsx` default export.

Usage
- Dry-run (safe, shows planned replacements):
  node -r ts-node/register scripts/generate-reexports.ts --root frontend --dry-run

- Apply changes (script creates .bak copies before overwriting):
  node -r ts-node/register scripts/generate-reexports.ts --root frontend --yes

Notes
- The TypeScript version requires either ts-node or compiling to JS. If you prefer the original JS script, `scripts/generate-reexports.js` remains available.
- Suggested package.json script (frontend or repo root):
  "scripts": {
    "generate-reexports": "node -r ts-node/register scripts/generate-reexports.ts --root frontend"
  }

Safety
- This tool is conservative: it only replaces legacy `.js`/`.jsx` files if a same-named `.ts` or `.tsx` implementation exists. It creates `.bak` backups for each overwritten legacy file.
