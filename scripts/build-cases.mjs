// Build a static case library for the browser (offline fallback + example loader).
// Reads the same bundled synthetic cases the server uses and writes public/cases.json.
import { writeFileSync } from 'node:fs';
import { cases } from '../demo/cases.js';

const out = new URL('../public/cases.json', import.meta.url);
writeFileSync(out, JSON.stringify(cases));
console.log(`Wrote ${cases.length} cases to public/cases.json`);
