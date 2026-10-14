import fs from 'node:fs';

const taxonomyPath = new URL('../taxonomy/deepmount_italian_ner_xxl_v2.labels.json', import.meta.url);
const datasetPath = new URL('../data/seed/it.deepmount_coverage.seed.jsonl', import.meta.url);

const taxonomy = JSON.parse(fs.readFileSync(taxonomyPath, 'utf8'));
const allowed = new Set(taxonomy.labels);
const covered = new Set();
const errors = [];

const lines = fs.readFileSync(datasetPath, 'utf8').split(/\r?\n/).filter(Boolean);
for (let i = 0; i < lines.length; i += 1) {
  let record;
  try {
    record = JSON.parse(lines[i]);
  } catch (err) {
    errors.push(`line ${i + 1}: invalid JSON: ${err.message}`);
    continue;
  }
  if (!record.id) errors.push(`line ${i + 1}: missing id`);
  if (record.language !== 'it') errors.push(`${record.id}: expected language=it`);
  if (!['train', 'validation', 'test'].includes(record.split)) errors.push(`${record.id}: invalid split`);
  if (typeof record.text !== 'string' || record.text.length === 0) errors.push(`${record.id}: empty text`);
  if (!Array.isArray(record.entities)) errors.push(`${record.id}: entities must be an array`);
  for (const entity of record.entities || []) {
    if (!allowed.has(entity.label)) errors.push(`${record.id}: unknown label ${entity.label}`);
    else covered.add(entity.label);
    if (!record.text.includes(entity.surface)) {
      errors.push(`${record.id}: surface not found: ${entity.surface}`);
    }
    if (!['none', 'low', 'medium', 'high', 'critical'].includes(entity.sensitivity)) {
      errors.push(`${record.id}: invalid sensitivity for ${entity.label}`);
    }
    if (!['keep', 'review', 'mask'].includes(entity.action)) {
      errors.push(`${record.id}: invalid action for ${entity.label}`);
    }
  }
}

const missing = taxonomy.labels.filter((label) => !covered.has(label));
if (missing.length > 0) errors.push(`missing label coverage: ${missing.join(', ')}`);

if (errors.length > 0) {
  console.error(errors.join('\n'));
  process.exit(1);
}

console.log(`seed dataset ok: ${lines.length} records, ${covered.size}/${taxonomy.labels.length} labels covered`);
