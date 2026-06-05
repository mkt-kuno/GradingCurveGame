import { readFileSync, writeFileSync } from 'fs';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
let html = readFileSync('index.html', 'utf8');

const placeholder = /Application: version \?\.\?\.\?/;
if (!placeholder.test(html)) {
  console.error('Error: Version placeholder "Application: version ?.?.?" not found in index.html');
  process.exit(1);
}

html = html.replace(placeholder, `Application: version ${pkg.version}`);
writeFileSync('index.html', html);
console.log(`index.html updated with version: Application: version ${pkg.version}`);
