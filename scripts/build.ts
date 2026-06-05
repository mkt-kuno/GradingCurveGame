import { readFileSync, writeFileSync } from 'fs';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
let html = readFileSync('index.html', 'utf8');

const placeholder = /App ver \?\.\?\.\?/;
if (!placeholder.test(html)) {
  console.error('Error: Version placeholder "App ver ?.?.?" not found in index.html');
  process.exit(1);
}

html = html.replace(placeholder, `App ver ${pkg.version}`);
writeFileSync('index.html', html);
console.log(`index.html updated with version: App ver ${pkg.version}`);
