import { readFileSync, writeFileSync } from 'fs';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
let html = readFileSync('index.html', 'utf8');

const versionPattern = /Application: version (\?\.\?\.\?|\d+\.\d+\.\d+)/;
if (!versionPattern.test(html)) {
  console.error('Error: Version pattern "Application: version ?.?.?" not found in index.html');
  process.exit(1);
}

html = html.replace(versionPattern, `Application: version ${pkg.version}`);
writeFileSync('index.html', html);
console.log(`index.html updated with version: Application: version ${pkg.version}`);
