import { readFileSync, writeFileSync } from 'fs';
import { generateSW } from 'workbox-build';

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

async function buildServiceWorker() {
  console.log('Generating Workbox Service Worker...');
  const { count, size, warnings } = await generateSW({
    globDirectory: '.',
    globPatterns: [
      'index.html',
      'dist/game.js',
      'manifest.json',
      'bgm.opus',
      'icons/*.png',
    ],
    globIgnores: ['node_modules/**/*', 'src/**/*', 'scripts/**/*', 'tsconfig.json'],
    swDest: 'sw.js',
    inlineWorkboxRuntime: true,
    directoryIndex: 'index.html',
    navigateFallback: 'index.html',
    clientsClaim: true,
    skipWaiting: true,
    cleanupOutdatedCaches: true,
    sourcemap: false,
    runtimeCaching: [
      {
        urlPattern: /\.(?:png|opus|ico)$/,
        handler: 'CacheFirst',
        options: {
          cacheName: 'media-cache',
          expiration: {
            maxEntries: 50,
            maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
          },
        },
      },
    ],
  });

  if (warnings.length > 0) {
    console.warn('Workbox warnings:', warnings);
  }
  console.log(`Workbox SW generated: ${count} files precached (${Math.round(size / 1024)} KB)`);
}

buildServiceWorker().catch((err) => {
  console.error('Workbox SW build failed:', err);
  process.exit(1);
});
