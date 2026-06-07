# AGENTS.md

## Project Overview

粒度分布ゲーム（Grading Curve Game）- 土質工学メロンゲーム。DEM物理シミュレーション + WebGPU/Canvas2Dレンダリングによるブラウザゲーム。

## Tech Stack

- Runtime: Bun 1.3
- Language: TypeScript (strict, ES2020, target: browser)
- Rendering: WebGPU (fallback: Canvas2D)
- No external dependencies

## Commands

- `bun run build` - Build for production (`bun build src/game.ts --outfile dist/game.js --target browser` + version stamp)
- `bun run serve` - Start local dev server on port 3000
- `bun run dev` - Build & serve

## Architecture

- `src/game.ts` - Main game logic (DEM physics, game state, input handling, Canvas2D fallback rendering)
- `src/constants.ts` - All game constants, particle types, levels, physics parameters
- `src/spatial.ts` - Spatial hash grid for collision detection optimization
- `src/renderer-webgpu.ts` - WebGPU rendering pipeline
- `scripts/build.ts` - Post-build: stamps version from package.json into index.html
- `server.ts` - Simple Bun static file server (port 3000)
- `index.html` - Game UI (single page, inline CSS)
- `sw.js` - Service worker for PWA
- `manifest.json` - PWA manifest

## CI/CD

- GitHub Actions: `.github/workflows/deploy.yml`
- Triggers on push to `master` branch
- Deploys to GitHub Pages via `bun run build` + artifact upload

## DEM Contact Models

- **Hertz** contact model uses Hertz-Mindlin formulation (nonlinear normal + Mindlin tangential)
- **Hooke** contact model uses linear (Hooke-Mindlin) formulation (linear normal + Mindlin tangential)
- Both models share: Coulomb friction cap, velocity-dependent tangential damping, rolling resistance
- Rolling resistance is relative-spin-based (Hertz-Mindlin compliant): opposes `ω_B - ω_A` at contact
- Wall contacts: equivalent since wall has `ω=0` (absolute spin = relative spin)
- **Future**: Incremental tangential spring (history tracking) for accurate static-kinetic friction transition

## Conventions

- Language: Code comments in English, UI text in Japanese/English bilingual
- Version: Managed in `package.json`, stamped into `index.html` by build script
- No linter/formatter configured - maintain existing code style
- No test framework configured