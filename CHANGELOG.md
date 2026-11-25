# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-25

First public release of real-time 3D fluid simulation in WebGPU compute shaders.

### Added
- PIC/FLIP on a MAC grid: fixed-point atomic P2G, Jacobi and red-black SOR pressure projection, RK2 advection
- Position Based Fluids solver with counting-sort neighbour search
- Screen-space fluid surface: bilateral depth smoothing, Fresnel, thickness absorption
- Measured in the browser: 129,600 particles at 53 fps, 4.5 ms per substep (GPU timestamps)

[0.1.0]: https://github.com/coopersimpson101-debug/flipgpu/releases/tag/v0.1.0
