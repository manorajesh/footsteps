# Footsteps

> Some paths are never walked alone.

An interactive floor installation: a camera watches people walk, and when a visitor
steps where someone once stood, that earlier path unfolds around them in light.

The authoritative design is [`FOOTSTEPS_DESIGN_COVENANT.md`](FOOTSTEPS_DESIGN_COVENANT.md).
Every spec and every change must respect its invariants (§4) and glossary (§3).

## Layout

```
Package.swift                 SwiftPM package — all core logic, testable with `swift test`
Sources/
  FootstepsCore/              F-1  Domain model, geometry, units, config, clock, random,
                                   FrameSource/PoseSource protocols, Calibration artifact
  FootstepsSimulation/        F-1  Scripted/simulated walkers, Step stream record & replay
  FootstepsMemory/            F-2  PathAssembler, Path Similarity, HistoryStore, Recency Order
  FootstepsEcho/              F-3  EchoSelector, Choreographer, Envelopes, restraint caps
  FootstepsSensing/           F-5  WalkerTracker, StepDetector, CoordinateMapper (headless)
  FootstepsCapture/           F-5  Camera/tracker implementations chosen by F-4
  FootstepsRender/            F-6  Metal FloorRenderer, ProjectionWarp, projector windows
  FootstepsCalibration/       F-7  Manual and automatic calibration
Tests/                        One test target per module (Swift Testing)
App/                          F-8  Thin Xcode app target (operator + projector windows)
Research/                     F-4  Sensing research prototypes (not part of the package)
docs/
  specs/                      Feature specs F-1 … F-8
  decisions/                  Decision records (e.g. F-4 camera/tracker choice)
```

### Module boundaries

The module graph enforces the covenant's boundaries (§6.4) at compile time:

- `FootstepsCore` depends on nothing.
- `FootstepsRender` depends only on Core — the renderer cannot see Paths or History.
- `FootstepsSensing` depends only on Core's `PoseSource` protocol; hardware specifics
  live in `FootstepsCapture`, which only the app imports (INV-11).
- `FootstepsEcho` depends on Memory; nothing in the headless core imports AppKit or Metal.

## Build & test

```sh
swift build
swift test
```

Requires macOS 14+ on Apple Silicon and Swift 6.

## Build order (§13.3)

1. F-1 Foundation & Test Harness, with F-4 Sensing Research alongside
2. F-2 → F-3 → F-6 → F-7 on simulation input
3. F-5 once F-4 decides
4. F-8 last
