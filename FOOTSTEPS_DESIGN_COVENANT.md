# Footsteps — Design Covenant

**Version:** 0.3
**Status:** Authoritative input for feature-spec generation
**Audience:** The spec-writing agent, and every human or agent who implements a spec derived from it

---

## 0. How to Use This Document

This covenant is the fixed ground that every feature spec stands on. It defines what Footsteps *is*, the vocabulary everyone must use, the invariants no spec may break, the architecture specs must fit into, and the defaults specs should assume.

Rules for the spec-writing agent:

1. **Every spec must cite the covenant sections it depends on** (e.g. "Implements §6.2, respects INV-3, INV-7").
2. **No spec may contradict an invariant (§4).** If a spec needs to, it must stop and raise a covenant amendment instead of working around it.
3. **Use the glossary terms (§3) exactly.** Do not invent synonyms ("trail", "track", "trace") for defined terms.
4. **Every tunable number comes from the parameter registry (§9).** Specs may add parameters, but must register them there with a default, unit, and range.
5. **Items in §12 (Open Questions) carry a proposed default.** Specs should implement the default and keep it switchable; they must not silently pick a different answer.
6. **Every spec must be testable without the physical installation** (see §10), using synthetic or recorded input.
7. **Decisions marked "Deferred to research" (§5) are not yet made.** No spec may assume a particular answer to them until the named research spec has produced a decision record and the covenant has been amended.

---

## 1. Intent

> Some paths are never walked alone.

Footsteps is an interactive floor installation. A camera watches people walk; a projector mounted at an angle lights up the floor. When a visitor steps where someone else once stood, that earlier person's footprint appears beneath the visitor's foot, and the rest of that earlier path unfolds around them — ahead and behind — one glowing footprint at a time, each fading in and fading out. For a moment, two paths become one.

The piece is about **quiet coincidence**, not spectacle. The system's job is to notice overlap and reveal it gracefully.

### 1.1 Experience Principles

These are ranked. When two conflict, the higher one wins.

1. **Presence before precision.** A footprint that appears promptly and roughly in place beats one that appears late and exactly in place. Latency is felt; an inch or two of error is not.
2. **Restraint.** The floor should never be busy. Few footprints, clear light, generous fades. Absence is part of the work.
3. **The echo belongs to the visitor.** Revealed footprints are triggered by and anchored to a live person's step. Nothing appears "on its own".
4. **Continuity of memory.** History accumulates across sessions and restarts. The installation on day 30 remembers day 1 (subject to the replacement and eviction rules).
5. **Anonymity.** The installation remembers where people walked, never who they were.

---

## 2. Scope

### 2.1 In Scope

- Live camera capture and body/foot tracking of one or more walkers
- Step detection and path assembly in a calibrated floor coordinate system
- A persistent history of paths, kept in recency order, with a similarity-based replacement rule
- Echo selection (matching a live step to a historical path) and sequential, faded playback
- Rendering footprints and projecting them onto the floor with perspective correction
- Automatic and manual calibration of the camera–floor–projector relationship
- Operator tooling: calibration mode, debug overlays, parameter tuning, simulation input
- Native and performant macOS execution

### 2.2 Out of Scope (v1)

- Identifying or re-identifying individuals across visits
- Storing or transmitting any camera imagery
- Networked or multi-site operation
- Sound
- Multiple projectors, and more than one camera rig (a single stereo pair counts as one rig; see §5)
- Non-flat floors (stairs, slopes)

---

## 3. Glossary

These terms are normative.

| Term | Definition |
|---|---|
| **Walker** | A live person currently tracked by the system. Has a transient `WalkerID` valid only while tracked. |
| **Step** | A single detected foot contact: a floor-space position, a timestamp, which foot (left/right/unknown), and a heading. The atomic unit of the system. |
| **Footprint** | The *visual* representation of a Step on the floor. Steps are data; footprints are light. |
| **Path** | An ordered sequence of Steps produced by one Walker between entering and leaving tracking. |
| **Live Path** | The Path currently being built for an active Walker. Not yet in History. |
| **History** | The persistent collection of completed Paths, held in Recency Order. |
| **Recency Order** | History's ordering, newest first, maintained as an LRU list: a Path moves to the front when it is committed or when it replaces an older Path, and eviction removes from the back. |
| **Match** | The event of a live Step landing within the match radius of a Step in some historical Path. |
| **Anchor Step** | The historical Step that was matched. It is the first footprint shown in an Echo. |
| **Echo** | A playback of a window of a historical Path around its Anchor Step, triggered by a Match. |
| **Echo Window** | The Steps of the historical Path from `anchor − N_behind` to `anchor + N_ahead`. |
| **Envelope** | The fade-in / hold / fade-out opacity curve applied to each Footprint. |
| **Path Similarity** | A distance measure between two Paths, used by the replacement rule. |
| **Camera Space** | Pixel coordinates in the camera image (or, for a stereo rig, the rectified reference image plus depth). |
| **Floor Space** | Physical coordinates on the floor plane, in **inches**, with origin and axes defined at calibration. The canonical space of the system. Operator-facing displays show feet and inches (e.g. `12′ 4″`). |
| **Projector Space** | Pixel coordinates in the projector's output frame. |
| **Calibration** | The pair of homographies `H_cam→floor` and `H_floor→proj`, plus metadata. |

---

## 4. Invariants

These hold everywhere, always. Violating one is a bug, regardless of what any spec says.

- **INV-1 — Floor space is canonical.** All Steps, Paths, and History are stored in Floor Space, in inches (floating point). Feet-and-inches formatting exists only in the operator UI. Camera and projector coordinates exist only at the edges of the pipeline.
- **INV-2 — Recalibration never mutates History.** Changing calibration changes how history is *seen and drawn*, never what it *is*.
- **INV-3 — No imagery is persisted.** Camera frames, depth maps, and pose skeletons never touch disk or network. Only Steps (position, time offset, foot, heading) are stored. Debug frame recording, if ever added, must be explicit, opt-in, and off by default.
- **INV-4 — A Walker never echoes itself.** A live Step may not match its own Live Path, and a Path committed moments ago may not be matched by the same Walker (see `self_match_cooldown`).
- **INV-5 — Echo selection is deterministic.** Given the same History, Recency Order, and live Step, the same Echo is chosen. Any randomness a future spec introduces must go through an injected, seedable random source.
- **INV-6 — Rendering never waits on sensing.** The render loop runs at display rate independently of the capture/tracking pipeline. A stalled or empty tracker produces a calm floor, never a frozen or glitching one.
- **INV-7 — Time is monotonic and injectable.** All timing uses an injected monotonic clock, never wall-clock time, so playback, fades, and recency can be tested deterministically.
- **INV-8 — History writes are atomic.** A crash or power loss mid-write leaves History (including Recency Order) in its previous valid state, never corrupted.
- **INV-9 — Calibration is a single artifact.** Automatic and manual calibration produce the same data structure in the same file. The rest of the system cannot tell which method made it.
- **INV-10 — Only committed Paths are matchable.** Live Paths of other current Walkers are not in History and are not matched (see §12, Q6).
- **INV-11 — Sensing hardware is replaceable.** Everything downstream of `PoseSource` depends only on the `FrameSource` / `PoseSource` protocols, never on a specific camera, tracker, or ML model.

---

## 5. Platform and Technology Decisions

| Concern | Decision | Status |
|---|---|---|
| OS | macOS 14 (Sonoma) or later, Apple Silicon | Decided |
| Language | Swift (5.9+ / Swift 6 language mode where practical) | Decided |
| Build | Swift Package Manager, with a thin Xcode app target, so core logic is testable via `swift test` without an app bundle | Decided |
| Rendering | Metal via `MTKView` | Decided |
| Windowing | AppKit, borderless fullscreen window on the projector display; separate operator window on the main display | Decided |
| Persistence | JSON files in `~/Library/Application Support/Footsteps/` (calibration, config, history); revisit if History exceeds ~10 MB | Decided |
| Tests | XCTest / Swift Testing on the core package | Decided |
| Units | Inches internally; feet and inches in the operator UI | Decided |
| **Camera** | **Deferred to research (F-4).** Candidates to evaluate: standard RGB webcam; IR camera with IR illumination; stereo RGB webcam pair; stereo IR camera (depth). | Deferred |
| **Body / foot tracker** | **Deferred to research (F-4).** Candidates are assessed together with the camera, because the right tracker depends on what the camera sees. Examples: Apple Vision body pose (2D and 3D); CoreML-converted pose models; depth-based foot segmentation from a stereo rig; overhead IR blob/foot tracking. | Deferred |

### 5.1 Sensing Research Requirements

The F-4 research spec must produce a **decision record** that amends this section. It must evaluate every candidate camera × tracker pairing against:

- Foot-contact → Step latency (must support the §9.1 target)
- Floor-space position accuracy of foot contacts, in inches
- Robustness under the installation's own projected light
- Occlusion and multiple simultaneous Walkers
- Mounting geometry (height, angle, field of view covering the floor region)
- Whether the tracker supplies stable identity or WalkerTracker must
- Native macOS support, driver/SDK maturity, and cost
- Whether depth (stereo) meaningfully improves contact detection over 2D

Until that decision lands, specs downstream of `PoseSource` build against the protocols and simulation input (§10) only.

---

## 6. Architecture

### 6.1 Pipeline

The system is two loops joined by a shared, thread-safe state. The **sensing loop** turns camera frames into Steps and Echoes. The **render loop** turns active Echoes into light on the floor.

**Sensing loop** (camera rate, background):

| # | Stage | Takes | Produces |
|---|---|---|---|
| 1 | FrameSource | camera hardware or recording | timestamped frames (and depth, for stereo) |
| 2 | PoseSource | frames | per-person foot/ankle joints + confidence, Camera Space |
| 3 | WalkerTracker | joints | joints tagged with stable `WalkerID`s |
| 4 | StepDetector | tagged joints over time | Steps, Camera Space |
| 5 | CoordinateMapper | Steps + `H_cam→floor` | Steps, Floor Space |
| 6a | PathAssembler | Steps | completed Paths → HistoryStore |
| 6b | EchoSelector | Steps + HistoryStore | Echoes → Choreographer |

**Render loop** (display rate, render thread):

| # | Stage | Takes | Produces |
|---|---|---|---|
| 7 | Choreographer | Echoes + clock | visible Footprints with current opacity |
| 8 | FloorRenderer | visible Footprints | texture in Floor Space |
| 9 | ProjectionWarp | texture + `H_floor→proj` | projector frame |

Every live Step goes to both 6a and 6b. Stage 7 is the only point where the two loops meet.

### 6.2 Module Responsibilities

Each module is a candidate for one or more feature specs.

- **FrameSource** — Delivers timestamped frames, and depth when the rig provides it. Implementations: whichever camera(s) F-4 selects, recorded input, and none (for pure simulation).
- **PoseSource** — Turns frames into per-person foot joints (at minimum left/right ankle or foot position, with confidence) in Camera Space. Implementation chosen by F-4.
- **WalkerTracker** — Assigns stable `WalkerID`s across frames if the chosen tracker does not. Handles entry, brief occlusion, and exit (`walker_lost_timeout`).
- **StepDetector** — Emits a Step when a foot becomes stationary (speed below `contact_speed_threshold` for at least `contact_min_duration`) after moving. Estimates heading from recent motion. Suppresses duplicate contacts for the same foot.
- **CoordinateMapper** — Applies `H_cam→floor`. Rejects Steps that fall outside the calibrated floor region.
- **PathAssembler** — Maintains one Live Path per Walker. On Walker exit, finalizes the Path and hands it to HistoryStore if it meets `min_path_steps`.
- **HistoryStore** — Persistent storage, Recency Order (LRU), a spatial index for fast "which Steps are near (x, y)?" queries, and the replacement rule (§7.2).
- **EchoSelector** — On each live Step, finds Matches and chooses one historical Path (§7.1). Enforces INV-4 and echo concurrency limits.
- **Choreographer** — Turns an Echo into a timeline of Footprints with Envelopes (§7.3). Owns all "what is visible right now" state.
- **FloorRenderer** — Draws visible Footprints into an offscreen texture in Floor Space units. Knows nothing about projectors.
- **ProjectionWarp** — Warps the floor texture into Projector Space using `H_floor→proj`. The *only* place perspective correction happens.
- **Calibration** — Produces and persists the Calibration artifact, via automatic or manual mode (§8).
- **Operator UI** — Calibration interface, live debug overlays, parameter tuning, history inspection/reset.

### 6.3 Threading Model

- The sensing loop (stages 1–5) runs on a dedicated serial queue.
- HistoryStore and EchoSelector may run on the sensing queue or their own actor; they must never block rendering (INV-6).
- Choreographer state is read by the renderer each frame; updates cross threads via a lock-free snapshot or actor hop, never via blocking waits on the render thread.
- History disk writes happen off both the sensing and render paths.

### 6.4 Boundaries Specs Must Respect

- Nothing downstream of CoordinateMapper touches Camera Space.
- Nothing except ProjectionWarp and Calibration touches Projector Space.
- The Choreographer does not know about pixels; the renderer does not know about Paths or History.
- Nothing downstream of PoseSource knows which camera or tracker is in use (INV-11).

---

## 7. Core Behaviours

### 7.1 Echo Selection (Requirement 1)

When a live Step `s` is detected for Walker `w`:

1. Query History for all Steps within `match_radius` of `s.position`.
2. Discard Steps belonging to Paths excluded by INV-4.
3. Discard Steps whose heading differs from `s.heading` by more than `match_heading_tolerance` (on by default; see §12, Q3).
4. Group remaining candidates by Path. For each Path, keep the candidate Step nearest to `s` as that Path's Anchor candidate.
5. If no candidates remain, do nothing.
6. If one or more remain, **choose the Path that appears earliest in Recency Order (the most recent one)**. Its nearest candidate becomes the Anchor Step. Because Recency Order is total, there are no ties.
7. Build the Echo Window: Steps from `anchor − echo_steps_behind` to `anchor + echo_steps_ahead`, clamped to the Path's bounds. "Behind" and "ahead" refer to the historical walker's own order of steps.
8. Hand the Echo to the Choreographer, subject to concurrency rules (§7.4).

Whether playing an Echo also moves its Path to the front of Recency Order is an open question (§12, Q9); by default it does not.

### 7.2 History Replacement and Recency (Requirement 2)

When PathAssembler finalizes a Path `p`:

1. Reject `p` if it has fewer than `min_path_steps` Steps.
2. Compute Path Similarity between `p` and each Path in History whose bounding box overlaps `p`'s (spatial pre-filter).
3. **Path Similarity** is the discrete Fréchet distance between the two Paths after both are resampled to uniform arc-length spacing (`similarity_resample_spacing`). It is direction-sensitive by default (a Path walked in reverse is not similar; see §12, Q4).
4. If one or more Paths have similarity distance ≤ `similarity_threshold`, **remove the single most similar Path and insert `p` at the front of Recency Order** (see §12, Q5).
5. Otherwise, insert `p` at the front of Recency Order.
6. If History exceeds `history_capacity`, evict from the back of Recency Order (least recently used).
7. Persist atomically (INV-8).

The effect: History converges toward one representative — the most recent — per distinct route, novel routes accumulate, and at any spot the newest walker's path is the one that answers.

### 7.3 Sequential Playback and Fades (Requirement 3)

An Echo is played as a timeline of Footprints:

- **The Anchor Step's footprint appears first, at t = 0**, directly beneath the visitor's foot. This is the moment "paths become one".
- The remaining Steps unfold **outward from the anchor in both directions**, sequentially: the k-th step ahead and the k-th step behind appear together at the same time offset. (Alternative mode "chronological" is described in §12, Q1.)
- The interval between consecutive footprints is the historical walker's **original recorded cadence**, clamped to [`cadence_min`, `cadence_max`] and scaled by `playback_speed`.
- Each Footprint follows an Envelope: fade in over `fade_in`, hold for `hold`, fade out over `fade_out`. Default easing is smooth (e.g. smoothstep or ease-in-out), never linear pops.
- Footprints are drawn at their Step's position and heading, as left or right footprint shapes when foot side is known, at full brightness (`footprint_brightness`).
- An Echo ends when its last footprint has fully faded out.

### 7.4 Concurrency and Restraint

- Each Walker may have at most `max_echoes_per_walker` active Echoes (default 1). While an Echo is active for a Walker, further Matches for that Walker are ignored unless `allow_echo_interrupt` is true.
- A global cap `max_active_echoes` limits total floor activity across all Walkers.
- A Walker's own live Steps are **not** projected by default (`show_live_footprints` = false): the visitor's real feet are the present; the light is the past (see §12, Q2).

---

## 8. Projection Mapping and Calibration (Requirement 4)

### 8.1 Model

The floor is a plane, so both mappings are exact planar homographies:

- `H_cam→floor` — maps Camera Space to Floor Space (inches).
- `H_floor→proj` — maps Floor Space to Projector Space.

Rendering happens in Floor Space; ProjectionWarp applies `H_floor→proj` once, as the final step. An angled projector is handled entirely by this single matrix, and footprints keep their true physical size and shape.

If F-4 selects a stereo rig, `H_cam→floor` still applies to the reference camera's foot positions; depth is used by PoseSource/StepDetector to detect contact, not to replace the floor mapping.

### 8.2 The Calibration Artifact

A single JSON file (`calibration.json`) containing:

- `H_cam_to_floor` and `H_floor_to_proj` (3×3, row-major)
- Floor region: the calibrated active rectangle, in inches
- Camera and projector resolutions the calibration was made at
- Method (`auto` | `manual`), timestamp, and reprojection error (inches) if known

If the camera or projector resolution at runtime differs from the stored one, the app must warn the operator rather than silently misalign.

### 8.3 Automatic Calibration

1. The operator places **four physical reference markers** at the corners of the active floor rectangle (e.g. tape crosses).
2. **Calibration interface — no typed values.** In the operator window, the operator click-selects each marker in the live camera preview. When the system can detect the markers itself, it pre-selects them and the operator confirms or corrects by clicking. The rectangle's width and height are set with click controls — common preset sizes plus feet/inch steppers — never by typing into a field. This yields `H_cam→floor`.
3. The projector displays a sequence of known patterns (e.g. a dot grid or high-contrast markers, shown one at a time against black); the camera detects them → correspondences between Projector Space and Camera Space → `H_proj→cam`.
4. Compose: `H_floor→proj = (H_cam→floor · H_proj→cam)⁻¹`.
5. Report reprojection error in inches; refuse to save above `calibration_max_error` unless the operator overrides.

Camera exposure and white balance (or IR gain, per F-4) are locked during and after calibration so projected light does not cause exposure drift.

### 8.4 Manual Calibration

Available both as an in-app mode and as a directly editable file:

- **In-app corner pin.** A calibration mode (toggled from the operator window or by a keyboard shortcut) projects the floor rectangle's outline and a grid. The operator drags its four corners in the operator window (and/or selects a corner and nudges with arrow keys: 1 px, Shift for 10 px) until the projected corners land on the physical reference markers. This directly defines `H_floor→proj`. A second view lets the operator click the markers in the camera preview to define `H_cam→floor`, using the same click controls for dimensions as §8.3.
- **Live preview.** Changes apply to the projection immediately.
- **Save / revert.** Changes are written to `calibration.json` only on save; revert restores the last saved state.
- **File editing.** `calibration.json` is human-editable. The app reloads it on change (or on a reload command) and validates it (matrices invertible, non-degenerate).
- Manual mode may start from an automatic result as a fine-tuning pass.

Per INV-9, both modes produce the identical artifact.

---

## 9. Parameter Registry

All tunables live in `config.json`, are editable in the operator UI, and have these defaults. Lengths are stored in inches; the UI shows feet and inches where that reads better. Specs adding parameters must add rows here.

| Parameter | Default | Unit | Range | Notes |
|---|---|---|---|---|
| `match_radius` | 8 | in | 2–24 | "Same coordinates" tolerance |
| `match_heading_tolerance` | 60 | degrees | 0–180 | On by default; 180 disables it. See §12, Q3 |
| `echo_steps_behind` | 5 | steps | 0–30 | |
| `echo_steps_ahead` | 8 | steps | 0–30 | |
| `playback_speed` | 1.0 | × | 0.25–3.0 | Multiplies original cadence |
| `cadence_min` | 0.30 | s | 0.1–1.0 | Clamp on inter-step interval |
| `cadence_max` | 0.90 | s | 0.3–3.0 | |
| `fade_in` | 0.35 | s | 0–2 | |
| `hold` | 1.20 | s | 0–5 | |
| `fade_out` | 1.50 | s | 0–5 | |
| `max_echoes_per_walker` | 1 | count | 1–5 | |
| `allow_echo_interrupt` | false | bool | | |
| `max_active_echoes` | 6 | count | 1–20 | Global restraint |
| `show_live_footprints` | false | bool | | See §12, Q2 |
| `echo_refreshes_recency` | false | bool | | See §12, Q9 |
| `self_match_cooldown` | 30 | s | 0–600 | Walker can't match a Path it just committed |
| `min_path_steps` | 4 | steps | 2–20 | Shorter paths are noise |
| `similarity_resample_spacing` | 10 | in | 2–40 | |
| `similarity_threshold` | 14 | in | 2–80 | Fréchet distance |
| `similarity_direction_sensitive` | true | bool | | See §12, Q4 |
| `history_capacity` | 5000 | paths | 100–100000 | LRU eviction from back of Recency Order |
| `contact_speed_threshold` | 6 | in/s | 1–20 | Measured in Floor Space |
| `contact_min_duration` | 0.12 | s | 0.03–0.5 | |
| `walker_lost_timeout` | 1.5 | s | 0.2–10 | Ends the Live Path |
| `pose_confidence_min` | 0.3 | 0–1 | 0–1 | Foot joints below this are ignored; may be redefined by F-4 |
| `calibration_max_error` | 1 | in | 0.25–8 | Auto-calibration acceptance |
| `footprint_length` | 11 | in | 4–20 | Visual size |
| `footprint_brightness` | 1.0 | 0–1 | 0–1 | Peak opacity; max by default |

### 9.1 Performance Targets

| Metric | Target |
|---|---|
| Foot contact → Anchor footprint visible | ≤ 150 ms (p95) |
| Render frame rate | Display rate (60 Hz typical), no dropped frames under `max_active_echoes` |
| Match query (incl. recency lookup) | ≤ 5 ms at `history_capacity` |
| History commit (incl. replacement and eviction) | ≤ 50 ms, off the render path |
| Sensing pipeline | ≥ 25 fps sustained |
| Idle CPU / GPU | Low enough to run unattended all day without thermal throttling on the target Mac |

---

## 10. Testability Covenant

Footsteps will spend most of its development life away from its floor, and its camera and tracker are not yet chosen. Therefore:

- **Simulation input.** A `SimulatedPoseSource` / `ScriptedStepSource` must exist that emits Walkers and Steps from scripted or procedurally generated walks (straight lines, curves, crossings, loiterers, two walkers side by side).
- **Recorded input.** Step streams (not video or depth, per INV-3) can be recorded to and replayed from JSON for regression tests.
- **Headless core.** Everything from StepDetector through Choreographer runs and is tested without a camera, GPU, or display.
- **Deterministic tests.** Tests inject the clock (INV-7); tests assert exact timelines, exact Recency Order, and exact Echo choices (INV-5).
- **Projector simulator.** A windowed preview renders the projector output alongside a top-down floor-space view, so mapping can be checked on a laptop.
- **Every spec's acceptance criteria must be expressible as automated tests** where the behaviour is not purely visual; purely visual behaviour must name the debug view used to verify it.

---

## 11. Operator Experience

- The app launches directly into running mode using the last saved calibration and config, so the installation survives a power cycle unattended.
- The operator window shows: camera preview with detected feet and Walker IDs, top-down floor view (with a feet/inch grid) showing History density and live Steps, active Echoes, and key metrics (fps, latency, History size).
- Calibration is done by clicking and dragging, not typing (§8.3, §8.4).
- Debug overlays can optionally be shown on the projector during setup but are never on by default.
- History can be inspected, exported, and cleared (with confirmation).
- If the camera disconnects or calibration is missing, the projector shows black, not errors; the operator window shows the problem.

---

## 12. Open Questions (with Proposed Defaults)

Specs implement the default and keep the alternative switchable where noted.

- **Q1 — Playback order.** Default: outward from the anchor in both directions. Alternative "chronological": show the anchor, then the ahead-steps sequentially, with the behind-steps fading in as a trail. Needs an artistic decision after the first on-floor test.
- **Q2 — Show the visitor's own steps?** Default: no. Alternative: faint live footprints to help visitors understand the interaction. Decide during playtesting.
- **Q3 — Heading-aware matching?** Default: **on** (`match_heading_tolerance` = 60°): paths only merge when walked roughly the same way, which is more "paths become one" but rarer. Setting 180° lets any direction match.
- **Q4 — Is a reversed path similar?** Default: no (direction-sensitive), because playback is directional.
- **Q5 — Replace one or all similar paths?** Default: the single most similar. Alternative: replace all within threshold, which keeps History leaner in high-traffic areas.
- **Q6 — Can two simultaneous visitors echo each other?** Default: no (INV-10); only committed Paths match. Revisiting this would require amending INV-10.
- **Q7 — Seeding.** Does the installation ship with an empty History, or pre-seeded paths so the first visitor experiences something? Default: empty, with an operator-importable seed file.
- **Q8 — Camera and tracker.** No default; decided by the F-4 research spec (§5.1). Candidates: RGB webcam, IR camera, stereo webcam, stereo IR camera, each paired with the trackers that suit it.
- **Q9 — Does an Echo count as "use" for recency?** Default: no (`echo_refreshes_recency` = false), so only committing or replacing a Path refreshes it. If true, a Path that is echoed moves to the front, which makes popular paths self-reinforcing: they keep answering at that spot and are never evicted, while others at the same spot fade away. Decide during playtesting.

---

## 13. Feature Breakdown

Footsteps is built as **eight features**. Each one becomes one spec. Every spec covers a complete, testable capability rather than a single class. A large spec may be split into internal milestones, but it keeps its feature ID, and other specs depend on the feature as a whole.

### 13.1 Overview

| ID | Feature | What it delivers | Depends on |
|---|---|---|---|
| F-1 | Foundation & Test Harness | The shared vocabulary, settings, clock, and fake input that everything else is built and tested on | — |
| F-2 | Path Memory | Turning a walker's Steps into Paths and keeping History (similarity, replacement, recency, persistence) | F-1 |
| F-3 | Echoes | Deciding which past Path answers a live Step, and choreographing its footprints over time | F-1, F-2 |
| F-4 | Sensing Research | A decision on camera and tracker, recorded as a covenant amendment | — |
| F-5 | Sensing Pipeline | Live camera → Walkers → Steps in Floor Space | F-1, F-4, F-6 (calibration artifact) |
| F-6 | Rendering & Projection | Drawing footprints to scale and warping them onto the angled floor | F-1, F-3 |
| F-7 | Calibration Tools | Automatic and manual ways to produce the Calibration artifact | F-6 |
| F-8 | Operator App & Unattended Running | The operator window, debug views, history management, and power-cycle-safe operation | F-2, F-3, F-5, F-6, F-7 |

### 13.2 Feature Scope

**F-1 — Foundation & Test Harness**
- Domain model: Step, Path, Walker, Echo, in Floor Space inches; geometry utilities (distance, heading, resampling, bounding boxes); feet-and-inches formatting for the UI
- Config: `config.json` loading, validation against §9 ranges, live reload
- Injectable monotonic clock (INV-7), plus a random source should one ever be needed (INV-5)
- `ScriptedStepSource` / `SimulatedPoseSource`: scripted and procedural walks (straight lines, curves, crossings, loiterers, side-by-side walkers)
- Recording and replay of Step streams as JSON (§10)
- *Done when:* a scripted walk can be generated, recorded, replayed, and asserted on in a headless test with a fake clock.

**F-2 — Path Memory**
- PathAssembler: one Live Path per Walker; finalize on exit; `min_path_steps` rejection
- Path Similarity: arc-length resampling and direction-sensitive discrete Fréchet distance (§7.2)
- Replacement rule, Recency Order (LRU), and eviction at `history_capacity`
- Spatial index for "Steps near (x, y)" queries
- HistoryStore persistence with atomic writes (INV-8); import/export of seed files (Q7)
- *Done when:* simulated walks produce the expected History contents and Recency Order across restarts, and performance targets for commit and query in §9.1 are met at capacity.

**F-3 — Echoes**
- EchoSelector: match radius, heading filter, self-match exclusion (INV-4), most-recent choice, Echo Window (§7.1)
- Choreographer: outward-from-anchor sequencing, cadence clamping, `playback_speed`, fade Envelopes (§7.3)
- Concurrency and restraint: per-walker and global caps, interrupt rule (§7.4)
- Switchable alternatives from Q1, Q2, Q9
- *Done when:* for a given History and live Step stream, the exact set of visible Footprints and their opacities at any clock time is asserted in headless tests.

**F-4 — Sensing Research**
- Evaluate every candidate camera × tracker pairing against §5.1
- Prototype enough of each viable pairing to measure latency, contact accuracy, and robustness under projected light
- *Done when:* a decision record is written and §5, Q8, and `pose_confidence_min` are amended accordingly.

**F-5 — Sensing Pipeline**
- FrameSource for the selected camera, with exposure/gain locking
- PoseSource for the selected tracker
- WalkerTracker (if the tracker lacks identity): entry, occlusion, `walker_lost_timeout`
- StepDetector: contact detection, heading estimate, duplicate suppression
- CoordinateMapper: `H_cam→floor`, reject Steps outside the floor region
- *Done when:* people walking on the real floor produce Steps whose Floor Space positions and timing meet §9.1, and downstream features run unchanged on live input instead of simulation.

**F-6 — Rendering & Projection**
- Calibration artifact: `calibration.json` schema, validation, loading, resolution-mismatch warning (§8.2, INV-9)
- FloorRenderer: Metal footprint drawing in Floor Space, left/right shapes, heading, brightness
- ProjectionWarp: `H_floor→proj` applied once, as the final step
- Projector window (borderless fullscreen on the projector display) and projector simulator preview (§10)
- Render loop independent of sensing (INV-6)
- *Done when:* Echoes from F-3 render at display rate with no dropped frames at `max_active_echoes`, at correct physical size under a hand-written calibration, in both the projector and simulator views.

**F-7 — Calibration Tools**
- Manual: in-app corner pin with drag and arrow-key nudge, camera-preview marker clicking, live preview, save/revert, file reload (§8.4)
- Automatic: click-select marker interface with preset/stepper dimensions, projected pattern capture, homography composition, reprojection error check (§8.3)
- *Done when:* an operator can go from an uncalibrated setup to projected footprints landing on the tape markers using either method, without typing a value.

**F-8 — Operator App & Unattended Running**
- Operator window: camera preview with feet and Walker IDs, top-down floor view with feet/inch grid, History density, active Echoes, metrics
- Parameter tuning UI over the §9 registry
- History inspection, export, and clear
- Optional debug overlays on the projector (off by default)
- Launch straight into running mode; black projector and operator-visible errors on camera loss or missing calibration (§11)
- *Done when:* the installation survives a power cycle and a camera unplug/replug without operator intervention beyond reconnecting hardware.

### 13.3 Build Order

1. **In parallel:** F-1 (start first), F-4 (research runs alongside)
2. **On simulation input:** F-2 → F-3 → F-6 → F-7
3. **Once F-4 decides:** F-5
4. **Last:** F-8, which brings everything together on live input

Everything except F-5 can be built and tested before the sensing hardware is chosen.

---

## 14. Amendment Process

The covenant changes only deliberately. An amendment states the section changed, the reason (usually an on-floor observation, a research outcome, or a spec that could not be satisfied), and which existing specs are affected. Version number increments on every amendment.

### 14.1 Change Log

**v0.3**
- Feature breakdown (§13) consolidated from 21 small specs into 8 features, each with scope, completion criteria, and a build order. The sensing research spec is now F-4 (was F-10).

**v0.2**
- Units changed from metres to inches (feet and inches in the UI); all length parameters converted.
- Echo selection picks the most recent matching Path instead of a random one; History is kept in LRU Recency Order and evicts least recently used. INV-5 rewritten as determinism. Added Q9 and `echo_refreshes_recency`.
- Heading-aware matching is on by default (60°).
- `footprint_brightness` defaults to max (1.0).
- Camera and body tracker decisions deferred to a new research spec (then F-10, now F-4); candidates now include IR, stereo webcam, and stereo IR. Added INV-11 and §5.1. Specs renumbered.
- Automatic calibration uses a click-select interface instead of typed width/height.
- Pipeline section rewritten as two stage tables in place of the ASCII diagram.
- "Native and performant macOS execution"; added an idle-load performance target.

**v0.1** — Initial draft.
