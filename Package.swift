// swift-tools-version: 6.0
import PackageDescription

// Module graph mirrors the covenant's boundaries (§6.4):
// - Core has no dependencies; everything else builds on it.
// - Render depends only on Core, so it cannot see Paths or History.
// - Sensing depends only on Core's PoseSource protocol; hardware lives in Capture (INV-11).
let package = Package(
    name: "Footsteps",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "FootstepsCore", targets: ["FootstepsCore"]),
        .library(name: "FootstepsSimulation", targets: ["FootstepsSimulation"]),
        .library(name: "FootstepsMemory", targets: ["FootstepsMemory"]),
        .library(name: "FootstepsEcho", targets: ["FootstepsEcho"]),
        .library(name: "FootstepsSensing", targets: ["FootstepsSensing"]),
        .library(name: "FootstepsCapture", targets: ["FootstepsCapture"]),
        .library(name: "FootstepsRender", targets: ["FootstepsRender"]),
        .library(name: "FootstepsCalibration", targets: ["FootstepsCalibration"]),
    ],
    targets: [
        // F-1
        .target(name: "FootstepsCore"),
        .target(name: "FootstepsSimulation", dependencies: ["FootstepsCore"]),
        // F-2
        .target(name: "FootstepsMemory", dependencies: ["FootstepsCore"]),
        // F-3
        .target(name: "FootstepsEcho", dependencies: ["FootstepsCore", "FootstepsMemory"]),
        // F-5
        .target(name: "FootstepsSensing", dependencies: ["FootstepsCore"]),
        .target(name: "FootstepsCapture", dependencies: ["FootstepsCore"]),
        // F-6
        .target(name: "FootstepsRender", dependencies: ["FootstepsCore"]),
        // F-7
        .target(name: "FootstepsCalibration", dependencies: ["FootstepsCore", "FootstepsRender"]),

        .testTarget(name: "FootstepsCoreTests", dependencies: ["FootstepsCore"]),
        .testTarget(name: "FootstepsSimulationTests", dependencies: ["FootstepsSimulation"]),
        .testTarget(name: "FootstepsMemoryTests", dependencies: ["FootstepsMemory", "FootstepsSimulation"]),
        .testTarget(name: "FootstepsEchoTests", dependencies: ["FootstepsEcho", "FootstepsSimulation"]),
        .testTarget(name: "FootstepsSensingTests", dependencies: ["FootstepsSensing", "FootstepsSimulation"]),
        .testTarget(name: "FootstepsCaptureTests", dependencies: ["FootstepsCapture"]),
        .testTarget(name: "FootstepsRenderTests", dependencies: ["FootstepsRender"]),
        .testTarget(name: "FootstepsCalibrationTests", dependencies: ["FootstepsCalibration"]),
    ]
)
