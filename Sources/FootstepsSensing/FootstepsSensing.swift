// FootstepsSensing — headless sensing stages (F-5, §6.1 stages 3–5).
//
// WalkerTracker, StepDetector, and CoordinateMapper. Depends only on the
// PoseSource protocol, never on a specific camera or tracker (INV-11).
// Nothing downstream of CoordinateMapper touches Camera Space (§6.4).
