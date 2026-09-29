# App

The thin Xcode app target (§5) lives here: the AppKit shell hosting the operator window
and the borderless fullscreen projector window. It wires the package modules together
and is the only place that imports `FootstepsCapture` (INV-11).

Created during F-6 (projector window) and completed in F-8.
