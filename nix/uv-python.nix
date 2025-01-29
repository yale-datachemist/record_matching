{pyproject-nix, pyproject-build-systems, pyproject-overrides, lib, workspace, python3, callPackage, protobuf}:
let overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };
in
(callPackage pyproject-nix.build.packages {
  python = python3;
}).overrideScope (
  lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    pyproject-overrides.cuda
    pyproject-overrides.default
    pyproject-overrides.vectorlink-source-projects
  ]
)
