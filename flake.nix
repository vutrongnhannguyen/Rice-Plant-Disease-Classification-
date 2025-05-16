{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {nixpkgs, ...}: let
    inherit (nixpkgs) lib;

    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f rec {
          pkgs = import nixpkgs {inherit system;};
          python = pkgs.python311;
        });
  in {
    # environment setup
    devShells = forEachSupportedSystem ({
      pkgs,
      python,
    }: {
      default = pkgs.mkShell {
        packages = [
          python
          pkgs.uv

          pkgs.nodejs
          pkgs.pnpm

          pkgs.typst
        ];

        venvDir = "./.venv";

        env =
          {
            # Prevent uv from managing Python downloads
            UV_PYTHON_DOWNLOADS = "never";
            # Force uv to use nixpkgs Python interpreter
            UV_PYTHON = python.interpreter;
          }
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
            LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };
        shellHook = ''
          unset PYTHONPATH
        '';
      };
    });
  };
}
