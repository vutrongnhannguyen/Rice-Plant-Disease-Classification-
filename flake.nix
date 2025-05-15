{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f rec {
          pkgs = import nixpkgs {inherit system;};
          python = pkgs.python312;
        });
  in {
    devShells = forEachSupportedSystem ({
      pkgs,
      python,
    }: {
      default = pkgs.mkShell {
        venvDir = ".venv";
        packages =
          [python pkgs.pandoc pkgs.texliveFull pkgs.ruff]
          ++ (with python.pkgs; [
            pandas
            numpy
            scikit-learn
            jupyter
            notebook
            matplotlib
            seaborn
            opencv4
            tqdm
            cleanvision
            venvShellHook
          ]);
      };
    });
  };
}
