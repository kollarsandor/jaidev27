{
  description = "JAIDE V27 Ultimate Beast IONQ IBM Hybrid - Quantum Neuromorphic AI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "jaide-v27-ultimate-ionq-ibm";
        src = ./.;
        buildInputs = with pkgs; [
          zig_0_12
          futhark
          yosys
          nextpnr
          lean4
          isabelle
          agda
          dedukti
          clash
          python3
          python3Packages.torch
          python3Packages.numpy
          python3Packages.scikit-learn
          python3Packages.qiskit
          python3Packages.qiskit-ibm-runtime
          python3Packages.qiskit-ibm-provider
          python3Packages.ionq
          python3Packages.torch-geometric
          python3Packages.networkx
          chapel
          nim2
          graphviz
          streamlit
          haskellPackages.lens
          julia
          halide
          terra
          elixir
          pony
          unison
          racket
          factor
          scala
          openjdk
        ];
        buildPhase = ''
          bash build.sh
        '';
        installPhase = ''
          mkdir -p $out/bin
          cp build/libjaide_v27_ultimate.so $out/bin/
          cp -r build/docs $out/bin/docs
          cp jaide_v27_ultimate_orchestrator.py $out/bin/
          cp config.json $out/bin/
          cp tokenizer.py $out/bin/
          cp model.py $out/bin/
          cp training_dataset_full.json $out/bin/
          cp preprocess.py $out/bin/
          cp train_pipeline.py $out/bin/
          cp security_detector.py $out/bin/
          cp app.py $out/bin/
          cp ray_ultimate_train.py $out/bin/
          cp ultimate_security_module.py $out/bin/
        '';
      };

      devShell.${system} = pkgs.mkShell {
        buildInputs = with pkgs; [
          zig_0_12
          futhark
          yosys
          nextpnr
          lean4
          isabelle
          agda
          dedukti
          clash
          python3
          python3Packages.torch
          python3Packages.numpy
          python3Packages.scikit-learn
          python3Packages.qiskit
          python3Packages.qiskit-ibm-runtime
          python3Packages.qiskit-ibm-provider
          python3Packages.ionq
          python3Packages.torch-geometric
          python3Packages.networkx
          chapel
          nim2
          graphviz
          streamlit
          haskellPackages.lens
          julia
          halide
          terra
          elixir
          pony
          unison
          racket
          factor
          scala
          openjdk
        ];
      };
    };
}
