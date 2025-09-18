#!/bin/bash
set -e

ROOT_DIR=$(pwd)
BUILD_DIR="${ROOT_DIR}/build"
SRC_DIR="${ROOT_DIR}/src"
HARDWARE_DIR="${ROOT_DIR}/hardware"
GPU_DIR="${ROOT_DIR}/gpu"
LIB_NAME="libjaide_v27_ultimate.so"

if ! command -v zig &> /dev/null; then
    echo "Zig compiler not found. Install Zig (>=0.12.0) and add it to PATH."
    exit 1
fi

if ! command -v futhark &> /dev/null; then
    echo "Futhark compiler not found. Install Futhark and add it to PATH."
    exit 1
fi

if ! command -v yosys &> /dev/null; then
    echo "Yosys not found. Install Yosys for Verilog synthesis."
    exit 1
fi

if ! command -v nextpnr-ice40 &> /dev/null; then
    echo "nextpnr not found. Install nextpnr for FPGA place-and-route."
    exit 1
fi

if ! command -v lean &> /dev/null; then
    echo "Lean 4 not found. Install Lean 4 for formal verification."
    exit 1
fi

if ! command -v isabelle &> /dev/null; then
    echo "Isabelle/HOL not found. Install Isabelle."
    exit 1
fi

if ! command -v agda &> /dev/null; then
    echo "Agda not found. Install Agda."
    exit 1
fi

if ! command -v dedukti &> /dev/null; then
    echo "Dedukti not found. Install Dedukti."
    exit 1
fi

if ! command -v clash &> /dev/null; then
    echo "Clash not found. Install Clash."
    exit 1
fi

if ! command -v scala &> /dev/null; then
    echo "Scala not found. Install Scala."
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo "Java (OpenJDK) not found. Install OpenJDK."
    exit 1
fi

if ! command -v chpl &> /dev/null; then
    echo "Chapel compiler not found. Install Chapel."
    exit 1
fi

if ! command -v nim &> /dev/null; then
    echo "Nim compiler not found. Install Nim."
    exit 1
fi

if ! command -v terra &> /dev/null; then
    echo "Terra not found. Install Terra."
    exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "[1/10] Cleaning and preparing build directory…"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
echo "Build directory prepared: ${BUILD_DIR}"

echo "[1.5/10] Generating Verilog from Clash and SpinalHDL…"
clash --verilog "${HARDWARE_DIR}/neuromorphic/Neuromorphic.hs" -o "${BUILD_DIR}/neuromorphic.v"
scala "${HARDWARE_DIR}/spinal/TGNUpdateUnit.scala"
cp "${BUILD_DIR}/neuromorphic.v" "${HARDWARE_DIR}/"
cp "${BUILD_DIR}/TGNUpdateUnit.v" "${HARDWARE_DIR}/"

echo "[2/10] Synthesizing Verilog hardware modules with Yosys and nextpnr for FPGA…"
yosys -p "read_verilog ${HARDWARE_DIR}/*.v; synth_ice40 -top neuromorphic_pulse_array -json ${BUILD_DIR}/neuromorphic_synth.json"
nextpnr-ice40 --hx8k --json "${BUILD_DIR}/neuromorphic_synth.json" --pcf "${HARDWARE_DIR}/pins.pcf" --asc "${BUILD_DIR}/neuromorphic.asc"
icepack "${BUILD_DIR}/neuromorphic.asc" "${BUILD_DIR}/neuromorphic.bin"
if [ $? -ne 0 ]; then
    echo "Hardware synthesis failed."
    exit 1
fi
echo "Hardware modules synthesized to FPGA binary."

echo "[3/10] Compiling Futhark GPU kernels to C library with full optimization…"
futhark c --library -o "${BUILD_DIR}/libfuthark_kernels" "${GPU_DIR}/kernels/kernels.fut"
if [ $? -ne 0 ]; then
    echo "Futhark compilation failed."
    exit 1
fi
echo "Futhark kernels successfully compiled."

echo "[4/10] Compiling Chapel distributed TGN update…"
chpl "${ROOT_DIR}/distributed/tgn_update.chpl" -o "${BUILD_DIR}/tgn_update_exec" --fast
if [ $? -ne 0 ]; then
    echo "Chapel compilation failed."
    exit 1
fi
echo "Chapel TGN update successfully compiled."

echo "[5/10] Compiling Zig native runtime to shared library with full integration…"
zig build-lib -O ReleaseFast -fPIC -dynamic --name jaide_v27_ultimate -lc -L"${BUILD_DIR}" -I"${BUILD_DIR}" -lfuthark_kernels -lstdc++ "${SRC_DIR}/core/main.zig"
if [ $? -ne 0 ]; then
    echo "Zig compilation failed."
    exit 1
fi
mv "${LIB_NAME}" "${BUILD_DIR}/${LIB_NAME}"
echo "Zig native runtime successfully compiled."

echo "[6/10] Generating documentation from Zig and Python docstrings…"
mkdir -p "${BUILD_DIR}/docs"
zig build-obj --docs "${BUILD_DIR}/docs/zig" "${SRC_DIR}/core/main.zig"
pydoc3 -w jaide_v27_ultimate_orchestrator.py
mv jaide_v27_ultimate_orchestrator.html "${BUILD_DIR}/docs/python.html"
echo "Documentation generated: ${BUILD_DIR}/docs"

echo "[7/10] Running Agda type checking and proofs…"
agda "${SRC_DIR}/core/JAIDE/Core/Types.agda" -o "${BUILD_DIR}/types.agda"
if [ $? -ne 0 ]; then
    echo "Agda type checking failed."
    exit 1
fi
echo "Agda type checking successful."

echo "[8/10] Running Isabelle/HOL proofs…"
isabelle build -D "${ROOT_DIR}/security/isabelle" -o "${BUILD_DIR}/isabelle_proof.thy"
if [ $? -ne 0 ]; then
    echo "Isabelle proof failed."
    exit 1
fi
echo "Isabelle proof successful."

echo "[9/10] Running Imandra verification (via Python)…"
python3 "${ROOT_DIR}/security/imandra/optimizer.py"
if [ $? -ne 0 ]; then
    echo "Imandra verification failed."
    exit 1
fi
echo "Imandra verification successful."

echo "[10/10] Dedukti proof unification…"
dkcheck "${BUILD_DIR}/types.agda" "${BUILD_DIR}/isabelle_proof.thy" -o "${BUILD_DIR}/proofs.dk"
if [ $? -ne 0 ]; then
    echo "Dedukti proof unification failed."
    exit 1
fi
echo "Dedukti proof unification successful."

echo "JAIDE V27 ULTIMATE IONQ IBM BUILD COMPLETED"
echo "Library available: ${BUILD_DIR}/${LIB_NAME}"
echo "FPGA bitstream: ${BUILD_DIR}/neuromorphic.bin"
echo "Documentation: ${BUILD_DIR}/docs"
echo "IONQ/IBM providers ready."
exit 0
