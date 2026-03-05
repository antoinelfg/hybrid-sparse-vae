#!/usr/bin/env bash
set -euo pipefail

# Generate Libri2Mix (2 speakers) with a lightweight config:
#   - wav8k
#   - min mode
#   - mix_clean only
#   - train-100 + dev + test subsets
#
# Usage:
#   bash scripts/setup_librimix.sh
#   bash scripts/setup_librimix.sh /path/to/storage_dir
#
# Output layout (default storage_dir=./data):
#   data/Libri2Mix/wav8k/min/{train-100,dev,test}/{mix_clean,s1,s2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_DIR="${1:-$REPO_ROOT/data}"
TOOLS_DIR="${LIBRIMIX_TOOLS_DIR:-$STORAGE_DIR/_tools/LibriMix}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENABLE_WHAM_AUGMENT="${ENABLE_WHAM_AUGMENT:-0}"

LIBRISPEECH_DIR="$STORAGE_DIR/LibriSpeech"
WHAM_DIR="$STORAGE_DIR/wham_noise"
LIBRI2MIX_ROOT="$STORAGE_DIR/Libri2Mix"
META_SRC="$TOOLS_DIR/metadata/Libri2Mix"
META_DST="$STORAGE_DIR/metadata/Libri2Mix_train100_dev_test"

echo "[setup] storage_dir=$STORAGE_DIR"
echo "[setup] tools_dir=$TOOLS_DIR"
mkdir -p "$STORAGE_DIR" "$(dirname "$TOOLS_DIR")"

# Ensure Python extension modules (e.g. scipy) resolve C++ runtime from conda first.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -d "$TOOLS_DIR/.git" && ! -f "$TOOLS_DIR/generate_librimix.sh" ]]; then
  echo "[clone] Detected incomplete LibriMix clone, resetting..."
  rm -rf "$TOOLS_DIR"
fi

if [[ ! -d "$TOOLS_DIR/.git" ]]; then
  echo "[clone] Cloning official LibriMix repo..."
  git clone https://github.com/JorisCos/LibriMix.git "$TOOLS_DIR"
else
  echo "[clone] Updating existing LibriMix clone..."
  git -C "$TOOLS_DIR" fetch origin
  if git -C "$TOOLS_DIR" show-ref --quiet refs/remotes/origin/main; then
    git -C "$TOOLS_DIR" checkout -B main origin/main
  else
    git -C "$TOOLS_DIR" checkout -B master origin/master
  fi
fi

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  echo "[deps] Installing LibriMix generation dependencies..."
  "$PYTHON_BIN" -m pip install -r "$TOOLS_DIR/requirements.txt"
else
  echo "[deps] SKIP_PIP_INSTALL=1 -> skipping pip install."
fi

# Preflight check for LibriMix generation runtime deps.
if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
missing = [m for m in ["soundfile", "pandas", "numpy", "scipy", "tqdm"] if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:" + ",".join(missing))
    sys.exit(1)
PY
then
  echo "[deps] Missing Python dependencies required for LibriMix generation."
  if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
    echo "[deps] Re-run with SKIP_PIP_INSTALL=0, or install manually:"
    echo "       $PYTHON_BIN -m pip install soundfile pandas numpy scipy tqdm"
  fi
  exit 1
fi

download_librispeech_subset() {
  local subset="$1"
  local url="$2"
  local target="$LIBRISPEECH_DIR/$subset"
  local archive="$STORAGE_DIR/$subset.tar.gz"

  if [[ -d "$target" ]]; then
    echo "[data] LibriSpeech/$subset already exists, skipping."
    return
  fi

  echo "[data] Downloading LibriSpeech/$subset ..."
  wget -c --tries=0 --read-timeout=20 "$url" -O "$archive"
  tar -xzf "$archive" -C "$STORAGE_DIR"
  rm -f "$archive"
}

download_librispeech_subset "train-clean-100" "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
download_librispeech_subset "dev-clean" "http://www.openslr.org/resources/12/dev-clean.tar.gz"
download_librispeech_subset "test-clean" "http://www.openslr.org/resources/12/test-clean.tar.gz"

WHAM_ZIP="$STORAGE_DIR/wham_noise.zip"
wham_ready=0
if [[ -d "$WHAM_DIR/tr" && -d "$WHAM_DIR/cv" && -d "$WHAM_DIR/tt" ]]; then
  if find "$WHAM_DIR/tr" -type f -name "*.wav" -print -quit | grep -q .; then
    wham_ready=1
  fi
fi

if [[ "$wham_ready" != "1" ]]; then
  if [[ ! -f "$WHAM_ZIP" ]]; then
    echo "[data] Downloading WHAM! noise..."
  else
    echo "[data] Resuming/using existing WHAM! archive: $WHAM_ZIP"
  fi
  wget -c --tries=0 --read-timeout=20 \
    "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip" \
    -O "$WHAM_ZIP"
  echo "[data] Extracting WHAM! noise (idempotent)..."
  unzip -qn "$WHAM_ZIP" -d "$STORAGE_DIR"
else
  echo "[data] WHAM! noise already extracted, skipping download/unzip."
fi

if [[ "$ENABLE_WHAM_AUGMENT" == "1" ]]; then
  if ! command -v sox >/dev/null 2>&1; then
    echo "[error] ENABLE_WHAM_AUGMENT=1 but 'sox' binary is missing from PATH."
    exit 1
  fi
  echo "[prep] Augmenting WHAM! train noise (needed for train-clean-360 workflows)..."
  "$PYTHON_BIN" "$TOOLS_DIR/scripts/augment_train_noise.py" --wham_dir "$WHAM_DIR"
else
  echo "[prep] Skipping WHAM augmentation (not required for train-clean-100/dev/test metadata)."
fi

echo "[prep] Preparing filtered metadata (train-clean-100/dev-clean/test-clean)..."
mkdir -p "$META_DST"
cp "$META_SRC/libri2mix_train-clean-100.csv" "$META_DST/"
cp "$META_SRC/libri2mix_dev-clean.csv" "$META_DST/"
cp "$META_SRC/libri2mix_test-clean.csv" "$META_DST/"

# LibriMix upstream quirk:
# `write_noise()` is called even when `--types mix_clean`, but the `noise/` folder
# is not created in that branch, which can crash with LibsndfileError.
TOOLS_CREATE_SCRIPT="$TOOLS_DIR/scripts/create_librimix_from_metadata.py"
if ! grep -q "os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)" "$TOOLS_CREATE_SCRIPT"; then
  echo "[patch] Applying local LibriMix fix for mix_clean noise-dir handling..."
  sed -i "/save_path = os.path.join(dir_path, 'noise', ex_filename)/,/sf.write(abs_save_path, noise, freq)/{ /abs_save_path = os.path.abspath(save_path)/a\\
    os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)
}" "$TOOLS_CREATE_SCRIPT"
fi

# Cleanup incomplete splits from previous failed runs.
# The official script skips a split if its directory already exists.
cleanup_split_if_incomplete() {
  local split="$1"
  local csv_name="$2"
  local split_dir="$LIBRI2MIX_ROOT/wav8k/min/$split"
  local csv_path="$META_DST/$csv_name"

  if [[ ! -f "$csv_path" ]]; then
    return
  fi

  local expected_count
  expected_count=$(( $(wc -l < "$csv_path") - 1 ))
  if [[ "$expected_count" -le 0 ]]; then
    return
  fi

  if [[ -d "$split_dir" ]]; then
    local n_mix=0
    local n_s1=0
    local n_s2=0
    if [[ -d "$split_dir/mix_clean" ]]; then
      n_mix=$(find "$split_dir/mix_clean" -type f -name "*.wav" | wc -l)
    fi
    if [[ -d "$split_dir/s1" ]]; then
      n_s1=$(find "$split_dir/s1" -type f -name "*.wav" | wc -l)
    fi
    if [[ -d "$split_dir/s2" ]]; then
      n_s2=$(find "$split_dir/s2" -type f -name "*.wav" | wc -l)
    fi

    if [[ "$n_mix" -lt "$expected_count" || "$n_s1" -lt "$expected_count" || "$n_s2" -lt "$expected_count" ]]; then
      echo "[cleanup] Removing incomplete split $split_dir (mix=$n_mix s1=$n_s1 s2=$n_s2 expected=$expected_count)"
      rm -rf "$split_dir"
    fi
  fi
}

cleanup_split_if_incomplete "train-100" "libri2mix_train-clean-100.csv"
cleanup_split_if_incomplete "dev" "libri2mix_dev-clean.csv"
cleanup_split_if_incomplete "test" "libri2mix_test-clean.csv"

echo "[build] Generating Libri2Mix (2 speakers, wav8k/min, mix_clean)..."
"$PYTHON_BIN" "$TOOLS_DIR/scripts/create_librimix_from_metadata.py" \
  --librispeech_dir "$LIBRISPEECH_DIR" \
  --wham_dir "$WHAM_DIR" \
  --metadata_dir "$META_DST" \
  --librimix_outdir "$STORAGE_DIR" \
  --n_src 2 \
  --freqs 8k \
  --modes min \
  --types mix_clean

echo "[done] Libri2Mix generated at: $LIBRI2MIX_ROOT"
echo "[done] Expected splits:"
find "$LIBRI2MIX_ROOT/wav8k/min" -maxdepth 1 -mindepth 1 -type d | sort
