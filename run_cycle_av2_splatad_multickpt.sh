#!/bin/bash
# AV2 cycle: train on raw AV2 → shifted render → copy metadata → flatten traj → train on shifted →
# reverse-shift render for each cycle step-*.ckpt.
#
# Env: SCENE_ROOT or REPO_DATA+CONFIG_ID; raw AV2 under <scene>/raw/; checkpoints under <scene>/logs/nerfstudio/.

set -e

SEQ="${SEQ:-05fa5048-f355-3274-b565-c0ddc547b315}"
SHIFT="${SHIFT:--3.0 0.0 0.0}"
GPU="${GPU:-0}"
SPLATAD_NUM_ITER="${SPLATAD_NUM_ITER:-30001}"
AV2_SENSOR_SPLIT="${AV2_SENSOR_SPLIT:-train}"

if [ -n "${SCENE_ROOT:-}" ]; then
    CONFIG_ID="${CONFIG_ID:-$(basename "$SCENE_ROOT")}"
else
    REPO_DATA="${REPO_DATA:-../data/03_spatad_cycle_multickpt}"
    CONFIG_ID="${CONFIG_ID:-scene_05}"
    SCENE_ROOT="$REPO_DATA/$CONFIG_ID"
fi
RAW_DIR="$SCENE_ROOT/raw"

NS_OUTPUT_DIR="$SCENE_ROOT/logs/nerfstudio"
mkdir -p "$NS_OUTPUT_DIR"

SHIFTED_TARGET="$SCENE_ROOT/shifted"
SHIFTED_SCENE="$SHIFTED_TARGET/sensor/$AV2_SENSOR_SPLIT/$SEQ"
SHIFTED_TRAJ="$SCENE_ROOT/shifted_traj"

REVERSE_SHIFT=$(echo "$SHIFT" | awk '{for(i=1;i<=NF;i++) printf "%s%s", -$i, (i<NF?" ":"\n")}')

echo "=== SplatAD multickpt (train + shifted render + cycle train + reverse per ckpt) ==="
echo "Sequence:       $SEQ"
echo "Config:         $CONFIG_ID"
echo "Raw scene_dir:  $RAW_DIR"
echo "NS outputs:     $NS_OUTPUT_DIR"
echo "Shift:          $SHIFT"
echo "Reverse shift:  $REVERSE_SHIFT"
echo "GPU:            $GPU"
echo "SplatAD iters:  $SPLATAD_NUM_ITER"
echo "=============================================="

echo ""
echo ">>> STEP 1: Training SplatAD on original AV2 scene (full split) ..."
python nerfstudio/scripts/train.py splatad \
    --experiment_name="full_av2_${CONFIG_ID}" \
    --output-dir "$NS_OUTPUT_DIR" \
    --max-num-iterations "$SPLATAD_NUM_ITER" \
    --pipeline.model.max-steps "$SPLATAD_NUM_ITER" \
    argoverse2-data \
    --sequence "$SEQ" \
    --scene_dir "$RAW_DIR" \
    --include-deformable-actors True

STEP1_CONFIG=$(ls -td "$NS_OUTPUT_DIR/full_av2_${CONFIG_ID}"/splatad/*/config.yml | head -1)
echo "Step 1 config: $STEP1_CONFIG"

echo ""
echo ">>> STEP 2: Shifted render (full checkpoint) → shifted/ (AV2 layout) ..."
python nerfstudio/scripts/render_shifted_splatad_av2.py \
    --load-config "$STEP1_CONFIG" \
    --shift $SHIFT \
    --render_point_clouds True \
    --pose_source train \
    --original_data_root "$RAW_DIR" \
    --target_data_root "$SHIFTED_TARGET"

echo ""
echo ">>> STEP 3: Copy AV2 metadata to shifted scene directory ..."
cp -rn "$RAW_DIR/calibration" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/city_SE3_egovehicle.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/annotations.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
if [ -d "$RAW_DIR/map" ]; then
    cp -rn "$RAW_DIR/map" "$SHIFTED_SCENE/" 2>/dev/null || true
fi

echo ""
echo ">>> STEP 4: Flatten shifted cameras → shifted_traj/<camera_name>/images/ ..."
rm -rf "$SHIFTED_TRAJ"
mkdir -p "$SHIFTED_TRAJ"
shopt -s nullglob
for cam_dir in "$SHIFTED_SCENE/sensors/cameras/"*/; do
    [ -d "$cam_dir" ] || continue
    cam_name=$(basename "$cam_dir")
    mkdir -p "$SHIFTED_TRAJ/$cam_name/images"
    for img in "$cam_dir"*.jpg; do
        [ -f "$img" ] || continue
        base=$(basename "$img")
        case "$base" in
            *_gt-rgb.jpg|*_depth.jpg) continue ;;
        esac
        cp -n "$img" "$SHIFTED_TRAJ/$cam_name/images/$base"
    done
done
shopt -u nullglob

echo ""
echo ">>> STEP 5: Training SplatAD on shifted data (cycle; multi-checkpoint run) ..."
python nerfstudio/scripts/train.py splatad \
    --experiment_name="cycle_av2_${CONFIG_ID}" \
    --output-dir "$NS_OUTPUT_DIR" \
    --max-num-iterations "$SPLATAD_NUM_ITER" \
    --pipeline.model.max-steps "$SPLATAD_NUM_ITER" \
    argoverse2-data \
    --sequence "$SEQ" \
    --scene_dir "$SHIFTED_SCENE" \
    --apply_shift True \
    --ego_shift_xyz $SHIFT \
    --add-missing-points False \
    --include-deformable-actors True

STEP4_CONFIG=$(ls -td "$NS_OUTPUT_DIR/cycle_av2_${CONFIG_ID}"/splatad/*/config.yml | head -1)
CYCLE_RUN_DIR=$(dirname "$STEP4_CONFIG")
echo "Step 5 cycle config: $STEP4_CONFIG"
echo "Cycle run dir:       $CYCLE_RUN_DIR"

echo ""
echo ">>> STEP 6: Reverse-shift renders for each cycle checkpoint ..."
mapfile -t CKPTS < <(ls -1 "$CYCLE_RUN_DIR/nerfstudio_models"/step-*.ckpt 2>/dev/null | sort -V)
if [ "${#CKPTS[@]}" -eq 0 ] || [ ! -f "${CKPTS[0]}" ]; then
    echo "ERROR: No step-*.ckpt under $CYCLE_RUN_DIR/nerfstudio_models" >&2
    exit 1
fi

for ckpt in "${CKPTS[@]}"; do
    [ -f "$ckpt" ] || continue
    base=$(basename "$ckpt" .ckpt)
    step_str=${base#step-}
    load_step=$((10#$step_str))
    REVERSE_TARGET="$SCENE_ROOT/reverse_shifted_${step_str}"
    rm -rf "$REVERSE_TARGET"
    echo "    Rendering reverse_shifted_${step_str} (load_step=$load_step) ..."
    python nerfstudio/scripts/render_shifted_splatad_av2.py \
        --load-config "$STEP4_CONFIG" \
        --load-step "$load_step" \
        --shift $REVERSE_SHIFT \
        --render_point_clouds False \
        --pose_source train \
        --original_data_root "$SHIFTED_SCENE" \
        --target_data_root "$REVERSE_TARGET"
done

echo ""
echo "=== Multickpt pipeline complete (rendered/ + pairs/ from pipeline save_pairs if used) ==="
echo "STEP1_CONFIG:              $STEP1_CONFIG"
echo "STEP4_CONFIG:              $STEP4_CONFIG"
echo "CYCLE_RUN_DIR:             $CYCLE_RUN_DIR"
echo "shifted_traj (scene root): $SHIFTED_TRAJ/"
echo "Reverse outputs:           $SCENE_ROOT/reverse_shifted_*/"
