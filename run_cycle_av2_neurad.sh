#!/bin/bash
set -e
# AV2 cycle reconstruction: train + render pipeline using NeuRAD (see run_cycle_av2_splatad.sh for SplatAD).
# Shifted RGB + lidar: render_shifted_neurad_av2.py (NeuRAD get_outputs_for_lidar + same AV2 feather merge as SplatAD).

# === CONFIGURATION ===
SEQ="${SEQ:-05fa5048-f355-3274-b565-c0ddc547b315}"
SHIFT="${SHIFT:--3.0 0.0 0.0}"
GPU="${GPU:-0}"
NEURAD_NUM_ITER="${NEURAD_NUM_ITER:-20001}"

# Default layout matches scenario ``02_neurad_cycle`` (Pipeline sets REPO_DATA from job config).
REPO_DATA="${REPO_DATA:-../data/02_neurad_cycle}"
CONFIG_ID="${CONFIG_ID:-scene_05}"

# Same layout as SplatAD run; experiment names are suffixed with _neurad so logs do not collide.
SCENE_ROOT="$REPO_DATA/$CONFIG_ID"
RAW_DIR="$SCENE_ROOT/raw"

NS_OUTPUT_DIR="$SCENE_ROOT/logs/nerfstudio"
mkdir -p "$NS_OUTPUT_DIR"

SHIFTED_TARGET="$SCENE_ROOT/shifted"
SHIFTED_SCENE="$SHIFTED_TARGET/sensor/train/$SEQ"

REVERSE_TARGET="$SCENE_ROOT/reverse_shifted"
REVERSE_SCENE="$REVERSE_TARGET/sensor/train/$SEQ"

REVERSE_SHIFT=$(echo "$SHIFT" | awk '{for(i=1;i<=NF;i++) printf "%s%s", -$i, (i<NF?" ":"\n")}')

EXP_FULL="full_av2_neurad_${CONFIG_ID}"
EXP_CYCLE="cycle_av2_neurad_${CONFIG_ID}"

echo "=== Cycle Reconstruction Pipeline for AV2 (NeuRAD) ==="
echo "Sequence:       $SEQ"
echo "Config:         $CONFIG_ID"
echo "Raw scene_dir:  $RAW_DIR"
echo "NS outputs:     $NS_OUTPUT_DIR"
echo "Shift:          $SHIFT"
echo "Reverse shift:  $REVERSE_SHIFT  (Step 5: undo cycle apply_shift → GT pose)"
echo "GPU:            $GPU"
echo "NeuRAD iters:   $NEURAD_NUM_ITER"
echo "======================================================"

echo ""
echo ">>> STEP 1: Training NeuRAD on original AV2 scene (full split) ..."
python nerfstudio/scripts/train.py neurad \
    --pipeline.datamanager.num_processes 0 \
    --experiment_name="$EXP_FULL" \
    --output-dir "$NS_OUTPUT_DIR" \
    --max-num-iterations "$NEURAD_NUM_ITER" \
    argoverse2-data \
    --sequence "$SEQ" \
    --scene_dir "$RAW_DIR" \
    --include-deformable-actors True

STEP1_CONFIG=$(ls -td "$NS_OUTPUT_DIR/$EXP_FULL"/neurad/*/config.yml | head -1)
echo "Step 1 config: $STEP1_CONFIG"

echo ""
echo ">>> STEP 2: Rendering shifted scene (RGB + AV2 lidar via NeuRAD) ..."
python nerfstudio/scripts/render_shifted_neurad_av2.py \
    --load-config "$STEP1_CONFIG" \
    --shift $SHIFT \
    --render_point_clouds True \
    --pose_source train \
    --original_data_root "$RAW_DIR" \
    --target_data_root "$SHIFTED_TARGET" \

echo ""
echo ">>> STEP 3: Copying AV2 metadata (and lidar files) to shifted scene directory ..."
cp -rn "$RAW_DIR/calibration" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/city_SE3_egovehicle.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/annotations.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
if [ -d "$RAW_DIR/map" ]; then
    cp -rn "$RAW_DIR/map" "$SHIFTED_SCENE/" 2>/dev/null || true
fi
echo "Metadata copied to $SHIFTED_SCENE"

echo ""
echo ">>> STEP 4: Training NeuRAD on shifted data (cycle) ..."
python nerfstudio/scripts/train.py neurad \
    --pipeline.datamanager.num_processes 0 \
    --experiment_name="$EXP_CYCLE" \
    --output-dir "$NS_OUTPUT_DIR" \
    --max-num-iterations "$NEURAD_NUM_ITER" \
    argoverse2-data \
    --sequence "$SEQ" \
    --scene_dir "$SHIFTED_SCENE" \
    --apply_shift True \
    --ego_shift_xyz $SHIFT \
    --add-missing-points False \
    --include-deformable-actors True

STEP4_CONFIG=$(ls -td "$NS_OUTPUT_DIR/$EXP_CYCLE"/neurad/*/config.yml | head -1)
echo "Step 4 config: $STEP4_CONFIG"

echo ""
echo ">>> STEP 5: Rendering reverse-shifted scene (GT pose) ..."
python nerfstudio/scripts/render_shifted_neurad_av2.py \
    --load-config "$STEP4_CONFIG" \
    --shift $REVERSE_SHIFT \
    --render_point_clouds True \
    --pose_source train \
    --original_data_root "$SHIFTED_SCENE" \
    --target_data_root "$REVERSE_TARGET" \

echo ""
echo ">>> Copying shifted images to rendered/ ..."
RENDERED_DIR="$SCENE_ROOT/rendered"
mkdir -p "$RENDERED_DIR"
cp -r "$SHIFTED_SCENE/sensors/cameras/"* "$RENDERED_DIR/" 2>/dev/null || true

echo ""
echo ">>> STEP 6: Creating pairs (gt + corrupted) ..."
PAIRS_DIR="$SCENE_ROOT/pairs"
GT_PAIRS="$PAIRS_DIR/gt"
CORRUPTED_PAIRS="$PAIRS_DIR/corrupted"

rm -rf "$PAIRS_DIR"
mkdir -p "$GT_PAIRS" "$CORRUPTED_PAIRS"

for cam_dir in "$REVERSE_SCENE/sensors/cameras/"*/; do
    [ -d "$cam_dir" ] || continue
    cam_name=$(basename "$cam_dir")
    mkdir -p "$GT_PAIRS/$cam_name" "$CORRUPTED_PAIRS/$cam_name"
    for corrupt in "$cam_dir"*.jpg; do
        [ -f "$corrupt" ] || continue
        fname=$(basename "$corrupt")
        gt_src="$SHIFTED_SCENE/gt/$cam_name/$fname"
        if [ -f "$gt_src" ]; then
            cp "$gt_src" "$GT_PAIRS/$cam_name/$fname"
            cp "$corrupt" "$CORRUPTED_PAIRS/$cam_name/$fname"
        fi
    done
done

echo ""
echo ">>> STEP 7: Generating comparison video ..."
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"

python3 nerfstudio/scripts/generate_comparison_video.py \
    --shifted-root "$SHIFTED_SCENE" \
    --reverse-root "$REVERSE_SCENE" \
    --output-dir "$SCENE_ROOT" \
    --height "$VIDEO_HEIGHT"

echo ""
echo "=== Pipeline complete (NeuRAD) ==="
echo "GT (comparison & pairs): $SHIFTED_SCENE/gt/"
echo "Shifted images:         $SHIFTED_SCENE/sensors/cameras/"
echo "Reverse-shifted images: $REVERSE_SCENE/sensors/cameras/"
echo "Rendered (repo dir):    $RENDERED_DIR/"
echo "Pairs (gt):        $GT_PAIRS/"
echo "Pairs (corrupted): $CORRUPTED_PAIRS/"
echo "Comparison videos:      $SCENE_ROOT/comparison_*.mp4"
