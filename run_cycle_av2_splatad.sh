#!/bin/bash
set -e
# AV2 cycle reconstruction: train + render pipeline using SplatAD (see also run_cycle_av2_neurad.sh).

# === CONFIGURATION ===
# Sequence UUID (must match the downloaded scene)
SEQ="${SEQ:-05fa5048-f355-3274-b565-c0ddc547b315}"
SHIFT="${SHIFT:--3.0 0.0 0.0}"
GPU="${GPU:-0}"
SPLATAD_NUM_ITER="${SPLATAD_NUM_ITER:-30001}"

# Repo data layout:
#   data/01_spatad_cycle/<config_id>/raw/       ← original AV2 scene (scene_dir)
#   data/01_spatad_cycle/<config_id>/rendered/   ← shifted scene output
#   data/01_spatad_cycle/<config_id>/pairs/      ← gt/ (from shifted/gt) + corrupted/ (reverse-shift render)
REPO_DATA="${REPO_DATA:-../data/01_spatad_cycle}"
CONFIG_ID="${CONFIG_ID:-scene_05}"

SCENE_ROOT="$REPO_DATA/$CONFIG_ID"
RAW_DIR="$SCENE_ROOT/raw"

# Nerfstudio outputs go under the scene tree so parallel Ray jobs never collide.
NS_OUTPUT_DIR="$SCENE_ROOT/logs/nerfstudio"
mkdir -p "$NS_OUTPUT_DIR"

# render_shifted_splatad_av2.py writes to: <target_root>/sensor/<split>/<log_id>/...
# We use intermediate dirs so scene_dir can point at the actual scene folder.
SHIFTED_TARGET="$SCENE_ROOT/shifted"
SHIFTED_SCENE="$SHIFTED_TARGET/sensor/train/$SEQ"

REVERSE_TARGET="$SCENE_ROOT/reverse_shifted"
REVERSE_SCENE="$REVERSE_TARGET/sensor/train/$SEQ"

REVERSE_SHIFT=$(echo "$SHIFT" | awk '{for(i=1;i<=NF;i++) printf "%s%s", -$i, (i<NF?" ":"\n")}')

echo "=== Cycle Reconstruction Pipeline for AV2 ==="
echo "Sequence:       $SEQ"
echo "Config:         $CONFIG_ID"
echo "Raw scene_dir:  $RAW_DIR"
echo "NS outputs:     $NS_OUTPUT_DIR"
echo "Shift:          $SHIFT"
echo "Reverse shift:  $REVERSE_SHIFT  (Step 5: undo cycle apply_shift → GT pose)"
echo "GPU:            $GPU"
echo "SplatAD iters:  $SPLATAD_NUM_ITER"
echo "=============================================="

# -------------------------------------------------------
# STEP 1: Train SplatAD on full original AV2 scene
#         train_split_fraction defaults to 1.0 in ADDataParserConfig
#         scene_dir bypasses data/sensor/split layout
# -------------------------------------------------------
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

# -------------------------------------------------------
# STEP 2: Render shifted scene (images + lidar) in AV2 format
#         Writes to SHIFTED_TARGET/sensor/train/<SEQ>/
# -------------------------------------------------------
echo ""
echo ">>> STEP 2: Rendering shifted scene ..."
python nerfstudio/scripts/render_shifted_splatad_av2.py \
    --load-config "$STEP1_CONFIG" \
    --shift $SHIFT \
    --render_point_clouds True \
    --pose_source train \
    --original_data_root "$RAW_DIR" \
    --target_data_root "$SHIFTED_TARGET" \

# -------------------------------------------------------
# STEP 3: Copy metadata from raw AV2 to shifted scene dir
#         (calibration, ego poses, annotations — needed by dataparser)
# -------------------------------------------------------
echo ""
echo ">>> STEP 3: Copying AV2 metadata to shifted scene directory ..."
cp -rn "$RAW_DIR/calibration" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/city_SE3_egovehicle.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
cp -n "$RAW_DIR/annotations.feather" "$SHIFTED_SCENE/" 2>/dev/null || true
if [ -d "$RAW_DIR/map" ]; then
    cp -rn "$RAW_DIR/map" "$SHIFTED_SCENE/" 2>/dev/null || true
fi
echo "Metadata copied to $SHIFTED_SCENE"

# -------------------------------------------------------
# STEP 4: Train SplatAD on shifted (rendered) GT
#         scene_dir points to the shifted scene folder
#         apply_shift=True for ego_shift_xyz on poses
#         add_missing_points=False (unlike STEP 1): shifted data is already
#         rendered; skip synthetic lidar fill (memory + consistency).
# -------------------------------------------------------
echo ""
echo ">>> STEP 4: Training SplatAD on shifted data (cycle) ..."
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
echo "Step 4 config: $STEP4_CONFIG"

# -------------------------------------------------------
# STEP 5: Render back at GT pose (reverse shift)
#         Cycle checkpoint keeps apply_shift=True: dataset cameras = M(P)+SHIFT.
#         render_shifted_splatad_av2 adds REVERSE_SHIFT=-SHIFT → M(P), matching raw GT.
# -------------------------------------------------------
echo ""
echo ">>> STEP 5: Rendering reverse-shifted scene (GT pose) ..."
python nerfstudio/scripts/render_shifted_splatad_av2.py \
    --load-config "$STEP4_CONFIG" \
    --shift $REVERSE_SHIFT \
    --render_point_clouds False \
    --pose_source train \
    --original_data_root "$SHIFTED_SCENE" \
    --target_data_root "$REVERSE_TARGET" \

# -------------------------------------------------------
# Copy results to repo rendered/ dir
# -------------------------------------------------------
echo ""
echo ">>> Copying shifted images to rendered/ ..."
RENDERED_DIR="$SCENE_ROOT/rendered"
mkdir -p "$RENDERED_DIR"
cp -r "$SHIFTED_SCENE/sensors/cameras/"* "$RENDERED_DIR/" 2>/dev/null || true

# -------------------------------------------------------
# STEP 6: Pairs for training: gt from render_shifted_splatad_av2 (SHIFTED_SCENE/gt/<cam>/...) + corrupted
# -------------------------------------------------------
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

# -------------------------------------------------------
# STEP 7: Generate comparison video (GT | Shifted | Reverse)
# -------------------------------------------------------
echo ""
echo ">>> STEP 7: Generating comparison video ..."
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"

python3 nerfstudio/scripts/generate_comparison_video.py \
    --shifted-root "$SHIFTED_SCENE" \
    --reverse-root "$REVERSE_SCENE" \
    --output-dir "$SCENE_ROOT" \
    --height "$VIDEO_HEIGHT"

echo ""
echo "=== Pipeline complete ==="
echo "GT (comparison & pairs): $SHIFTED_SCENE/gt/"
echo "Shifted images:         $SHIFTED_SCENE/sensors/cameras/"
echo "Reverse-shifted images: $REVERSE_SCENE/sensors/cameras/"
echo "Rendered (repo dir):    $RENDERED_DIR/"
echo "Pairs (gt):        $GT_PAIRS/"
echo "Pairs (corrupted): $CORRUPTED_PAIRS/"
echo "Comparison videos:      $SCENE_ROOT/comparison_*.mp4"
