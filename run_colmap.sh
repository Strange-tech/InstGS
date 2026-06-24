SCENE_NAME=figurines
DATASET_PATH=./data/$SCENE_NAME

# COLMAP
echo "Running COLMAP feature extraction and matching..."
colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images
colmap exhaustive_matcher --database_path $DATASET_PATH/database.db
mkdir $DATASET_PATH/sparse
colmap mapper --database_path $DATASET_PATH/database.db \
                --image_path $DATASET_PATH/images \
                --output_path $DATASET_PATH/sparse

# # Dense reconstruction
# echo "Running COLMAP dense reconstruction..."
# colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \
#     --max_image_size 2000

# colmap patch_match_stereo \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# colmap stereo_fusion \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path $DATASET_PATH/dense/fused.ply