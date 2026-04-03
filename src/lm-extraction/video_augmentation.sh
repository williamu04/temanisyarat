#!/bin/bash

INPUT_DIR="./VIDEOSET/"
OUTPUT_DIR="./VIDEO_AUG"

mkdir -p "$OUTPUT_DIR"

# Loop setiap label (folder a, b, dst)
for LABEL_PATH in "$INPUT_DIR"/*/; do
    LABEL=$(basename "$LABEL_PATH")

    mkdir -p "$OUTPUT_DIR/$LABEL"

    # 8 augmentations per video:
    #   1. Horizontal Flip (mirrored gesture)
    #   2. Slow down 20% (helps model see details)
    #   3. Speed up 20%
    #   4. HFlip + Slow
    #   5. HFlip + Fast
    #   6. Zoom in 15% (simulate closer signer)
    #   7. Shift left 10% (camera offset)
    #   8. Shift right 10% (camera offset)
    for VIDEO in "$LABEL_PATH"*.mp4; do
        BASENAME=$(basename "$VIDEO" .mp4)

        echo "Processing $VIDEO..."

        # 1. Horizontal Flip
        ffmpeg -y -i "$VIDEO" \
            -vf "hflip" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_hflip.mp4"
        
        # 2. Slow down 20%
        ffmpeg -y -i "$VIDEO" \
            -filter:v "setpts=1.2*PTS" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_slow.mp4"
        
        # 3. Speed up 20%
        ffmpeg -y -i "$VIDEO" \
            -filter:v "setpts=0.8*PTS" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_fast.mp4"
        
        # 4. Horizontal Flip + Slow down
        ffmpeg -y -i "$VIDEO" \
            -filter:v "hflip,setpts=1.2*PTS" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_hflip_slow.mp4"
        
        # 5. Horizontal Flip + Speed up
        ffmpeg -y -i "$VIDEO" \
            -filter:v "hflip,setpts=0.8*PTS" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_hflip_fast.mp4"
        
        # 6. Slight zoom in (center crop + scale)
        ffmpeg -y -i "$VIDEO" \
            -vf "crop=iw/1.18:ih/1.18:(iw-iw/1.18)/2:(ih-ih/1.18)/2,scale=1280:720" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_zoom_in.mp4"

        # 7. Horizontal shift left
        ffmpeg -y -i "$VIDEO" \
            -vf "crop=iw*0.9:ih:iw*0.9:0,scale=iw*1.11:ih*1.0" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_shift_left.mp4"
        
        # 8. Horizontal shift right
        ffmpeg -y -i "$VIDEO" \
            -vf "crop=iw*0.9:ih:0:0,scale=iw*1.11:ih*1.0" \
            "$OUTPUT_DIR/$LABEL/${BASENAME}_shift_right.mp4"
    done
done

echo "Augmentasi selesai."
