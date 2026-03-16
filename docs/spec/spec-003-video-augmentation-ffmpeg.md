Augmentasi data video memanfaatkan ffmpeg karena dapat menghandle multimedia dengan sangat efisien

Spesifikasi augmentasi: 
1. Brightness $+0.1$
2. Brightness $-0.1$
3. Horizontal Flip
4. Horizontal Flip + Brightness $+0.1$ 
5. Horizontal Flip + Brightness $-0.1$

Sehingga setiap video akan berjumlah 6 setelah proses augmentasi, yang mana cukup untuk menambah variabilitas data.
# Normalized Format
```sh
# 720p 24fps Video Only H.264
ffmpeg -y -i "$VIDEO" \
	-vf "scale-1280:720, fps=24" \
	-an -c:v libx264 \ 
	"$OUTPUT_DIR/$LABEL/${BASENAME}.mp4"
```
# Augmentation
```sh
BASENAME=$(basename "$VIDEO" .mp4)

# 1. Brightness +
ffmpeg -y -i "$VIDEO" \
	-vf "eq=brightness=0.1" \
	"$OUTPUT_DIR/$LABEL/${BASENAME}_bright_plus.mp4"

# 2. Brightness -
ffmpeg -y -i "$VIDEO" \
	-vf "eq=brightness=-0.1" \
	"$OUTPUT_DIR/$LABEL/${BASENAME}_bright_minus.mp4"

# 3. Horizontal Flip
ffmpeg -y -i "$VIDEO" \
	-vf "hflip" \
	"$OUTPUT_DIR/$LABEL/${BASENAME}_hflip.mp4"

# 4. Horizontal Flip and Brightness +
ffmpeg -y -i "$VIDEO" \
	-vf "hflip, eq=brightness=0.1" \
	"$OUTPUT_DIR/$LABEL/${BASENAME}_hflip_bright_plus.mp4"

# 5. Horizontal Flip and Brightness -
ffmpeg -y -i "$VIDEO" \
	-vf "hflip, eq=brightness=-0.1" \
	"$OUTPUT_DIR/$LABEL/${BASENAME}_hflip_bright_minus.mp4"
```


