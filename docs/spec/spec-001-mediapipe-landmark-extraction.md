Sumber dokumentasi utama: [AI Google Dev](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#video_1)

# Holistic recognition
Mediapipe versi terbaru cukup berbeda dengan versi yang sebelumnya. Perbedaan utama adalah versi terbaru menggunakan tasks dan memanfaatkan model yang sudah disediakan oleh Google

Model (task by mediapipe)
- [Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [Pose Landmarker]()

Contoh kode [Google Colab](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#scrollTo=Iy4r2_ePylIa)
[Dokumentasi Holistic](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md)

# Code Idea Analysis
Untuk mengekstraksi landmark holistik terdiri dari tangan, pose, dan wajah bisa menggunakan praktik yang sesuai dengan dokumentasi resmi dari google
## Install Library
```sh
pip install mediapipe opencv-python
```
Mediapipe versi 0.10.32
Opencv-python versi 4.13.0.92
## Load Video
```py
import cv2

cap        = cv2.VideoCapture(VIDEO_PATH)
```
## Initialize Tasks Model
```py
import mediapipe as mp

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
	base_options=mp.tasks.BaseOptions(POSE_MODEL_PATH),
	running_mode=mp.tasks.vision.RunningMode.VIDEO,
	num_poses=1,
	min_pose_detection_confidence=0.5,
	min_pose_presence_confidence=0.5,
	min_tracking_confidence=0.5,
)

face_options = mp.tasks.vision.FaceLandmarkerOptions(
	base_options=mp.tasks.BaseOptions(FACE_MODEL_PATH),
	running_mode=mp.tasks.vision.RunningMode.VIDEO,
	num_faces=1,
	min_face_detection_confidence=0.5,
	min_face_presence_confidence=0.5,
	min_tracking_confidence=0.5,
	output_face_blendshapes=False,
	output_facial_transformation_matrixes=False,
)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
	base_options=mp.tasks.BaseOptions(HAND_MODEL_PATH),
	running_mode=mp.tasks.vision.RunningMode.VIDEO,
	num_hands=2,
	min_hand_detection_confidence=0.5,
	min_hand_presence_confidence=0.5,
	min_tracking_confidence=0.5,
)
```
## Helper: Landmark to Array
```py
def landmarks_to_array(list_lm, extra_fields=None):
    base_fields = ['x', 'y', 'z']
    cols = base_fields + (extra_fields or [])
    arr = np.full((len(list_lm), len(cols)), np.nan, dtype=np.float32)
    for i, lm in enumerate(list_lm):
        for j, name in enumerate(cols):
            arr[i, j] = getattr(lm, name, np.nan)
    return arr
```
## Extract Landmark from Video
```py
with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, FaceLandmarker.create_from_options(face_options) as face_landmarker, HandLandmarker.create_from_options(hand_options) as hand_landmarker:

	while True:
		ok, frame_bgr = cap.read()
		if not ok:
			break

		# konversi BGR -> RGB (mediapipe pakai SRGB)
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

		mp_image = mp.Image(
			image_format=mp.ImageFormat.SRGB,
			data=frame_rgb
		)

		timestamp_ms = int(frame_idx * 1000.0 / fps)

		pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
		face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
		hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

		# ====== ambil landmark pose ======
		if getattr(pose_result, "pose_landmarks", None) and len(pose_result.pose_landmarks) > 0:
			pose_lm = pose_result.pose_landmarks[0]
			pose_arr = landmarks_to_array(pose_lm, extra_fields=["visibility"])
		else:
			pose_arr = np.full((N_POSE, 4), np.nan, dtype=np.float32)

		# ====== ambil landmark face ======
		if getattr(face_result, "face_landmarks", None) and len(face_result.face_landmarks) > 0:
			face_lm = face_result.face_landmarks[0]
			face_arr = landmarks_to_array(face_lm)  # x,y,z
		else:
			face_arr = np.full((N_FACE, 3), np.nan, dtype=np.float32)

		# ====== ambil landmark hand (bisa 0,1,2 tangan) ======
		frame_hands = np.full((2, N_HAND, 3), np.nan, dtype=np.float32)

		if getattr(hand_result, "hand_landmarks", None):
			# hand_result.hand_landmarks adalah list [hand][landmark]
			for hand_i, lm_list in enumerate(hand_result.hand_landmarks[:2]):
				hand_arr = landmarks_to_array(lm_list)  # x,y,z
				# kalau jumlah landmark kurang dari N_HAND, kita isi sebagian
				n = min(len(hand_arr), N_HAND)
				frame_hands[hand_i, :n, :] = hand_arr[:n, :]
```