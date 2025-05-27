import streamlit as st
import numpy as np
import tempfile
import cv2
import mediapipe as mp


# Pose extraction + normalization code
def detect_swing_start_end(frames, keypoint_index=16, speed_threshold=0.01):
    positions = np.array([frame[keypoint_index] for frame in frames])
    diffs = np.diff(positions, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)

    start = next((i for i, v in enumerate(speeds) if v > speed_threshold), 0)
    end = next(
        (i for i in range(start + 5, len(speeds)) if speeds[i] < speed_threshold),
        len(speeds) - 1,
    )
    return frames[start : end + 1]


def extract_keypoints_from_video(video_path, num_frames=32):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y) for lm in landmarks]
            frames.append(keypoints)

    cap.release()
    pose.close()
    if len(frames) == 0:
        return None

    frames = detect_swing_start_end(frames, keypoint_index=16)
    if len(frames) < 2:
        return None

    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = [frames[i] for i in idx]
    return sampled


def normalize_pose(pose):
    pose = np.array(pose)
    center = pose[0]
    pose -= center
    scale = np.linalg.norm(pose).mean()
    if scale > 0:
        pose /= scale
    return pose.flatten()


def get_flattened_normalized_vector(video_path):
    poses = extract_keypoints_from_video(video_path)
    if poses is None:
        return None
    norm_poses = [normalize_pose(pose) for pose in poses]
    return np.concatenate(norm_poses)


# Streamlit UI
st.title("🏏 Swing Similarity Checker")
st.write("Upload your swing and a pro's swing to compare!")

# User swing
uploaded_user = st.file_uploader("Upload YOUR swing video", type=["mp4"], key="user")
# Pro swing
uploaded_pro = st.file_uploader("Upload PRO swing video", type=["mp4"], key="pro")

if uploaded_user and uploaded_pro:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_user:
        tmp_user.write(uploaded_user.read())
        user_video_path = tmp_user.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_pro:
        tmp_pro.write(uploaded_pro.read())
        pro_video_path = tmp_pro.name

    vec_user = get_flattened_normalized_vector(user_video_path)
    vec_pro = get_flattened_normalized_vector(pro_video_path)

    if vec_user is not None and vec_pro is not None:
        similarity = np.dot(vec_user, vec_pro) / (
            np.linalg.norm(vec_user) * np.linalg.norm(vec_pro)
        )
        st.metric("Similarity Score", f"{similarity * 100:.2f}%")
    else:
        st.error("Could not process one of the videos. Try again with clearer swings.")
