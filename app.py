
import io
import json

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Smart Classroom AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Smart Classroom AI")
st.write("Classroom Sound Classification")
st.caption("Teacher • Students • Noise")

# Model settings: keep consistent with training
SAMPLE_RATE = 16000
CLIP_DURATION = 5
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
TARGET_FRAMES = 501

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(
        "classroom_sound_model.keras"
    )

    with np.load("audio_preprocessing.npz") as data:
        mean = float(data["mean"])
        std = float(data["std"])

    with open("class_labels.json", "r") as f:
        labels = json.load(f)

    return model, mean, std, labels


model, mean, std, labels = load_resources()


def decode_audio(audio_bytes):
    """Read uploaded or recorded audio without manual conversion."""
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise ValueError("No audio samples were recorded.")

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    return audio


def split_audio(audio):
    """Split the complete recording into 5-second segments."""
    segment_samples = SAMPLE_RATE * CLIP_DURATION
    segments = []

    for start in range(0, len(audio), segment_samples):
        segment = audio[start:start + segment_samples]

        # Ignore an extremely short final fragment (< 1 second)
        if len(segment) < SAMPLE_RATE:
            continue

        # Pad the final segment if it is shorter than 5 seconds
        if len(segment) < segment_samples:
            segment = np.pad(
                segment,
                (0, segment_samples - len(segment))
            )

        segments.append((start / SAMPLE_RATE, segment))

    if not segments:
        raise ValueError(
            "Recording is too short. Please record at least 1 second."
        )

    return segments


def extract_features(audio):
    """Convert one 5-second segment into normalized MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Match the frame count used by the trained model
    if mfcc.shape[1] < TARGET_FRAMES:
        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, TARGET_FRAMES - mfcc.shape[1])
            )
        )
    else:
        mfcc = mfcc[:, :TARGET_FRAMES]

    # Use the same training-set normalization parameters
    features = (mfcc - mean) / (std + 1e-8)

    return features[np.newaxis, ..., np.newaxis].astype(
        np.float32
    )


def analyze_recording(audio_bytes):
    audio = decode_audio(audio_bytes)
    segments = split_audio(audio)

    results = []

    for start_time, segment in segments:
        features = extract_features(segment)

        expected_shape = tuple(model.input_shape[1:])

        if tuple(features.shape[1:]) != expected_shape:
            raise ValueError(
                f"Input shape mismatch: {features.shape[1:]} "
                f"vs {expected_shape}"
            )

        scores = model.predict(
            features,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(scores))

        results.append({
            "start": start_time,
            "end": start_time + CLIP_DURATION,
            "label": labels[predicted_index],
            "confidence": float(scores[predicted_index]),
            "scores": scores
        })

    return results


recording = st.audio_input("🎤 Record classroom sound")

if recording is not None:
    st.audio(recording)

    if st.button("Analyze All Segments", type="primary"):
        try:
            with st.spinner("Analyzing classroom sound..."):
                results = analyze_recording(
                    recording.getvalue()
                )



            # Average probabilities across all segments
            all_scores = np.array(
                [result["scores"] for result in results]
            )

            average_scores = all_scores.mean(axis=0)
            overall_index = int(np.argmax(average_scores))
            overall_label = labels[overall_index]

            # Count segment predictions
            counts = {
                label: sum(
                    result["label"] == label
                    for result in results
                )
                for label in labels
            }

            st.divider()
            st.subheader("Overall Analysis")

            st.success(
                f"Overall result: {overall_label}"
            )

            st.metric(
                "Average model confidence",
                f"{average_scores[overall_index] * 100:.2f}%"
            )

            st.write("Average class probabilities")

            for label, score in zip(labels, average_scores):
                st.write(
                    f"**{label}:** {score * 100:.2f}%"
                )

            st.write("Number of segments predicted per class")

            for label in labels:
                st.write(f"**{label}:** {counts[label]} segment(s)")

            st.caption(
                "Overall result is based on average class probabilities. "
                "Confidence is the model score, not a guarantee of accuracy."
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
