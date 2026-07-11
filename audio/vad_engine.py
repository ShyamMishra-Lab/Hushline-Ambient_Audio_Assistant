import os
import urllib.request
import numpy as np
import onnxruntime as ort
import config


class VADEngine:
    """
    Neural Voice Activity Detector using Silero VAD v5 ONNX model.
    Effectively rejects non-speech sounds like keyboard clicks, hums, and music.
    """

    def __init__(self, aggressiveness=2):
        self.aggressiveness = aggressiveness

        model_dir = os.path.expanduser("~/.cache/hushline")
        self.model_path = os.path.join(model_dir, "silero_vad.onnx")
        self._ensure_model_exists()

        # Load the ONNX model using CPUExecutionProvider
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.reset_state()

    def reset_state(self):
        # Silero VAD v5 expects an LSTM state of shape [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # Context window size is 64 samples for 16kHz
        self._context = np.zeros((1, 64), dtype=np.float32)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            urls = [
                "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
                "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/silero_vad.onnx",
                "https://huggingface.co/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx",
            ]
            success = False
            last_err = None
            for url in urls:
                print(
                    f"Downloading Silero VAD ONNX model (~1.8MB) from {url}...",
                    flush=True,
                )
                try:
                    urllib.request.urlretrieve(url, self.model_path)
                    print("Download complete.", flush=True)
                    success = True
                    break
                except Exception as e:
                    print(f"Failed to download: {e}", flush=True)
                    last_err = e
            if not success:
                print(
                    "Error: All download attempts for Silero VAD model failed.",
                    flush=True,
                )
                raise last_err

    def detect_speech(self, indata, volume_threshold):
        """
        Main method called every frame from the audio callback.
        """
        audio_float = indata[:, 0]

        # --- GATE 1: volume floor check ---
        # Skip neural inference only on dead silence to optimize CPU usage
        rms = np.sqrt(np.mean(audio_float**2)) * 1000
        if rms < 2.0:  # Very low threshold to filter out absolute silence only
            return False

        # Ensure correct window size (512 samples)
        if len(audio_float) != 512:
            if len(audio_float) < 512:
                audio_float = np.pad(
                    audio_float, (0, 512 - len(audio_float)), "constant"
                )
            else:
                audio_float = audio_float[:512]

        # --- GATE 2: Peak Normalization for low-volume audio ---
        # Scale frames with audio activity to standard training levels (0.55 peak)
        max_val = np.max(np.abs(audio_float))
        if max_val > 0.003:  # Only scale if above baseline noise gate
            audio_float = audio_float * (0.55 / max_val)

        audio_input = np.expand_dims(audio_float, axis=0).astype(np.float32)

        # Silero VAD v5 requires appending a 64-sample context before the 512-sample frame
        x = np.concatenate([self._context, audio_input], axis=1)

        inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(config.SAMPLE_RATE, dtype=np.int64),
        }

        try:
            outputs = self.session.run(None, inputs)
            speech_prob = outputs[0][0][0]
            self._state = outputs[1]  # Save RNN state for next frame
            self._context = x[:, -64:]  # Save last 64 samples as context for next frame

            # Binary speech trigger threshold (>= 0.5 indicates human speech)
            return speech_prob >= 0.5
        except Exception as e:
            print("VAD inference error:", e, flush=True)
            return False