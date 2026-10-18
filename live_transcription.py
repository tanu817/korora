
from __future__ import annotations

import argparse
import os
import re
import signal
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from faster_whisper import WhisperModel

from feeder import config as feeder_config
from feeder.flux_hands import FluxHands, FluxSettings
from feeder.interfaces import Hands


def parse_camera_value(value: str) -> str | int:
    """Keep numeric camera indices numeric for LeRobot/OpenCV."""
    return int(value) if value.isdigit() else value


# ============================================================
# NVIDIA DLL setup (Windows)
# Kept from the original project.
# Whisper will still run on CPU.
# ============================================================

_DLL_HANDLES = []


def add_nvidia_dll_path(package) -> None:
    package_path = Path(next(iter(package.__path__)))
    bin_path = package_path / "bin"

    if not bin_path.exists():
        raise RuntimeError(
            f"NVIDIA DLL directory does not exist: {bin_path}"
        )

    _DLL_HANDLES.append(os.add_dll_directory(str(bin_path)))
    os.environ["PATH"] = (
        str(bin_path) + os.pathsep + os.environ.get("PATH", "")
    )


def setup_nvidia_dlls() -> None:
    import nvidia.cublas
    import nvidia.cudnn
    import nvidia.cuda_nvrtc

    for package in (
        nvidia.cublas,
        nvidia.cudnn,
        nvidia.cuda_nvrtc,
    ):
        add_nvidia_dll_path(package)


# ============================================================
# Configuration
# ============================================================

class Config:
    # A model identifier is downloaded by faster-whisper from its model
    # registry and cached by the library, rather than loaded from this repo.
    MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_DURATION = 0.1

    # Keep this much recent audio for each transcription.
    WINDOW_SECONDS = 5.0

    # Start another transcription this often.
    # Smaller values mean more frequent checks, but more CPU use.
    TRANSCRIBE_INTERVAL = 1.0

    # Avoid repeatedly firing on the same detected phrase.
    COMMAND_COOLDOWN = 2.0

    # Require a recognized command to be detected in a recent window.
    MIN_AUDIO_SECONDS = 1.0

    LANGUAGE = "en"

    # Keep diagnostic output manageable.
    PRINT_TRANSCRIPTIONS = True

    # Keep this enabled until the cameras, calibration, policy and scene have
    # been checked. Set FEEDER_DRY_RUN=false only on the robot computer.
    FEEDER_DRY_RUN = (
        os.environ.get("FEEDER_DRY_RUN", "true").lower()
        not in {"0", "false", "no"}
    )
    FEEDER_BITES = max(1, int(os.environ.get("FEEDER_BITES", "1")))
    FLUX_POLICY_PATH = os.environ.get(
        "FLUX_POLICY_PATH",
        "models/flux-3-action-so101",
    )
    FLUX_USE_PEFT = (
        os.environ.get("FLUX_USE_PEFT", "false").lower()
        in {"1", "true", "yes"}
    )
    FLUX_ROBOT_PORT = os.environ.get("FLUX_ROBOT_PORT", "/dev/ttyACM0")
    FLUX_SCENE_CAMERA = parse_camera_value(
        os.environ.get("FLUX_SCENE_CAMERA", "0")
    )
    FLUX_WRIST_CAMERA = parse_camera_value(
        os.environ.get("FLUX_WRIST_CAMERA", "1")
    )


# ============================================================
# Modes and commands
# ============================================================

class RobotMode(Enum):
    STANDBY = "standby"
    FEEDING = "feeding"


class VoiceCommand(Enum):
    START_FEEDING = "start_feeding"
    STOP_FEEDING = "stop_feeding"


FEEDING_PHRASES = (
    "i want food",
    "feed me",
    "give me food"
)

STANDBY_PHRASES = (
    "stop",
    "i am done",
    "im done",
    "its enough",
    "it is enough",
)


def normalize_text(text: str) -> str:
    """Normalize punctuation and contractions for command matching."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = text.replace("'", "")
    return " ".join(text.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Match a complete phrase, not a substring inside another word."""
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)

    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_phrase)}"
            rf"(?![a-z0-9])",
            normalized_text,
        )
    )


class CommandDetector:
    """Detect voice commands inside longer transcription results."""

    def detect(self, transcript: str) -> VoiceCommand | None:
        # Standby takes priority if a transcript contains both types.
        if any(
            contains_phrase(transcript, phrase)
            for phrase in STANDBY_PHRASES
        ):
            return VoiceCommand.STOP_FEEDING

        if any(
            contains_phrase(transcript, phrase)
            for phrase in FEEDING_PHRASES
        ):
            return VoiceCommand.START_FEEDING

        return None


# ============================================================
# Robot controller
# ============================================================

class CommandController:
    """
    Owns the current robot mode.

    Replace the placeholder callbacks with hardware integrations
    when the feeding mechanism is ready.
    """

    def __init__(
        self,
        on_feeding_started: Callable[[], None] | None = None,
        on_standby_started: Callable[[], None] | None = None,
    ) -> None:
        self.mode = RobotMode.STANDBY
        self.on_feeding_started = on_feeding_started
        self.on_standby_started = on_standby_started
        self._lock = threading.Lock()

    def print_mode(self) -> None:
        print(f"Current mode: {self.mode.value.upper()}")

    def process(self, command: VoiceCommand) -> bool:
        """
        Apply a command.

        Returns True only if the command changes the current mode.
        Repeated commands do not restart the same mode.
        """
        with self._lock:
            if command == VoiceCommand.START_FEEDING:
                if self.mode == RobotMode.FEEDING:
                    return False

                self.mode = RobotMode.FEEDING
                callback = self.on_feeding_started

            elif command == VoiceCommand.STOP_FEEDING:
                if self.mode == RobotMode.STANDBY:
                    return False

                self.mode = RobotMode.STANDBY
                callback = self.on_standby_started

            else:
                return False

        print(f"\nMode changed: {self.mode.value.upper()}")

        if callback is not None:
            try:
                callback()
            except Exception as exc:
                print(f"Robot callback failed: {exc}")

        return True


class FeedingSequence:
    """Sequences the configured feeding steps through the FLUX hands."""

    def __init__(
        self,
        hands: Hands,
        bites: int,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self.hands = hands
        self.bites = bites
        self.on_finished = on_finished
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="feeding-sequence",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.hands.stop()

    def _do(self, instruction_key: str) -> bool:
        if self._stop_event.is_set():
            return False

        instruction = feeder_config.INSTRUCTIONS[instruction_key]
        print(f"Feeding step: {instruction}")
        self.hands.do(instruction)
        return not self._stop_event.is_set()

    def _run(self) -> None:
        try:
            if not self._do("pick_spoon"):
                return

            for bite in range(1, self.bites + 1):
                print(f"Starting bite {bite}/{self.bites}.")
                if not self._do("scoop"):
                    return
                if not self._do("to_mouth"):
                    return
                if not self._do("back_to_rest"):
                    return

            self._do("place_spoon")
        except Exception as exc:
            print(f"Feeding sequence failed: {exc}")
        finally:
            if self.on_finished is not None:
                self.on_finished()


def run_print_test() -> None:
    """Exercise command detection and print the complete dry-run sequence."""
    print("Running safe transcription integration test.")
    print("No microphone, Whisper model, or robot will be used.\n")

    detector = CommandDetector()
    test_phrases = (
        "feed me",
        "please stop",
        "unrecognized phrase",
    )

    for phrase in test_phrases:
        command = detector.detect(phrase)
        print(f"Detector: {phrase!r} -> {command.value if command else 'none'}")

    hands = FluxHands(
        FluxSettings(
            policy_path=Config.FLUX_POLICY_PATH,
            use_peft=Config.FLUX_USE_PEFT,
            port=Config.FLUX_ROBOT_PORT,
            scene_cam=Config.FLUX_SCENE_CAMERA,
            wrist_cam=Config.FLUX_WRIST_CAMERA,
            confirm_each=False,
            dry_run=True,
        )
    )
    sequence = FeedingSequence(hands, Config.FEEDER_BITES)
    sequence.start()

    if sequence._thread is not None:
        sequence._thread.join()

    print("\nSafe integration test completed.")


# ============================================================
# Continuous audio capture
# ============================================================

class AudioBuffer:
    """Thread-safe rolling buffer of microphone audio."""

    def __init__(
        self,
        sample_rate: int,
        window_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * window_seconds)
        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0
        self._lock = threading.Lock()

    def add(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        with self._lock:
            self._chunks.append(audio.copy())
            self._sample_count += len(audio)

            while (
                self._sample_count > self.max_samples
                and self._chunks
            ):
                excess = self._sample_count - self.max_samples
                first = self._chunks[0]

                if len(first) <= excess:
                    self._chunks.popleft()
                    self._sample_count -= len(first)
                else:
                    self._chunks[0] = first[excess:]
                    self._sample_count -= excess

    def get_audio(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            return np.concatenate(tuple(self._chunks))

    def duration(self) -> float:
        with self._lock:
            return self._sample_count / self.sample_rate


class MicrophoneReader:
    """Reads microphone blocks continuously in a background thread."""

    def __init__(
        self,
        audio_buffer: AudioBuffer,
        sample_rate: int,
        block_duration: float,
    ) -> None:
        self.audio_buffer = audio_buffer
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.stream: sd.InputStream | None = None
        self.error: Exception | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"\nMicrophone warning: {status}")

        # Convert int16 PCM samples to float32 in [-1, 1].
        audio = indata[:, 0].astype(np.float32) / 32768.0
        self.audio_buffer.add(audio)

    def start(self) -> None:
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._callback,
        )

        self.stream.start()
        print("Microphone started.")

    def stop(self) -> None:
        self.stop_event.set()

        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


# ============================================================
# Transcription
# ============================================================

class SpeechTranscriber:
    """Wraps Faster-Whisper behind a small reusable interface."""

    def __init__(self, config: type[Config]) -> None:
        print(f"Loading Whisper model: {config.MODEL_NAME}")

        self.model = WhisperModel(
            config.MODEL_NAME,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        self.language = config.LANGUAGE

        print("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        # Segment generation is lazy, so consume it here.
        return " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()


# ============================================================
# Application
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Korora voice control")
    parser.add_argument(
        "--test",
        action="store_true",
        help="print a safe integration test without microphone or robot use",
    )
    args = parser.parse_args()

    if args.test:
        run_print_test()
        return

    if Config.DEVICE == "cuda":
        setup_nvidia_dlls()

    hands = FluxHands(
        FluxSettings(
            policy_path=Config.FLUX_POLICY_PATH,
            use_peft=Config.FLUX_USE_PEFT,
            port=Config.FLUX_ROBOT_PORT,
            scene_cam=Config.FLUX_SCENE_CAMERA,
            wrist_cam=Config.FLUX_WRIST_CAMERA,
            dry_run=Config.FEEDER_DRY_RUN,
        )
    )
    feeding_sequence = FeedingSequence(hands, Config.FEEDER_BITES)

    audio_buffer = AudioBuffer(
        sample_rate=Config.SAMPLE_RATE,
        window_seconds=Config.WINDOW_SECONDS,
    )

    transcriber = SpeechTranscriber(Config)
    detector = CommandDetector()

    controller = CommandController(
        on_feeding_started=feeding_sequence.start,
        on_standby_started=feeding_sequence.stop,
    )
    feeding_sequence.on_finished = (
        lambda: controller.process(VoiceCommand.STOP_FEEDING)
    )

    stop_event = threading.Event()

    def request_shutdown(signum=None, frame=None) -> None:
        stop_event.set()

    # Ctrl+C is also handled by the outer KeyboardInterrupt handler.
    signal.signal(signal.SIGINT, request_shutdown)

    reader = MicrophoneReader(
        audio_buffer=audio_buffer,
        sample_rate=Config.SAMPLE_RATE,
        block_duration=Config.BLOCK_DURATION,
    )

    last_transcription_time = 0.0
    last_command_time = 0.0
    last_command: VoiceCommand | None = None

    controller.print_mode()

    try:
        reader.start()
        print("Listening continuously. Speak a command at any time.")
        print("Press Ctrl+C to exit.\n")

        while not stop_event.is_set():
            now = time.monotonic()

            if (
                now - last_transcription_time
                < Config.TRANSCRIBE_INTERVAL
            ):
                stop_event.wait(0.05)
                continue

            last_transcription_time = now

            if audio_buffer.duration() < Config.MIN_AUDIO_SECONDS:
                continue

            audio = audio_buffer.get_audio()

            try:
                transcript = transcriber.transcribe(audio)
            except Exception as exc:
                print(f"\nTranscription error: {exc}")
                stop_event.wait(0.5)
                continue

            if not transcript:
                continue

            if Config.PRINT_TRANSCRIPTIONS:
                print(f"\nHeard: {transcript}")

            command = detector.detect(transcript)

            if command is None:
                continue

            now = time.monotonic()

            # Overlapping windows may recognize the same phrase
            # repeatedly. Suppress rapid repeats of the same command.
            if (
                command == last_command
                and now - last_command_time < Config.COMMAND_COOLDOWN
            ):
                continue

            changed = controller.process(command)

            last_command = command
            last_command_time = now

            if changed:
                print("Listening for the next command...\n")

    except KeyboardInterrupt:
        print("\nShutdown requested.")

    finally:
        reader.stop()
        print("Microphone closed.")
        print("Application stopped.")


if __name__ == "__main__":
    main()
