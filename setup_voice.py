"""Download the public, local speech models used by Jarvis."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

import requests


ROOT = Path(__file__).resolve().parent / "models"
FILES = [
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx", ROOT / "en_GB-alan-medium.onnx", None),
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json", ROOT / "en_GB-alan-medium.onnx.json", None),
    ("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", ROOT / "vosk-model-small-en-us-0.15.zip", None),
    ("https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.onnx", ROOT / "silero_vad.onnx", None),
    ("https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", ROOT / "face_detection_yunet_2023mar.onnx", 232589),
]


def download(item):
    url, path, expected_size = item
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return path
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 30)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected download size for {path.name}.")
    partial.replace(path)
    print(f"Downloaded {path.name}.", flush=True)
    return path


def main():
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, FILES))
    model = ROOT / "vosk-model-small-en-us-0.15"
    if not (model / "am" / "final.mdl").exists():
        with zipfile.ZipFile(FILES[2][1]) as archive:
            for item in archive.infolist():
                target = (ROOT / item.filename).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise RuntimeError("Speech model archive contains an invalid path.")
            archive.extractall(ROOT)
        print("Extracted the offline speech recognition model.", flush=True)
    print("Local voice models are ready.", flush=True)


if __name__ == "__main__":
    main()
