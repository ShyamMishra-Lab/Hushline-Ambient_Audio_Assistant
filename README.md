# Hushline Ambient Audio Assistant

Automatically pauses active system media when speech is detected and resumes playback after silence using a neural Voice Activity Detector (VAD).

## Installation

Ensure you have system dependencies installed:
* Linux: `sudo apt-get install -y libportaudio2 playerctl`
* Windows: None (uses built-in APIs)

Install dependencies using uv:
```bash
uv pip install -e .
```

## Running

Start the assistant:
```bash
uv run hushline
```

The application automatically downloads and caches the 1.8MB neural VAD model on its first run.

## Packaging for Windows

To build a standalone Windows executable:
```bash
uv pip install pyinstaller
pyinstaller hushline.spec
```
The packaged file will be generated in `dist/Hushline.exe`.
