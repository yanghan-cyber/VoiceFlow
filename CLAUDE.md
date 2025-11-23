# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a voice input method application that provides real-time speech-to-text functionality with global hotkey support. The application supports both streaming and offline ASR modes using Sherpa-ONNX models.

### Core Architecture

The application follows a modular design with four main components:

1. **ASR Engine** (`src/asr/`) - Abstract speech recognition interface with multiple implementations:
   - `ASRBase` abstract class defining the interface
   - `ASRFactory` for creating configured ASR engines
   - `SherpaOnnxASR` for Paraformer models
   - `SherpaSenseVoiceASR` for SenseVoice models

2. **Audio System** (`src/audio/`) - Audio capture and processing:
   - `AudioRecorder` handles microphone input using sounddevice
   - Thread-based audio streaming with configurable chunk sizes

3. **Input System** (`src/hotkeys/`) - Global hotkey management:
   - `HotkeyManager` with debouncing to prevent key state conflicts
   - Support for press, release, and long-press events

4. **Text Output** (`src/utils/`) - Smart text injection:
   - `TextTyper` provides intelligent streaming text updates
   - Calculates common prefixes to minimize backspace/delete operations

### Key Features

- **Dual Mode Operation**:
  - Stream mode: Real-time recognition with live text updates
  - Offline mode: Record-then-transcribe workflow
- **Global Hotkey**: F2 long-press to start/stop recording
- **Smart Text Injection**: Incremental updates during streaming
- **Multiple ASR Backends**: Paraformer and SenseVoice model support
- **Punctuation Restoration**: Optional punctuation model integration
- **Hotword Support**: Custom dictionary and replacement rules

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download required models (see download.md)
# Models should be placed in ckpts/ directory
```

### Configuration
- Main configuration: `config.yaml`
- Mode selection: `app.mode` (stream/offline)
- ASR engine: `asr.active_engine`
- Model paths: Update paths in config to match ckpts/ directory structure

### Running the Application
```bash
python src/main.py
```

### Testing ASR Components
The `examples/` directory contains standalone scripts for testing:
- `examples/streaming-paraformer-asr-microphone.py` - Test Paraformer streaming
- `examples/simulate-streaming-sense-voice-microphone.py` - Test SenseVoice
- `examples/offline-sense-voice-ctc-decode-files-with-hr.py` - Test offline recognition

## Development Notes

### Model Configuration
- Models are configured via YAML files in `config.yaml`
- Two primary ASR engines: Paraformer (streaming) and SenseVoice (multilingual)
- Punctuation model is optional but recommended for better output quality
- VAD (Voice Activity Detection) configuration affects endpoint detection

### Audio Processing
- Default sample rate: 16kHz
- Audio chunks: 100ms duration by default
- Float32 audio format required by ASR engines

### Threading Model
- Main thread: Hotkey detection and UI updates
- Audio thread: Microphone capture via sounddevice callback
- Processing thread: ASR inference and text generation
- All thread communication uses thread-safe patterns

### Hotkey Implementation
The hotkey system includes sophisticated debouncing to handle conflicts between:
- Key repeat events from OS
- Simulated keystrokes from text injection
- Physical key release detection

### Text Injection Strategy
The `TextTyper` class implements an optimization that:
1. Calculates common prefix between current and new text
2. Only deletes differing characters (minimal backspaces)
3. Only types new characters (minimal typing)
4. Provides smooth real-time updates during streaming

## Troubleshooting

### Common Issues
- **Permission errors**: May require admin privileges for global hotkeys on Linux
- **Model not found**: Check paths in config/models.yaml match ckpts/ directory
- **Audio issues**: Verify microphone permissions and sample rate compatibility
- **Hotkey conflicts**: Ensure F2 is not used by other applications

### Debug Configuration
Set `debug: true` in ASR configuration sections for detailed logging.