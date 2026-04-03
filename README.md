# Parliament Advisor AI

Real-time parliamentary audio monitor. Captures microphone or livestream audio, streams to Gemini 3.1 Flash Live for analysis, and pushes AT_RISK alerts to Google Chat when potential rule violations are detected.

---

## What it does

- **Mic mode**: Captures your microphone and streams audio to Gemini Live in real time
- **Livestream mode**: Pipes a YouTube/stream URL through FFmpeg to Gemini Live
- **AT_RISK detection**: Gemini monitors for speech that may violate parliamentary rules or constitution, responding only when risk is detected
- **Google Chat alerts**: AT_RISK responses are pushed to a shared Google Chat Space within seconds
- **Knowledge base**: Gemini cites from 10 Thai parliamentary/legal PDFs embedded in its system prompt

---

## Architecture

```
Mic or Livestream → core.py (audio capture) → Gemini Live API (analysis)
                                                     ↓
                                            AT_RISK responses
                                                     ↓
                         main.py (CLI / orchestration) → push.py (Google Chat webhook)
```

| File | Purpose |
|---|---|
| `config.py` | Loads `.env`, `system_prompt.txt`, extracts text from PDFs in `knowledge/` |
| `core.py` | `GeminiSession` (WebSocket session manager) + `capture_mic()` + `capture_livestream(url)` |
| `push.py` | Google Chat webhook with exponential backoff retry + 1 msg/sec rate limiting |
| `main.py` | CLI entry point, argparse, signal handling, reconnection loop |

---

## Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/) (works on Windows, macOS, and Linux)
- **Gemini API key** ([ai.google.dev](https://ai.google.dev)) — free tier sufficient for testing
- **FFmpeg** — for livestream mode
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`
- **PortAudio** — for mic mode
  - **Windows**: Usually included with `sounddevice` pip package
  - **macOS**: `brew install portaudio`
  - **Linux**: `sudo apt install libportaudio2`

### Google Chat Space (optional for testing)

1. Create or open a Google Chat Space
2. Space settings → Apps & Integrations → Webhooks → Add Webhook → copy URL
3. Add the URL to your `.env` as `GOOGLE_CHAT_WEBHOOK_URL`

---

## Installation

### Step 1: Get the code

**macOS/Linux:**
```bash
git clone git@github.com:nerometa/parliament-advisor-ai.git
cd parliament-advisor-ai
```

**Windows (Command Prompt):**
```cmd
git clone https://github.com/nerometa/parliament-advisor-ai.git
cd parliament-advisor-ai
```

### Step 2: Create a virtual environment

This keeps project dependencies separate from your other Python projects.

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*Note: On Windows, if you get an execution policy error in PowerShell, use Command Prompt instead, or run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`*

### Step 3: Install Python packages

```bash
pip install -r requirements.txt
```

### System dependencies (install once per computer)

**macOS (using Homebrew):**
```bash
brew install ffmpeg portaudio
```

**Windows:**
- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)
- Extract and add `bin` folder to your PATH, or install via: `winget install ffmpeg`
- PortAudio is usually included automatically

**Ubuntu/Debian Linux:**
```bash
sudo apt update && sudo apt install -y libportaudio2 ffmpeg
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/...
GEMINI_MODEL=gemini-3.1-flash-live-preview
SYSTEM_PROMPT_PATH=system_prompt.txt
KNOWLEDGE_DIR=knowledge
```

---

## Usage

### Understanding the modes

This program works like a smart listening assistant. You can tell it to listen to either:
- **Your microphone** — for analyzing live conversations or recordings
- **A YouTube/stream URL** — for analyzing live parliamentary broadcasts

It then sends the audio to Google's Gemini AI, which checks if anything said might break parliamentary rules.

---

### Method 1: Test without alerts (recommended for first-timers)

Use this to make sure everything works before connecting to Google Chat.

#### Option A: Listen to your microphone

**What this does:** Records your voice and analyzes it in real-time.

**macOS/Linux:**
```bash
python main.py --mode mic --dry-run
```

**Windows:**
```cmd
python main.py --mode mic --dry-run
```

**To stop:** Press `Ctrl+C` (or `Cmd+C` on Mac)

---

#### Option B: Listen to a YouTube livestream

**What this does:** Takes audio from a YouTube video/stream and analyzes it.

**macOS/Linux:**
```bash
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..." --dry-run
```

**Windows:**
```cmd
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..." --dry-run
```

**To stop:** Press `Ctrl+C`

**Tip:** You'll see output in your terminal like:
```
Listening... Speak now or play your audio.
[INFO] Audio chunk sent
```

If Gemini detects a potential rule violation, you'll see:
```
🔴 AT_RISK — 10:43:12
สถานะ: AT_RISK
ข้อบังคับที่เกี่ยวข้อง: ...
```

---

### Method 2: Send alerts to Google Chat

Once you've tested with `--dry-run` and everything works:

**macOS/Linux:**
```bash
# Microphone mode
python main.py --mode mic

# Livestream mode  
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..."
```

**Windows:**
```cmd
:: Microphone mode
python main.py --mode mic

:: Livestream mode
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..."
```

Now when Gemini detects a rule violation, it will send a message to your Google Chat Space instantly.

---

### Quick reference: all command options

| Flag | What it does | Required? |
|---|---|---|
| `--mode mic` or `--mode livestream` | Choose audio source | **Yes** |
| `--url "URL"` | Paste a YouTube/stream link | Only for livestream mode |
| `--dry-run` | Show alerts in terminal only (no Google Chat) | No |
| `--verbose` or `-v` | Show extra debug info | No |

---

### Example commands

```bash
# Test microphone with extra details shown
python main.py --mode mic --dry-run --verbose

# Monitor a Thai parliamentary livestream
python main.py --mode livestream --url "https://www.youtube.com/watch?v=EXAMPLE" --dry-run

# Live monitoring with Google Chat alerts
python main.py --mode mic
```

---

## Knowledge Base

The `knowledge/` directory contains 10 Thai parliamentary reference PDFs. Their text is automatically extracted and embedded into Gemini's system prompt at startup (~340K characters).

PDFs included:
- Constitution of Thailand (รัฐธรรมนูญ 60)
- House of Representatives Meeting Regulations 2562
- Organic Act on Counter Corruption (พ.ร.บ.ปปง.)
- State Audit Office Act (พ.ร.บ.สตง.)
- Parliamentary procedures, written questions (กระทู้ถาม), motions (ญัตติ)
- Parliamentary guidelines for MPs

---

## AT_RISK Response Format

When Gemini detects potential rule violation, it responds in this format:

```
สถานะ: AT_RISK
ข้อบังคับที่เกี่ยวข้อง: (regulation reference)
เหตุผล: (brief explanation)
```

Example in Google Chat:

```
🔴 AT_RISK — 10:43:12

สถานะ: AT_RISK
ข้อบังคับที่เกี่ยวข้อง: ข้อบังคับการประชุม ข้อ 45
เหตุผล: อาจเข้าข่ายการใช้ถ้อยคำเสียดสีต่อสมาชิกในห้องประชุม
```

---

## Troubleshooting

**WebSocket 1011 error**: Your API key may not have access to Gemini 3.1 Flash Live in this region/server environment. Test on your local machine with your own key.

**PortAudio not found / `ImportError: No module named '_portaudio'`**
- **Windows**: Usually fixed by `pip install sounddevice` (PortAudio is bundled)
- **macOS**: `brew install portaudio` then `pip install --force-reinstall sounddevice`
- **Linux**: `sudo apt install libportaudio2`

**FFmpeg not found**
- **Windows**: Install via `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

**"'python' is not recognized" (Windows)**
- Use `py` instead of `python`, or use the full path to your Python installation
- Make sure you checked "Add Python to PATH" during installation

**Virtual environment won't activate (Windows)**
- Try Command Prompt instead of PowerShell
- Or run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell (as Administrator)

**No audio chunks sent**: Check your microphone permissions and that `sounddevice` can list devices with `python -c "import sounddevice; print(sounddevice.query_devices())"`

---

## Development

```bash
# Run tests (when added)
pytest tests/ -v

# Verify all imports
python -c "import config, core, push, main"
```
