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

- Python 3.10+
- Gemini API key ([ai.google.dev](https://ai.google.dev)) — free tier sufficient for testing
- FFmpeg (`sudo apt install ffmpeg`) — for livestream mode
- PortAudio (`sudo apt install libportaudio2`) — for mic mode

### Google Chat Space (optional for testing)

1. Create or open a Google Chat Space
2. Space settings → Apps & Integrations → Webhooks → Add Webhook → copy URL
3. Add the URL to your `.env` as `GOOGLE_CHAT_WEBHOOK_URL`

---

## Installation

```bash
git clone git@github.com:nerometa/parliament-advisor-ai.git
cd parliament-advisor-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### System dependencies (once per machine)

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

### Dry run (no webhook, terminal output only)

```bash
# Microphone mode
python main.py --mode mic --dry-run

# Livestream mode
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..." --dry-run
```

### With Google Chat alerts

```bash
python main.py --mode mic
python main.py --mode livestream --url "https://www.youtube.com/watch?v=..."
```

### Options

| Flag | Description |
|---|---|
| `--mode mic\|livestream` | Audio source (required) |
| `--url URL` | Livestream URL (required for livestream mode) |
| `--dry-run` | Print alerts to stdout instead of sending to Google Chat |
| `--verbose, -v` | Enable debug-level logging |

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

**PortAudio not found**: Run `sudo apt install libportaudio2`

**FFmpeg not found**: Run `sudo apt install ffmpeg`

**No audio chunks sent**: Check your microphone permissions and that `sounddevice` can list devices with `python -c "import sounddevice; print(sounddevice.query_devices())"`

---

## Development

```bash
# Run tests (when added)
pytest tests/ -v

# Verify all imports
python -c "import config, core, push, main"
```
