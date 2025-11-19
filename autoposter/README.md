# SharpsSignal AutoPoster (Local Prototype)

This script:
1. Reads your daily picks from a CSV
2. Selects picks within your sweet-spot window (15–60 minutes before game time, configurable)
3. Renders a clean 1080×1920 vertical card image
4. Posts to **X (Twitter)** automatically (other platforms stubbed for now)
5. Logs results to the console and saves assets.

## Quick start
```bash
cd autoposter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys and settings
python post_daily_picks.py --once
