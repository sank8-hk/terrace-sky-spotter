# 🔭 Terrace Sky Spotter

Built for the **Hacktoberfest Week 1 Challenge: Touch Grass** (DEV `#hf26challenge`).

Getting people off the screen and under the sky: an offline stargazing guide for
Bengaluru terraces. It computes what's actually above the horizon *right now* —
real altitude/azimuth math on this device — then a local open-weight Gemma model
narrates the tour and tells you star-hop by star-hop how to find each constellation.

## How it works

- **Deterministic astronomy (code):** 16 anchor stars from open catalogs
  (Yale Bright Star Catalogue / HYG, public domain) + Julian-date → sidereal-time
  math picks constellations above 15° for 12.97°N 77.59°E. No GPS, no network.
- **Moon phase + washout filter:** a synodic model computes illumination; the
  naked-eye limit slides from ~mag 5.5 (new moon) to ~mag 2.3 (full), and faint
  targets get a 🌕 washed-out badge. Validated against the Mar 3 2026 lunar eclipse.
- **Planets:** low-precision orbital math (good to ~1–2°, naked-eye grade) adds
  Venus, Mars, Jupiter and Saturn — verified by the Venus-elongation invariant
  (never more than 47° from the Sun).
- **Time picker:** plan ahead for any date/time, not just right now.
- **Open-weight narration (local Gemma 3 1B via Ollama):** the night's tour,
  myths, and per-constellation finder guides. The model never states positions —
  code owns *what's up and where*; the model only narrates *how to find* it.
- **Stack:** vanilla HTML/JS + zero-dependency Python server (`app.py`, stdlib only).

## Why open matters here

1. **Works with no signal.** Terrace, trail, village — the catalog is bundled data
   and the model runs on-device. A closed API guide dies the moment bars disappear,
   exactly where you need it most.
2. **Location stays yours.** Stargazing needs your coordinates; here they never
   leave the laptop.
3. **Free + swappable.** `OLLAMA_MODEL` swaps Gemma 3 1B for anything else tomorrow.

## Run it

```bash
# 1. Install Ollama: https://ollama.com
ollama pull gemma3:1b

# 2. No pip install needed — stdlib only
python3 app.py
# Terrace Sky Spotter on http://localhost:8002

# 3. Open http://localhost:8002 → tonight's table → "Narrate my tour" → go outside 🌌
```

Env overrides: `PORT`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_NUM_GPU`
(default `0` = CPU inference, most portable).

## License

MIT — see [LICENSE](LICENSE).
