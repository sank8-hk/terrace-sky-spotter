#!/usr/bin/env python3
"""Terrace Sky Spotter — offline stargazing guide for Bengaluru skies.

Deterministic astronomy (no cloud, no GPS): bundled open star-catalog data +
real altitude/azimuth math picks what's above the horizon tonight. A local
open-weight Gemma model (via Ollama) narrates the tour and answers
"how do I find it?" — location never leaves the machine.

Stdlib only. Env: PORT (8000), OLLAMA_HOST, OLLAMA_MODEL (gemma3:1b),
OLLAMA_NUM_GPU (0 = CPU, most portable).
"""
import json
import math
import os
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8002"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", "0"))

# Anchor stars from open catalogs (Yale Bright Star Catalogue / HYG database,
# both public-domain compilations). RA in hours, Dec in degrees. Approximate
# to ~0.1h/1deg — plenty for naked-eye "look that way" guidance.
CONSTELLATIONS = [
    {"name": "Pegasus", "star": "Markab", "ra": 23.08, "dec": 15.2,
     "hint": "Find the Great Square: four stars forming a big box high in the south."},
    {"name": "Andromeda", "star": "Alpheratz", "ra": 0.14, "dec": 29.1,
     "hint": "Start at the top-left star of Pegasus' Square and follow the chain going northeast."},
    {"name": "Cassiopeia", "star": "Schedar", "ra": 0.67, "dec": 56.5,
     "hint": "Look northeast for a bright 'W' (or 'M') of five stars, high up."},
    {"name": "Perseus", "star": "Mirfak", "ra": 3.40, "dec": 49.9,
     "hint": "Below Cassiopeia's W, a curved chain of stars rising in the northeast."},
    {"name": "Aries", "star": "Hamal", "ra": 2.12, "dec": 23.5,
     "hint": "A small bent line of three stars east of Pisces, mid-sky in the east."},
    {"name": "Pisces", "star": "Alrescha", "ra": 2.03, "dec": 2.75,
     "hint": "Faint V-shape below Pegasus; darkest corner of your terrace helps."},
    {"name": "Aquarius", "star": "Sadalsuud", "ra": 21.52, "dec": -5.6,
     "hint": "Southwest of Pegasus: look for a small triangle of faint stars."},
    {"name": "Capricornus", "star": "Deneb Algedi", "ra": 21.78, "dec": -16.1,
     "hint": "Low in the southwest, a thin smile-shaped arc. Sets early — catch it first."},
    {"name": "Cygnus", "star": "Deneb", "ra": 20.69, "dec": 45.3,
     "hint": "The Northern Cross: a huge cross diving toward the northwest horizon."},
    {"name": "Lyra", "star": "Vega", "ra": 18.62, "dec": 38.8,
     "hint": "Brilliant blue-white Vega low in the northwest — the brightest thing there."},
    {"name": "Aquila", "star": "Altair", "ra": 19.85, "dec": 8.9,
     "hint": "Bright Altair with two fainter guards each side, sinking in the west."},
    {"name": "Scorpius", "star": "Antares", "ra": 16.49, "dec": -26.4,
     "hint": "Red Antares very low in the southwest right after sunset — hurry."},
    {"name": "Sagittarius", "star": "Kaus Australis", "ra": 18.40, "dec": -34.4,
     "hint": "The Teapot pours low over the southern horizon after dark."},
    {"name": "Taurus", "star": "Aldebaran", "ra": 4.60, "dec": 16.5,
     "hint": "Orange Aldebaran climbing in the east with the Pleiades cluster above it."},
    {"name": "Orion", "star": "Betelgeuse", "ra": 5.92, "dec": 7.4,
     "hint": "Three belt stars in a row rising in the east — unmistakable by 10pm."},
    {"name": "Ursa Major", "star": "Dubhe", "ra": 11.06, "dec": 61.75,
     "hint": "The Big Dipper scrapes low across the northern sky from Bengaluru."},
]

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def sunapprox_alt_az(ra_h, dec_d, lat_d, lon_e_d, dt_utc):
    """Altitude/azimuth (deg) of an RA/Dec point for lat/lon at a UTC datetime."""
    jd = dt_utc.timestamp() / 86400.0 + 2440587.5
    d = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * d) % 360.0
    lst = (gmst + lon_e_d) % 360.0
    ha = math.radians((lst - ra_h * 15.0) % 360.0)
    if ha > math.pi:
        ha -= 2 * math.pi
    lat, dec = math.radians(lat_d), math.radians(dec_d)
    alt = math.asin(max(-1.0, min(1.0,
        math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(ha))))
    y = math.sin(ha)
    x = math.cos(ha) * math.sin(lat) - math.tan(dec) * math.cos(lat)
    az = (math.degrees(math.atan2(y, x)) + 180.0) % 360.0
    return math.degrees(alt), az


def tonights_sky(lat=12.97, lon=77.59, dt_utc=None, min_alt=15.0):
    dt_utc = dt_utc or datetime.now(timezone.utc)
    out = []
    for c in CONSTELLATIONS:
        alt, az = sunapprox_alt_az(c["ra"], c["dec"], lat, lon, dt_utc)
        if alt >= min_alt:
            out.append({**c, "alt": round(alt, 1),
                        "dir": COMPASS[int(((az + 22.5) % 360) // 45)]})
    out.sort(key=lambda c: -c["alt"])
    return out, dt_utc


TOUR_SYSTEM = (
    "You are a warm, brief stargazing guide for a beginner on a Bengaluru terrace. "
    "Write a naked-eye tour under 350 words: viewing order (setting constellations first), "
    "one star-hop instruction each, one myth or story bite for the top 3, and 3 practical tips "
    "(dark adaptation, phone brightness, best posture). Plain Markdown, no preamble."
)

DETAIL_SYSTEM = (
    "You are a concise stargazing guide. In under 120 words explain how to find the "
    "asked constellation tonight from Bengaluru: anchor stars, direction, height, and one "
    "memorable fact. Plain Markdown."
)


def ollama_chat(system, user, num_predict=500):
    payload = json.dumps({
        "model": OLLAMA_MODEL, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.7,
                    "num_gpu": OLLAMA_NUM_GPU},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return (d.get("message") or {}).get("content", "")


class Handler(BaseHTTPRequestHandler):
    server_version = "TerraceSkySpotter/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes())
        elif self.path == "/api/tonight":
            sky, dt = tonights_sky()
            self._send(200, json.dumps({"for": dt.strftime("%Y-%m-%d %H:%M UTC"),
                                        "visible": sky}).encode(), "application/json")
        elif self.path == "/api/health":
            try:
                with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
                    tags = json.loads(r.read().decode())
                models = [m.get("name", "") for m in tags.get("models", [])]
                ok = any(OLLAMA_MODEL in m for m in models)
                self._send(200, json.dumps({"ok": ok, "model": OLLAMA_MODEL}).encode(),
                           "application/json")
            except Exception as e:  # noqa: BLE001
                self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(),
                           "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:  # noqa: BLE001
            self._send(400, b"invalid JSON", "text/plain")
            return
        try:
            if self.path == "/api/tour":
                sky, dt = tonights_sky()
                lineup = "; ".join(
                    f"{c['name']} (anchor {c['star']}, {c['alt']}° {c['dir']})" for c in sky)
                text = ollama_chat(TOUR_SYSTEM,
                    f"Tonight {dt.strftime('%b %d')} over Bengaluru, these are up: {lineup}. "
                    f"Write the tour.")
            elif self.path == "/api/detail":
                name = str(body.get("name", ""))[:40]
                c = next((x for x in CONSTELLATIONS if x["name"].lower() == name.lower()), None)
                if not c:
                    self._send(404, b"unknown constellation", "text/plain")
                    return
                sky, _ = tonights_sky()
                live = next((x for x in sky if x["name"] == c["name"]), None)
                if live:
                    task = (f"{c['name']} (brightest star {c['star']}) is up right now: "
                            f"{live['alt']}° high in the {live['dir']}. "
                            f"Finder hint: {c['hint']}")
                else:
                    task = (f"{c['name']} (brightest star {c['star']}) is BELOW the horizon right now — "
                            f"do not describe it as visible. Say plainly it is not up, then say when to "
                            f"catch it (which season / evening hours) and give this finder hint for then: {c['hint']}")
                text = ollama_chat(DETAIL_SYSTEM, task)
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, text.encode(), "text/plain; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(502, f"ollama unreachable: {e}".encode(), "text/plain")


if __name__ == "__main__":
    print(f"Terrace Sky Spotter on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
