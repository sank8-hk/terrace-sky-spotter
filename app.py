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

# Approximate V magnitude of each anchor star (naked-eye brightness reference).
STAR_MAG = {"Markab": 2.5, "Alpheratz": 2.1, "Schedar": 2.2, "Mirfak": 1.8,
            "Hamal": 2.0, "Alrescha": 3.8, "Sadalsuud": 2.9, "Deneb Algedi": 2.9,
            "Deneb": 1.25, "Vega": 0.03, "Altair": 0.77, "Antares": 1.1,
            "Kaus Australis": 1.8, "Aldebaran": 0.85, "Betelgeuse": 0.5, "Dubhe": 1.8}


def _julian_day(dt_utc):
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def moon_phase(dt_utc):
    """Illumination 0..1 and phase name from a simple synodic model."""
    age = (_julian_day(dt_utc) - 2451550.1) % 29.530588853
    illum = (1 - math.cos(2 * math.pi * age / 29.530588853)) / 2
    names = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
             "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return round(illum, 2), names[int(((age / 29.530588853) * 8 + 0.5) % 8)]


def _kepler(m_deg, e):
    m = math.radians(m_deg % 360.0)
    e_star = e
    ecc = m
    for _ in range(8):
        ecc = ecc - (ecc - e_star * math.sin(ecc) - m) / (1 - e_star * math.cos(ecc))
    return math.degrees(ecc)


def _helio(n, i, w, a, e, m):
    """Heliocentric ecliptic rectangular coords (AU) from elements (deg, AU)."""
    ecc = _kepler(m, e)
    xv = a * (math.cos(math.radians(ecc)) - e)
    yv = a * (math.sqrt(1 - e * e) * math.sin(math.radians(ecc)))
    v = math.degrees(math.atan2(yv, xv))
    r = math.hypot(xv, yv)
    n, i, w, vw = map(math.radians, (n, i, w, v + w))
    xh = r * (math.cos(n) * math.cos(vw) - math.sin(n) * math.sin(vw) * math.cos(i))
    yh = r * (math.sin(n) * math.cos(vw) + math.cos(n) * math.sin(vw) * math.cos(i))
    zh = r * (math.sin(vw) * math.sin(i))
    return xh, yh, zh, v + math.degrees(w), r


def _ecl_to_ra_dec(x, y, z):
    obl = math.radians(23.4393)
    xe, ye, ze = x, y * math.cos(obl) - z * math.sin(obl), y * math.sin(obl) + z * math.cos(obl)
    ra = (math.degrees(math.atan2(ye, xe)) / 15.0) % 24.0
    return ra, math.degrees(math.atan2(ze, math.hypot(xe, ye)))


def planet_ra_dec(name, dt_utc):
    """Low-precision geocentric RA (h) / Dec (deg). Good to ~1-2 deg: naked-eye grade."""
    d = _julian_day(dt_utc) - 2451543.5
    el = {
        "Venus":   (76.6799 + 2.46590e-5 * d, 3.3946, 54.8910, 0.723330, 0.006773, 48.0052 + 1.60213 * d),
        "Mars":    (49.5574 + 2.11081e-5 * d, 1.8497, 286.5016, 1.523688, 0.093405, 18.6021 + 0.5240207766 * d),
        "Jupiter": (100.4542 + 2.76854e-5 * d, 1.3030, 273.8777, 5.20256, 0.048498, 19.8950 + 0.0830853001 * d),
        "Saturn":  (113.6634 + 2.38980e-5 * d, 2.4886, 339.3939, 9.55475, 0.055546, 316.9670 + 0.0334442282 * d),
    }[name]
    sun = (0.0, 0.0, 282.9404 + 4.70935e-5 * d, 1.0, 0.016709, 356.0470 + 0.9856002585 * d)
    px, py, pz, _, _ = _helio(*el)
    # NOTE: Schlyter's Sun row yields the Sun GEOCENTRIC directly, so Earth's
    # heliocentric vector is its negation: geo = helio_planet + sun_geo.
    ex, ey, ez, _, _ = _helio(*sun)
    return _ecl_to_ra_dec(px + ex, py + ey, pz + ez)


def moon_ra_dec(dt_utc):
    """Simplified lunar position, ~0.5-1 deg: fine for 'look that way'."""
    d = _julian_day(dt_utc) - 2451543.5
    ms = math.radians((356.0470 + 0.9856002585 * d) % 360.0)  # Sun mean anomaly
    mm = (115.3654 + 13.0649929509 * d) % 360.0  # Moon mean anomaly (deg)
    d_elong = (297.85036 + 12.19074912 * d) % 360.0  # mean elongation
    f = (93.2720950 + 13.22935021 * d) % 360.0
    lon = (218.3164477 + 13.17639648 * d
           + 6.288774 * math.sin(math.radians(mm))
           + 1.274027 * math.sin(math.radians(2 * d_elong - mm))
           + 0.658314 * math.sin(math.radians(2 * d_elong))
           + 0.213618 * math.sin(math.radians(2 * mm))
           - 0.185116 * math.sin(ms)
           - 0.114332 * math.sin(math.radians(2 * f)))
    lat = (5.128122 * math.sin(math.radians(f))
           + 0.280602 * math.sin(math.radians(mm + f))
           + 0.277693 * math.sin(math.radians(mm - f))
           + 0.173237 * math.sin(math.radians(2 * d_elong - f)))
    lon_r, lat_r = math.radians(lon % 360.0), math.radians(lat)
    x = math.cos(lon_r) * math.cos(lat_r)
    y = math.sin(lon_r) * math.cos(lat_r)
    z = math.sin(lat_r)
    return _ecl_to_ra_dec(x, y, z)


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


PLANETS = [
    {"name": "Venus", "mag": -4.4,
     "hint": "Blazing white — the brightest 'star' around, always near the Sun: low west after sunset or low east before dawn."},
    {"name": "Jupiter", "mag": -2.2,
     "hint": "Creamy-white and steady (planets don't twinkle). Binoculars reveal its four moons."},
    {"name": "Saturn", "mag": 0.6,
     "hint": "Pale gold point of light; its rings need a small telescope."},
    {"name": "Mars", "mag": 1.2,
     "hint": "Faint orange-red ember. Currently far and small — manage expectations."},
]


def tonights_sky(lat=12.97, lon=77.59, dt_utc=None, min_alt=15.0):
    dt_utc = dt_utc or datetime.now(timezone.utc)
    illum, phase = moon_phase(dt_utc)
    # Moonlight drowns faint targets: naked-eye limit slides from ~5.5 (new moon) to ~2.3 (full).
    mag_limit = 5.5 - illum * 3.2
    out = []
    for c in CONSTELLATIONS:
        alt, az = sunapprox_alt_az(c["ra"], c["dec"], lat, lon, dt_utc)
        if alt >= min_alt:
            mag = STAR_MAG.get(c["star"], 3.0)
            out.append({**c, "kind": "star", "mag": mag,
                        "washed": mag > mag_limit,
                        "alt": round(alt, 1),
                        "dir": COMPASS[int(((az + 22.5) % 360) // 45)]})
    for p in PLANETS:
        try:
            ra, dec = planet_ra_dec(p["name"], dt_utc)
        except KeyError:
            continue
        alt, az = sunapprox_alt_az(ra, dec, lat, lon, dt_utc)
        if alt >= min_alt:
            out.append({"name": p["name"], "star": p["name"], "kind": "planet",
                        "mag": p["mag"], "washed": False, "hint": p["hint"],
                        "alt": round(alt, 1),
                        "dir": COMPASS[int(((az + 22.5) % 360) // 45)]})
    try:
        mra, mdec = moon_ra_dec(dt_utc)
        malt, maz = sunapprox_alt_az(mra, mdec, lat, lon, dt_utc)
        moon = {"illum": illum, "phase": phase, "alt": round(malt, 1),
                "dir": COMPASS[int(((maz + 22.5) % 360) // 45)],
                "up": malt >= 0}
    except Exception:  # noqa: BLE001
        moon = {"illum": illum, "phase": phase, "alt": None, "dir": "?", "up": False}
    out.sort(key=lambda c: -c["alt"])
    return out, dt_utc, moon


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

    def _at_time(self):
        """Optional ?at=ISO or {'at': ISO} override so users can plan ahead."""
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        iso = qs.get("at", [None])[0]
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes())
        elif self.path.startswith("/api/tonight"):
            sky, dt, moon = tonights_sky(dt_utc=self._at_time())
            self._send(200, json.dumps({"for": dt.strftime("%Y-%m-%d %H:%M UTC"),
                                        "moon": moon, "visible": sky}).encode(),
                       "application/json")
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
            at = None
            if isinstance(body, dict) and body.get("at"):
                try:
                    at = datetime.fromisoformat(
                        str(body["at"]).replace("Z", "+00:00"))
                except ValueError:
                    at = None
            if self.path == "/api/tour":
                sky, dt, moon = tonights_sky(dt_utc=at)
                lineup = "; ".join(
                    f"{c['name']} ({c.get('kind', 'star')} {c['star']}, "
                    f"{c['alt']}° {c['dir']}"
                    f"{', washed out by moonlight' if c.get('washed') else ''})"
                    for c in sky)
                text = ollama_chat(TOUR_SYSTEM,
                    f"Night of {dt.strftime('%b %d')} over Bengaluru. Moon: {moon['phase']} "
                    f"({int(moon['illum'] * 100)}% lit). Up now: {lineup}. "
                    f"Write the tour; skip or de-prioritize washed-out targets.")
            elif self.path == "/api/detail":
                name = str(body.get("name", ""))[:40]
                c = next((x for x in CONSTELLATIONS if x["name"].lower() == name.lower()), None)
                kind = "constellation"
                if not c:
                    c = next((x for x in PLANETS if x["name"].lower() == name.lower()), None)
                    kind = "planet"
                if not c:
                    if name.lower() == "moon":
                        sky, dt, moon = tonights_sky(dt_utc=at)
                        text = (f"The Moon is {moon['phase']} ({int(moon['illum'] * 100)}% lit)"
                                + (f", {moon['alt']}° up in the {moon['dir']}." if moon["up"]
                                   else " and currently below the horizon."))
                        self._send(200, text.encode(), "text/plain; charset=utf-8")
                        return
                    self._send(404, b"unknown target", "text/plain")
                    return
                sky, _, _ = tonights_sky(dt_utc=at)
                live = next((x for x in sky if x["name"] == c["name"]), None)
                label = (f"planet {c['name']}" if kind == "planet"
                         else f"{c['name']} (brightest star {c['star']})")
                if live:
                    task = (f"{label} is up right now: "
                            f"{live['alt']}° high in the {live['dir']}. "
                            f"Finder hint: {c['hint']}")
                else:
                    task = (f"{label} is BELOW the horizon right now — "
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
