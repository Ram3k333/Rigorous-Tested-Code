#!/usr/bin/env python3
"""
visibility.py - Udaya / Asta (Darshan / Lop) engine on the Swiss Ephemeris
==========================================================================

For the Moon, Mercury, Venus, Mars, Jupiter and Saturn this program finds when each
body is lost in the Sun's rays (Lop / Asta) and when it is seen again (Darshan / Udaya),
in the Indian sense only: there are no acronychal or cosmical events.

Events
    Mars, Jupiter, Saturn   Lop  in the west (Pashchim asta, evening)  before conjunction
                            Darshan in the east (Purva udaya, morning) after conjunction
    Mercury, Venus          around inferior conjunction (retrograde):
                                Lop in the west (evening), Darshan in the east (morning)
                            around superior conjunction (direct):
                                Lop in the east (morning), Darshan in the west (evening)
    Moon                    Lop in the east (last crescent, morning) before amavasya
                            Darshan in the west (first crescent, Chandra darshan, evening)

Methods (--method)
    siddhanta            True kalamsha (degrees of time): the time between the planet's and the
                         Sun's rising (morning) or setting (evening), 4 minutes = 1 degree,
                         with Hindu rising (disc centre, no refraction). Visible when the
                         kalamsha reaches the Surya Siddhanta value. Default.
    siddhanta-longitude  The same values used as a distance in longitude from the Sun (the
                         common panchang practice). Independent of the place.
    av-fixed             Arcus visionis: the Sun's depression below the horizon when the planet
                         is at the critical altitude. Default values are PLSV's (Schoch 1928),
                         one per event; --av-preset ptolemy for Ptolemy's; --av to override.
    av-variable          Arcus visionis AV = c + k * magnitude.
    av-altaz             Altitude/azimuth model: the planet's magnitude against the twilight
                         sky's limiting magnitude at its altitude and azimuth (Schaefer model,
                         swe.vis_limit_mag), checked through the twilight every few minutes.
    For the Moon the AV methods use PLSV's crescent rule (lower limb altitude + 1/3 x the
    Sun-Moon distance >= 11.3 deg); the siddhanta methods use 12 kalamsha.

Lop / Darshan date-time
    The moment the body crosses the visibility limit, interpolated between the last day it
    is seen and the first day it is not (Lop), or the last day it is not seen and the first
    day it is (Darshan) - like the fractional-day result of the siddhantic rule. The detail
    table also gives the actual last / first sighting.

Requirements
    pip install pyswisseph matplotlib numpy      (pyswisseph 2.10 or newer; matplotlib and
                                                  numpy only for diagrams)
    Swiss Ephemeris files (sepl_18.se1, semo_18.se1, ...) in a folder passed with --ephe.

Examples
    python visibility.py                         (uses DEFAULT_RUN below)
    python visibility.py --name Neemuch --lat 24.4667 --lon 74.85 --tz 5.5 --year 2026 \
        --method siddhanta --ephe ./ephe --html neemuch_2026.html
"""
from __future__ import annotations

import argparse
import base64
import calendar
import csv
import html
import io
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import swisseph as swe
except ImportError:  # pragma: no cover
    sys.exit("pyswisseph is required:  pip install pyswisseph")

__version__ = "2.5"

# --------------------------------------------------------------------------------------
# DEFAULT RUN - used when the script is started with no command-line arguments
# (for example with the Run button of an editor). Command-line arguments, when given,
# replace all of these.
# --------------------------------------------------------------------------------------
DEFAULT_RUN = {
    "name": "Bhopal",
    "lat": 23.2599,          # north positive
    "lon": 77.4126,          # east positive
    "elev": 527,             # metres
    "tz": 5.5,               # hours east of UTC (IST)
    "year": None,            # None = the current year
    "method": "av-fixed",   # siddhanta | siddhanta-longitude | av-fixed | av-variable | av-altaz
    "bodies": "moon,mercury,venus,mars,jupiter,saturn",
    "ephe": "ephe",          # folder with the .se1 files, relative to this script
    "html": "udaya_asta_report.html",   # written next to this script
    "csv": None,             # e.g. "udaya_asta.csv"
    "txt": None,             # e.g. "udaya_asta.txt" - the console report saved to a file
    "diagrams": True,        # needs matplotlib and numpy
    "kalamsha": {},          # overrides, e.g. {"mercury": 13} or {"venus.EL": 9}
    "av_preset": "plsv",     # av-fixed values: "plsv" (Schoch 1928, as in Alcyone PLSV) or "ptolemy"
    "av": {},                # av-fixed overrides on top of the preset, e.g. {"venus.MF": 6}
                             # or {"jupiter": 10} for all events of a planet
    "crit_alt": 0.0,         # critical altitude for the AV methods (deg)
    "sunrise": None,         # None = method default; "hindu" (disc centre, no refraction)
                             # or "standard" (upper limb, refraction)
    "ayanamsa": None,        # e.g. "lahiri" for sidereal longitudes in the detail table
    "time": None,            # None = method default; "crossing" (siddhantic moment the limit
                             # is crossed) or "sighting" (actual last / first sighting)
}

# --------------------------------------------------------------------------------------
# Bodies, events, constants
# --------------------------------------------------------------------------------------
SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN = (
    swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN)
BODY_NAMES = {SUN: "Sun", MOON: "Moon", MERCURY: "Mercury", VENUS: "Venus",
              MARS: "Mars", JUPITER: "Jupiter", SATURN: "Saturn"}
BODY_BY_NAME = {v.lower(): k for k, v in BODY_NAMES.items()}
PLANETS = (MERCURY, VENUS, MARS, JUPITER, SATURN)
INFERIOR = (MERCURY, VENUS)
SUPERIOR = (MARS, JUPITER, SATURN)
ALL_BODIES = (MOON,) + PLANETS

METHODS = ("siddhanta", "siddhanta-longitude", "av-fixed", "av-variable", "av-altaz")
METHOD_ALIASES = {"kalamsha": "siddhanta", "longitude": "siddhanta-longitude",
                  "fixed": "av-fixed", "ptolemy": "av-fixed", "variable": "av-variable",
                  "altaz": "av-altaz"}
METHOD_TEXT = {
    "siddhanta": "Siddhanta - true kalamsha (time between rising/setting of planet and Sun)",
    "siddhanta-longitude": "Siddhanta - kalamsha used as distance in longitude from the Sun",
    "av-fixed": "Arcus visionis - fixed (Sun's depression when the planet is at the critical altitude)",
    "av-variable": "Arcus visionis - variable, AV = c + k x magnitude",
    "av-altaz": "Arcus visionis - altitude/azimuth model (limiting magnitude, Schaefer)",
}

# Event codes: the half of the day and what the body does at the horizon then.
#   L / EL  evening last  -> Lop in the west      ML  morning last  -> Lop in the east
#   F / MF  morning first -> Darshan in the east  EF  evening first -> Darshan in the west
HALF = {"L": "evening", "EL": "evening", "EF": "evening", "F": "morning", "MF": "morning", "ML": "morning"}
LOP_CODES = ("L", "EL", "ML")
DARSHAN_CODES = ("F", "MF", "EF")
DIRECTION = {"morning": "Purva", "evening": "Pashchim"}          # east / west
ANCHOR_EVENTS = {"conjunction": ("L", "F"), "inferior conjunction": ("EL", "MF"),
                 "superior conjunction": ("ML", "EF"), "new moon": ("ML", "EF")}
VALID_CODES = {MOON: ("ML", "EF"), MERCURY: ("ML", "EF", "EL", "MF"), VENUS: ("ML", "EF", "EL", "MF"),
               MARS: ("L", "F"), JUPITER: ("L", "F"), SATURN: ("L", "F")}

# Surya Siddhanta kalamsha (degrees of time). Mercury and Venus have the smaller value
# when retrograde, i.e. around inferior conjunction (EL / MF).
KALAMSHA_SS = {
    MOON: {"ML": 12.0, "EF": 12.0},
    MERCURY: {"ML": 14.0, "EF": 14.0, "EL": 12.0, "MF": 12.0},
    VENUS: {"ML": 10.0, "EF": 10.0, "EL": 8.0, "MF": 8.0},
    MARS: {"L": 17.0, "F": 17.0},
    JUPITER: {"L": 11.0, "F": 11.0},
    SATURN: {"L": 15.0, "F": 15.0},
}
# Arcus visionis presets for av-fixed: the Sun's depression below the horizon when the planet
# is on the horizon (deg).
# "plsv": Karl Schoch (1928), the default fixed values of Alcyone PLSV. For Mars, Jupiter and
#         Saturn, F = heliacal rising (morning first), L = heliacal setting (evening last).
# "ptolemy": Almagest XIII.7, one value per planet.
AV_PRESETS = {
    "plsv": {
        MERCURY: {"MF": 13.0, "ML": 9.5, "EF": 10.5, "EL": 11.0},
        VENUS: {"MF": 5.7, "ML": 6.0, "EF": 6.0, "EL": 5.2},
        MARS: {"F": 14.5, "L": 13.2},
        JUPITER: {"F": 9.3, "L": 7.4},
        SATURN: {"F": 13.0, "L": 10.0},
    },
    "ptolemy": {
        MERCURY: {c: 10.0 for c in ("MF", "ML", "EF", "EL")},
        VENUS: {c: 5.0 for c in ("MF", "ML", "EF", "EL")},
        MARS: {"F": 11.5, "L": 11.5},
        JUPITER: {"F": 10.0, "L": 10.0},
        SATURN: {"F": 11.0, "L": 11.0},
    },
}
AV_PRESET_TEXT = {"plsv": "PLSV / Schoch 1928", "ptolemy": "Ptolemy, Almagest XIII.7"}

# Days to walk away from a conjunction / new moon when looking for an event
SCAN_DAYS = {MERCURY: 50, VENUS: 80, MARS: 180, JUPITER: 75, SATURN: 75, MOON: 5}
SIDEREAL_DEG_PER_DAY = 360.98564736629   # 1 solar day of time = this many degrees of time

SWE_EVENT = {"F": 1, "MF": 1, "L": 2, "EL": 2, "EF": 3, "ML": 4}

AYANAMSAS = {
    "lahiri": swe.SIDM_LAHIRI, "raman": swe.SIDM_RAMAN,
    "krishnamurti": swe.SIDM_KRISHNAMURTI, "fagan_bradley": swe.SIDM_FAGAN_BRADLEY,
    "yukteshwar": swe.SIDM_YUKTESHWAR, "true_citra": swe.SIDM_TRUE_CITRA,
}


def swe_name(body: int) -> str:
    """Object name for the heliacal functions. Swiss Ephemeris lower-cases the name it is
    given in place, so always hand it a fresh copy, never a shared constant."""
    return "".join(BODY_NAMES[body].lower())


def norm180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def ang_sep(az1: float, alt1: float, az2: float, alt2: float) -> float:
    r = math.radians
    c = (math.sin(r(alt1)) * math.sin(r(alt2))
         + math.cos(r(alt1)) * math.cos(r(alt2)) * math.cos(r(az1 - az2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pressure_from_elevation(elev_m: float) -> float:
    return 1013.25 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclass
class Location:
    name: str
    lat: float
    lon: float
    elev: float = 0.0
    tz: float = 0.0

    @property
    def geopos(self) -> Tuple[float, float, float]:
        return (self.lon, self.lat, self.elev)


@dataclass
class Atmosphere:
    pressure: Optional[float] = None   # hPa; None = from elevation
    temperature: float = 15.0
    humidity: float = 40.0
    extinction: float = 0.0            # 0 auto; 0<k<1 coefficient; >=1 meteorological range km


@dataclass
class Observer:
    age: float = 36.0
    snellen: float = 1.0

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.age, self.snellen, 0.0, 0.0, 0.0, 0.0)


@dataclass
class PlanetParams:
    av_events: Dict[str, float]        # av-fixed: AV per event (ML EF EL MF or L F)
    critical_alt: float = 0.0          # AV methods: least altitude at which it can be seen
    c: float = 10.5                    # av-variable: AV = c + k * mag
    k: float = 1.4


def default_planet_params(preset: str = "plsv") -> Dict[int, PlanetParams]:
    return {p: PlanetParams(av_events=dict(AV_PRESETS[preset][p])) for p in PLANETS}


def default_kalamsha() -> Dict[int, Dict[str, float]]:
    return {b: dict(v) for b, v in KALAMSHA_SS.items()}


@dataclass
class MoonParams:
    vis_arc: float = 11.3        # PLSV lunar visibility arc
    coef: float = 1.0 / 3.0      # PLSV coefficient of the Sun-Moon centre distance (prints 0.333)


@dataclass
class Settings:
    method: str = "siddhanta"
    planets: Dict[int, PlanetParams] = field(default_factory=default_planet_params)
    kalamsha: Dict[int, Dict[str, float]] = field(default_factory=default_kalamsha)
    moon: MoonParams = field(default_factory=MoonParams)
    sunrise: Optional[str] = None       # None = method default; 'hindu' or 'standard'
    refraction: bool = True             # AV methods
    step_minutes: float = 2.0           # av-altaz twilight step
    moonlight: bool = False             # av-altaz
    ayanamsa: Optional[str] = None
    gregorian: bool = False             # True = proleptic Gregorian before 1582
    av_preset: str = "plsv"             # name of the av-fixed preset, for the report
    round_minutes: bool = False         # False = drop the seconds (as PLSV); True = round
    magnitudes: Optional[str] = None    # None = method default; 'plsv' (Mueller) or 'swe'

    def __post_init__(self):
        if self.magnitudes is None:
            # av-altaz uses Swiss Ephemeris' own visibility model, so its magnitudes too
            self.magnitudes = "swe" if self.method in ("av-altaz", "altaz") else "plsv"
    time_mode: Optional[str] = None     # None = method default; 'crossing' or 'sighting'

    @property
    def is_siddhanta(self) -> bool:
        return self.method.startswith("siddhanta")

    @property
    def hindu_sunrise(self) -> bool:
        if self.sunrise:
            return self.sunrise == "hindu"
        return self.is_siddhanta

    @property
    def report_crossing(self) -> bool:
        """Report the moment the limit is crossed (siddhantic, fractional day) rather than
        the actual last / first sighting. The longitude method has no sighting."""
        if self.method == "siddhanta-longitude":
            return True
        if self.time_mode:
            return self.time_mode == "crossing"
        return self.is_siddhanta


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class Event:
    body: int
    code: str                    # L EL ML (lop) / F MF EF (darshan)
    anchor_kind: str
    anchor_jd: float
    method: str
    jd: float                    # reported Lop / Darshan moment (crossing or sighting)
    seen_jd: Optional[float]     # last sighting (lop) / first sighting (darshan)
    cross_jd: Optional[float] = None   # moment the limit is crossed (interpolated)
    value: Optional[float] = None      # measure on the sighting day (kalamsha, Sun depression, ...)
    limit: Optional[float] = None      # the limit it is compared with
    geo: dict = field(default_factory=dict)   # geometry at the sighting moment
    swe_jd: Optional[float] = None

    @property
    def name(self) -> str:
        return BODY_NAMES[self.body]

    @property
    def is_lop(self) -> bool:
        return self.code in LOP_CODES

    @property
    def half(self) -> str:
        return HALF[self.code]

    @property
    def direction(self) -> str:
        return DIRECTION[self.half]


@dataclass
class Period:
    body: int
    anchor_kind: str
    anchor_jd: float
    lop: Optional[Event]
    darshan: Optional[Event]

    @property
    def name(self) -> str:
        return BODY_NAMES[self.body]

    @property
    def days(self) -> Optional[float]:
        if self.lop and self.darshan:
            return self.darshan.jd - self.lop.jd
        return None


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
class UdayaAstaEngine:
    def __init__(self, location: Location, settings: Optional[Settings] = None,
                 atmosphere: Optional[Atmosphere] = None, observer: Optional[Observer] = None,
                 ephe_path: Optional[str] = None):
        self.loc = location
        self.st = settings or Settings()
        self.st.method = METHOD_ALIASES.get(self.st.method, self.st.method)
        if self.st.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.atm = atmosphere or Atmosphere()
        self.obs = observer or Observer()
        swe.set_ephe_path(ephe_path)
        swe.set_topo(location.lon, location.lat, location.elev)
        self.press = (self.atm.pressure if self.atm.pressure is not None
                      else pressure_from_elevation(location.elev))
        self.temp = self.atm.temperature
        self.datm = (self.press, self.atm.temperature, self.atm.humidity, self.atm.extinction)
        self.dobs = self.obs.as_tuple()
        self.eflag = swe.FLG_SWIEPH
        if self.st.ayanamsa:
            key = self.st.ayanamsa.lower()
            if key not in AYANAMSAS:
                raise ValueError(f"unknown ayanamsa {self.st.ayanamsa!r}; use one of {list(AYANAMSAS)}")
            swe.set_sid_mode(AYANAMSAS[key])

    # ------------------------------------------------------------------ time
    def _cal_for_date(self, y: int, m: int, d: int) -> int:
        if self.st.gregorian or (y, m, d) >= (1582, 10, 15):
            return swe.GREG_CAL
        return swe.JUL_CAL

    def jd_local(self, y: int, m: int, d: int, hour: float = 0.0) -> float:
        return swe.julday(y, m, d, hour, self._cal_for_date(y, m, d)) - self.loc.tz / 24.0

    def local(self, jd: float) -> Tuple[int, int, int, float]:
        jl = jd + self.loc.tz / 24.0
        cal = swe.GREG_CAL if (self.st.gregorian or jl >= 2299160.5) else swe.JUL_CAL
        return swe.revjul(jl, cal)

    def local_midnight(self, jd: float) -> float:
        y, m, d, _ = self.local(jd)
        return self.jd_local(y, m, d, 0.0)

    def _shift(self, jd: float) -> float:
        """Minutes are truncated (seconds dropped), as PLSV prints them, or rounded."""
        return jd + (0.5 / 1440.0 if self.st.round_minutes else 1e-7)

    def fmt_date(self, jd: Optional[float]) -> str:
        if jd is None:
            return "-"
        y, m, d, _ = self.local(jd)
        return f"{y:04d}-{m:02d}-{d:02d}" if y > 0 else f"{y:05d}-{m:02d}-{d:02d}"

    def _hhmm(self, t: float) -> str:
        _, _, _, h = self.local(t)
        hh = int(h)
        mm = int((h - hh) * 60.0)
        return f"{hh:02d}:{mm:02d}"

    def fmt_time(self, jd: Optional[float]) -> str:
        return "-" if jd is None else self._hhmm(self._shift(jd))

    def fmt_dt(self, jd: Optional[float]) -> str:
        if jd is None:
            return "-"
        t = self._shift(jd)          # date and time from the same shifted instant
        return f"{self.fmt_date(t)} {self._hhmm(t)}"

    # ------------------------------------------------------------------ positions
    def ecl(self, jd: float, body: int) -> Tuple[float, float, float]:
        x, _ = swe.calc_ut(jd, body, self.eflag)
        return x[0], x[1], x[2]

    def out_lon(self, jd: float, lon: float) -> float:
        if self.st.ayanamsa:
            return (lon - swe.get_ayanamsa_ut(jd)) % 360.0
        return lon

    def horizon(self, jd: float, body: int) -> Tuple[float, float, float]:
        """Topocentric azimuth (north = 0, east = 90), true altitude, apparent altitude."""
        x, _ = swe.calc_ut(jd, body, self.eflag | swe.FLG_EQUATORIAL | swe.FLG_TOPOCTR)
        az, alt_true, alt_app = swe.azalt(jd, swe.EQU2HOR, self.loc.geopos,
                                          self.press, self.temp, (x[0], x[1], x[2]))
        return (az + 180.0) % 360.0, alt_true, self.apparent_alt(alt_true, alt_app)

    def apparent_alt(self, true_alt: float, swe_app: float) -> float:
        """Continuous apparent altitude near the horizon (Swiss Ephemeris returns the true
        altitude once a body is judged below the horizon, which shows as about -0.56)."""
        if true_alt > 2.0 or true_alt < -2.0:
            return swe_app
        h = true_alt
        r = 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0
        return h + r * (self.press / 1010.0) * (283.0 / (273.0 + self.temp))

    def pheno(self, jd: float, body: int) -> tuple:
        return swe.pheno_ut(jd, body, self.eflag)

    def magnitude(self, jd: float, body: int) -> float:
        """Apparent magnitude. 'plsv': G. Mueller's formulas (as in Meeus, Astronomical
        Algorithms ch. 41), which Alcyone PLSV uses; 'swe': Swiss Ephemeris (newer formulas,
        Mercury near inferior conjunction up to ~0.7 mag fainter)."""
        if self.st.magnitudes == "swe" or body not in PLANETS:
            return self.pheno(jd, body)[4]
        return self.mueller_magnitude(jd, body)

    def mueller_magnitude(self, jd: float, body: int) -> float:
        xh, _ = swe.calc_ut(jd, body, self.eflag | swe.FLG_HELCTR)
        xg, _ = swe.calc_ut(jd, body, self.eflag)
        r, delta = xh[2], xg[2]
        i = self.pheno(jd, body)[0]                      # phase angle Earth-planet-Sun
        m5 = 5.0 * math.log10(r * delta)
        if body == MERCURY:
            return 1.16 + m5 + 0.02838 * (i - 50.0) + 0.0001023 * (i - 50.0) ** 2
        if body == VENUS:
            return -4.00 + m5 + 0.01322 * i + 0.0000004247 * i ** 3
        if body == MARS:
            return -1.30 + m5 + 0.01486 * i
        if body == JUPITER:
            return -8.93 + m5
        # Saturn with its rings (Meeus ch. 45)
        t = (jd + swe.deltat(jd) - 2451545.0) / 36525.0
        inc = math.radians(28.075216 - 0.012998 * t + 0.000004 * t * t)
        node = math.radians(169.508470 + 1.394681 * t + 0.000412 * t * t)
        lam, bet = math.radians(xg[0]), math.radians(xg[1])
        l, b = math.radians(xh[0]), math.radians(xh[1])
        n = math.radians(113.6655 + 0.8771 * t)
        l1 = l - math.radians(0.01759) / r
        b1 = b - math.radians(0.000764) * math.cos(l - n) / r
        big_b = math.asin(math.sin(inc) * math.cos(bet) * math.sin(lam - node)
                          - math.cos(inc) * math.sin(bet))
        u1 = math.atan2(math.sin(inc) * math.sin(b1) + math.cos(inc) * math.cos(b1) * math.sin(l1 - node),
                        math.cos(b1) * math.cos(l1 - node))
        u2 = math.atan2(math.sin(inc) * math.sin(bet) + math.cos(inc) * math.cos(bet) * math.sin(lam - node),
                        math.cos(bet) * math.cos(lam - node))
        du = abs(norm180(math.degrees(u1 - u2)))      # degrees
        sb = math.sin(abs(big_b))
        return -8.68 + m5 + 0.044 * du - 2.60 * sb + 1.25 * sb * sb

    def rise_set(self, start: float, body: int, rising: bool, hindu: bool,
                 alt: float = 0.0) -> Optional[float]:
        """Next rising / setting after `start`.
        hindu=True: centre of the disc, no refraction (Hindu rising).
        hindu=False: the Sun by the almanac rule (centre 50' below the horizon: 16' semi-
        diameter + 34' standard refraction), as PLSV; the Moon by its upper limb with
        refraction; planets by the centre, at apparent altitude `alt` (critical altitude)."""
        rsmi = swe.CALC_RISE if rising else swe.CALC_SET
        if body == SUN and not hindu and self.st.refraction:
            rsmi |= swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION
            res, tret = swe.rise_trans_true_hor(start, body, rsmi, self.loc.geopos, self.press,
                                                self.temp, -50.0 / 60.0, self.eflag)
            return tret[0] if res == 0 else None
        if hindu:
            rsmi |= swe.BIT_HINDU_RISING if body == SUN else (swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION)
        else:
            if body not in (SUN, MOON):
                rsmi |= swe.BIT_DISC_CENTER
            if not self.st.refraction:
                rsmi |= swe.BIT_NO_REFRACTION
        if alt == 0.0:
            res, tret = swe.rise_trans(start, body, rsmi, self.loc.geopos, self.press,
                                       self.temp, self.eflag)
        else:
            res, tret = swe.rise_trans_true_hor(start, body, rsmi, self.loc.geopos,
                                                self.press, self.temp, alt, self.eflag)
        return tret[0] if res == 0 else None

    def sun_times(self, jd: float) -> Tuple[Optional[float], Optional[float]]:
        """Sun udaya and Sun asta of the local date containing jd."""
        d0 = self.local_midnight(jd)
        hindu = self.st.hindu_sunrise
        return self.rise_set(d0, SUN, True, hindu), self.rise_set(d0, SUN, False, hindu)

    # ------------------------------------------------------------------ anchors
    def find_anchors(self, body: int, jd0: float, jd1: float) -> List[Tuple[str, float]]:
        """Conjunctions with the Sun (inferior / superior for Mercury and Venus) or new moons."""
        step = 0.5 if body == MOON else 1.0

        def f(t):
            return norm180(self.ecl(t, body)[0] - self.ecl(t, SUN)[0])

        out, t, fa = [], jd0, f(jd0)
        while t < jd1:
            tb = t + step
            fb = f(tb)
            if fa == 0.0 or (fa * fb < 0 and abs(fa - fb) < 90.0):
                a, b, fa_ = t, tb, fa
                for _ in range(45):
                    m = 0.5 * (a + b)
                    fm = f(m)
                    if fa_ * fm <= 0:
                        b = m
                    else:
                        a, fa_ = m, fm
                root = 0.5 * (a + b)
                if body == MOON:
                    kind = "new moon"
                elif body in INFERIOR:
                    kind = ("inferior conjunction" if self.ecl(root, body)[2] < self.ecl(root, SUN)[2]
                            else "superior conjunction")
                else:
                    kind = "conjunction"
                out.append((kind, root))
            t, fa = tb, fb
        return out

    # ------------------------------------------------------------------ limits
    def kalamsha(self, body: int, code: str) -> float:
        return self.st.kalamsha[body][code]

    def av_value(self, body: int, code: str, mag: float) -> float:
        p = self.st.planets[body]
        if self.st.method == "av-variable":
            return p.c + p.k * mag
        return p.av_events[code]

    # ------------------------------------------------------------------ daily measures
    # Each measure returns a dict with 'value' (>= 0 means seen that morning/evening),
    # 't_ref' (the moment used for interpolation and for the sighting) and details,
    # or None when the body does not take part in that half day.
    def measure(self, day0: float, body: int, code: str) -> Optional[dict]:
        m = self.st.method
        if m == "siddhanta":
            return self._m_kalamsha(day0, body, code)
        if m == "siddhanta-longitude":
            return self._m_longitude(day0, body, code)
        if body == MOON:
            return self._m_moon_plsv(day0, code)
        if m == "av-altaz":
            return self._m_altaz(day0, body, code)
        return self._m_av(day0, body, code)

    def _m_kalamsha(self, day0: float, body: int, code: str) -> Optional[dict]:
        """True kalamsha: time between the body's and the Sun's rising (morning) or setting
        (evening) in degrees of time, Hindu rising for both."""
        morning = HALF[code] == "morning"
        start = day0 if morning else day0 + 0.5
        t_sun = self.rise_set(start, SUN, morning, hindu=True)
        if t_sun is None:
            return None
        # the body's rising (setting) nearest the sunrise (sunset), within 12 hours before it
        t_obj = self.rise_set(t_sun - 0.5, body, morning, hindu=True)
        if t_obj is None:
            return None
        k = (t_sun - t_obj) if morning else (t_obj - t_sun)
        kal = k * SIDEREAL_DEG_PER_DAY
        lim = self.kalamsha(body, code)
        return dict(value=kal - lim, t_ref=t_obj, t_obj=t_obj, t_sun=t_sun, measure=kal, limit=lim)

    def _m_longitude(self, day0: float, body: int, code: str) -> Optional[dict]:
        """Kalamsha as a distance in longitude, at the Sun's rising (morning) / setting."""
        morning = HALF[code] == "morning"
        start = day0 if morning else day0 + 0.5
        t_sun = self.rise_set(start, SUN, morning, hindu=self.st.hindu_sunrise)
        if t_sun is None:
            return None
        dl = norm180(self.ecl(t_sun, body)[0] - self.ecl(t_sun, SUN)[0])
        side_ok = (dl < 0) if morning else (dl > 0)     # morning: west of the Sun
        lim = self.kalamsha(body, code)
        val = (abs(dl) - lim) if side_ok else -abs(dl) - lim
        return dict(value=val, t_ref=t_sun, t_obj=None, t_sun=t_sun, measure=abs(dl), limit=lim)

    def _m_av(self, day0: float, body: int, code: str) -> Optional[dict]:
        """Fixed / variable arcus visionis at the moment the planet is at the critical altitude."""
        morning = HALF[code] == "morning"
        start = day0 if morning else day0 + 0.5
        p = self.st.planets[body]
        t_obj = self.rise_set(start, body, morning, hindu=False, alt=p.critical_alt)
        if t_obj is None or t_obj >= start + 0.5:
            return None
        _, sun_alt, _ = self.horizon(t_obj, SUN)
        mag = self.magnitude(t_obj, body)
        av = self.av_value(body, code, mag)
        return dict(value=-sun_alt - av, t_ref=t_obj, t_obj=t_obj,
                    t_sun=self.rise_set(start, SUN, morning, hindu=False),
                    measure=-sun_alt, limit=av)

    def _vislim(self, t: float, body: int) -> Optional[tuple]:
        flags = self.eflag | (0 if self.st.moonlight else swe.HELFLAG_VISLIM_NOMOON)
        try:
            res, dret = swe.vis_limit_mag(t, self.loc.geopos, self.datm, self.dobs,
                                          swe_name(body), flags)
        except swe.Error:
            return None
        return None if res < 0 else dret   # lim, altO, aziO, altS, aziS, altM, aziM, magO

    def _m_altaz(self, day0: float, body: int, code: str) -> Optional[dict]:
        """Best limiting-magnitude margin in the twilight window of that morning/evening."""
        morning = HALF[code] == "morning"
        start = day0 if morning else day0 + 0.5
        p = self.st.planets[body]
        t_obj = self.rise_set(start, body, morning, hindu=False, alt=p.critical_alt)
        if t_obj is None or t_obj >= start + 0.5:
            return None
        t_sun = self.rise_set(start, SUN, morning, hindu=False)
        if t_sun is None:
            return None
        a, b = (t_obj, t_sun) if morning else (t_sun, t_obj)
        best = None
        if b > a:
            step = self.st.step_minutes / 1440.0
            n = int((b - a) / step) + 1
            for t in [a + i * step for i in range(n)] + [b]:
                d = self._vislim(t, body)
                if d is None or d[3] >= 0.0:
                    continue
                margin = d[0] - d[7]
                if best is None or margin > best[0]:
                    best = (margin, t, d)
        if best is None:
            # the planet is never in a dark enough sky that morning/evening: not seen
            return dict(value=None, t_ref=t_obj, t_obj=t_obj, t_sun=t_sun, measure=None, limit=0.0)
        margin, t, d = best
        return dict(value=margin, t_ref=t_obj, t_obj=t_obj, t_sun=t_sun, t_best=t,
                    measure=margin, limit=0.0, lim_mag=d[0], arcv=d[1] - d[3], best_alt=d[1])

    def _m_moon_plsv(self, day0: float, code: str) -> Optional[dict]:
        """PLSV crescent rule at sunset (darshan) / sunrise (lop)."""
        evening = HALF[code] == "evening"
        start = day0 + (0.5 if evening else 0.0)
        ts = self.rise_set(start, SUN, not evening, hindu=False)
        if ts is None:
            return None
        az_m, alt_m_true, alt_m_app = self.horizon(ts, MOON)
        az_s, alt_s_true, _ = self.horizon(ts, SUN)
        sd = self._moon_sd_topo(ts)
        crit = (alt_m_app - sd) + self.st.moon.coef * ang_sep(az_m, alt_m_true, az_s, alt_s_true)
        # the sighting: moonrise before sunrise (lop) / moonset after sunset (darshan)
        t_moon = self.rise_set(ts if evening else ts - 0.5, MOON, not evening, hindu=False)
        if t_moon is not None and ((evening and t_moon <= ts) or (not evening and t_moon >= ts)):
            t_moon = None
        return dict(value=crit - self.st.moon.vis_arc, t_ref=ts, t_obj=t_moon, t_sun=ts,
                    measure=crit, limit=self.st.moon.vis_arc)

    def _moon_sd_topo(self, jd: float) -> float:
        x, _ = swe.calc_ut(jd, MOON, self.eflag | swe.FLG_TOPOCTR)
        return math.degrees(math.asin(1737.4 / (x[2] * 149597870.7)))

    # ------------------------------------------------------------------ events
    @staticmethod
    def _seen(r: Optional[dict]) -> bool:
        return r is not None and r.get("value") is not None and r["value"] >= 0.0

    def _crossing(self, vis: Tuple[float, dict], inv: Tuple[float, Optional[dict]]) -> float:
        """Interpolate the moment the measure crosses its limit between the day the body is
        seen and the neighbouring day it is not."""
        d_v, r_v = vis
        d_i, r_i = inv
        tv = r_v["t_ref"]
        ti = r_i["t_ref"] if (r_i and r_i.get("t_ref") is not None) else tv + (d_i - d_v)
        vi = r_i.get("value") if r_i else None
        if vi is None or r_v["value"] - vi <= 0:
            f = 0.5
        else:
            f = r_v["value"] / (r_v["value"] - vi)
        return tv + max(0.0, min(1.0, f)) * (ti - tv)

    def _geometry(self, t: float, body: int) -> dict:
        az_o, _, alt_o = self.horizon(t, body)
        az_s, alt_s, _ = self.horizon(t, SUN)
        lo, la, _ = self.ecl(t, body)
        ls, _, _ = self.ecl(t, SUN)
        ph = self.pheno(t, body)
        return dict(az_obj=az_o, alt_obj=alt_o, az_sun=az_s, sun_alt=alt_s, obj_lon=lo,
                    obj_lat=la, sun_lon=ls, mag=self.magnitude(t, body), elong=ph[2], phase=ph[1])

    def find_event(self, body: int, anchor_kind: str, anchor_jd: float, code: str) -> Optional[Event]:
        if self.st.method == "siddhanta-longitude":
            return self._longitude_event(body, anchor_kind, anchor_jd, code)
        direction = -1 if code in LOP_CODES else 1
        day0 = self.local_midnight(anchor_jd)
        n = SCAN_DAYS[body]
        r0 = self.measure(day0, body, code)
        vis = inv = None
        if self._seen(r0):
            # seen on the anchor day itself: the run crosses the anchor, follow it to its end
            prev = (day0, r0)
            for k in range(1, n):
                d = day0 - direction * k
                r = self.measure(d, body, code)
                if not self._seen(r):
                    vis, inv = prev, (d, r)
                    break
                prev = (d, r)
        else:
            prev = (day0, r0)
            for k in range(1, n):
                d = day0 + direction * k
                r = self.measure(d, body, code)
                if self._seen(r):
                    vis, inv = (d, r), prev
                    break
                prev = (d, r)
        if vis is None:
            return None
        r = vis[1]
        extra = {k: v for k, v in r.items()
                 if k in ("t_best", "lim_mag", "arcv", "best_alt", "t_sun", "t_obj")}
        cross = self._crossing(vis, inv)
        # the sighting: av-altaz best moment of the twilight; otherwise the body's rise/set
        seen = r.get("t_best") or r.get("t_obj") or r["t_ref"]
        return Event(body=body, code=code, anchor_kind=anchor_kind, anchor_jd=anchor_jd,
                     method=self.st.method, jd=cross if self.st.report_crossing else seen,
                     seen_jd=seen, cross_jd=cross, value=r.get("measure"), limit=r.get("limit"),
                     geo={**self._geometry(r["t_ref"], body), **extra})

    def _longitude_event(self, body: int, anchor_kind: str, anchor_jd: float, code: str) -> Optional[Event]:
        """Moment the longitude distance from the Sun equals the kalamsha value."""
        lim = self.kalamsha(body, code)
        direction = -1 if code in LOP_CODES else 1

        def f(t):
            return abs(norm180(self.ecl(t, body)[0] - self.ecl(t, SUN)[0])) - lim

        step = 0.1 if body == MOON else 0.5
        t, f0 = anchor_jd, f(anchor_jd)
        if f0 >= 0:
            return None
        found = False
        for _ in range(int(SCAN_DAYS[body] / step) + 1):
            t1 = t + direction * step
            f1 = f(t1)
            if f1 >= 0:
                found = True
                break
            t, f0 = t1, f1
        if not found:
            return None
        a, b = t, t1           # f(a) < 0 <= f(b)
        for _ in range(40):
            m = 0.5 * (a + b)
            if f(m) >= 0:
                b = m
            else:
                a = m
        jd = 0.5 * (a + b)
        return Event(body=body, code=code, anchor_kind=anchor_kind, anchor_jd=anchor_jd,
                     method=self.st.method, jd=jd, seen_jd=None, cross_jd=jd, value=lim, limit=lim,
                     geo=self._geometry(jd, body))

    def periods_for_body(self, body: int, year: int) -> List[Period]:
        y0, y1 = self.jd_local(year, 1, 1), self.jd_local(year + 1, 1, 1)
        pad = SCAN_DAYS[body] + 5
        out = []
        for kind, jd in self.find_anchors(body, y0 - pad, y1 + pad):
            lop_code, dar_code = ANCHOR_EVENTS[kind]
            lop = self.find_event(body, kind, jd, lop_code)
            dar = self.find_event(body, kind, jd, dar_code)
            if any(y0 <= e.jd < y1 for e in (lop, dar) if e):
                out.append(Period(body, kind, jd, lop, dar))
        return out

    def crosscheck(self, ev: Event) -> Optional[float]:
        """The same sighting from swe.heliacal_ut (Schaefer model), or None."""
        if abs(self.loc.lat) > 60.0 or ev.code not in SWE_EVENT:
            return None
        if ev.body == MOON:
            start = ev.anchor_jd - (3.0 if ev.is_lop else 0.0)
        else:
            start = ev.anchor_jd - (SCAN_DAYS[ev.body] if ev.is_lop else 0.0)
        try:
            dret = swe.heliacal_ut(start, self.loc.geopos, self.datm, self.dobs,
                                   swe_name(ev.body), SWE_EVENT[ev.code], self.eflag)
        except swe.Error:
            return None
        return dret[1] if dret[1] > 0 else dret[0]

    def run(self, year: int, bodies: Sequence[int] = ALL_BODIES,
            crosscheck: bool = False) -> Dict[int, List[Period]]:
        res = {}
        for b in bodies:
            periods = self.periods_for_body(b, year)
            if crosscheck:
                for p in periods:
                    for e in (p.lop, p.darshan):
                        if e:
                            e.swe_jd = self.crosscheck(e)
            res[b] = periods
        return res

    # ------------------------------------------------------------------ diagram grid
    def visibility_grid(self, body: int, year: int, step_min: int = 4):
        """Rows = days, columns = local time noon to noon. States: 0 below horizon,
        1 above horizon with the Sun, 2 above horizon but not seen, 3 seen. Positions are
        interpolated (fast); the tables use full-precision calls."""
        import numpy as np
        ndays = int(round(self.jd_local(year + 1, 1, 1) - self.jd_local(year, 1, 1)))
        ncol = 1440 // step_min
        t0 = self.jd_local(year, 1, 1, 12.0)
        tt = t0 + np.arange(ndays)[:, None] + (np.arange(ncol) * step_min / 1440.0)[None, :]

        def eq_track(b: int, hstep: float):
            ts = np.arange(t0 - 1.0, t0 + ndays + 2.0, hstep / 24.0)
            ra, dec = np.empty_like(ts), np.empty_like(ts)
            for i, t in enumerate(ts):
                x, _ = swe.calc_ut(float(t), b, self.eflag | swe.FLG_EQUATORIAL | swe.FLG_TOPOCTR)
                ra[i], dec[i] = x[0], x[1]
            ra = np.degrees(np.unwrap(np.radians(ra)))
            return (np.interp(tt.ravel(), ts, ra).reshape(tt.shape),
                    np.interp(tt.ravel(), ts, dec).reshape(tt.shape))

        lat = math.radians(self.loc.lat)
        lst = 280.46061837 + 360.98564736629 * (tt - 2451545.0) + self.loc.lon

        def altitude(ra, dec):
            h = np.radians(lst - ra)
            d = np.radians(dec)
            return np.degrees(np.arcsin(np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(h)))

        def refract(h):
            hh = np.clip(h, -1.0, 90.0)
            r = 1.02 / np.tan(np.radians(hh + 10.3 / (hh + 5.11))) / 60.0
            r *= (self.press / 1010.0) * (283.0 / (273.0 + self.temp))
            return np.where(h > -1.0, h + r, h)

        sun_true = altitude(*eq_track(SUN, 6.0))
        body_app = refract(altitude(*eq_track(body, 1.0 if body == MOON else 6.0)))
        sun_up = sun_true > -0.833
        up = body_app > 0.0
        state = np.zeros(tt.shape, dtype=np.int8)
        state[up & sun_up] = 1
        state[up & ~sun_up] = 2
        after_midnight = (np.arange(ncol) * step_min / 60.0) >= 12.0

        if body == MOON:
            bright = np.array([self.pheno(float(t0 + d + 0.5), MOON)[1] for d in range(ndays)])
            state[up & ~sun_up] = 3
            return state, bright, sun_true, t0, step_min

        mags = np.array([self.magnitude(float(t0 + d + 0.5), body) for d in range(ndays)])
        lo, hi = mags.min(), mags.max()
        bright = (hi - mags) / (hi - lo) if hi > lo else np.ones(ndays)
        codes_m = ("F",) if body in SUPERIOR else ("MF", "ML")
        codes_e = ("L",) if body in SUPERIOR else ("EF", "EL")
        p = self.st.planets[body]
        m = self.st.method
        if m in ("av-fixed", "av-variable"):
            cand = up & ~sun_up & (body_app >= p.critical_alt)
            av_m = np.array([min(self.av_value(body, c, mg) for c in codes_m) for mg in mags])
            av_e = np.array([min(self.av_value(body, c, mg) for c in codes_e) for mg in mags])
            av = np.where(after_midnight[None, :], av_m[:, None], av_e[:, None])
            vis = cand & (-sun_true >= av)
        elif m == "av-altaz":
            cand = up & ~sun_up & (body_app >= p.critical_alt)
            vis = cand & (sun_true < -18.0)
            for d, c in np.argwhere(cand & (sun_true >= -18.0)):
                r = self._vislim(float(tt[d, c]), body)
                if r is not None and r[3] < 0 and r[0] >= r[7]:
                    vis[d, c] = True
        else:
            # siddhanta methods: a whole morning / evening is seen or not, by the daily measure
            cand = up & ~sun_up
            seen_m = np.zeros(ndays, dtype=bool)
            seen_e = np.zeros(ndays, dtype=bool)
            for d in range(ndays):
                eve0 = float(t0 + d - 0.5)        # local midnight starting the row's first date
                mor0 = float(t0 + d + 0.5)        # local midnight in the middle of the row
                seen_e[d] = any(self._seen(self.measure(eve0, body, c)) for c in codes_e)
                seen_m[d] = any(self._seen(self.measure(mor0, body, c)) for c in codes_m)
            seen = np.where(after_midnight[None, :], seen_m[:, None], seen_e[:, None])
            vis = cand & seen
        state[vis] = 3
        return state, bright, sun_true, t0, step_min


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _n(x: Optional[float], nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return "-"
    v = round(x, nd) + 0.0
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _days(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x < 0:
        return "never lost"
    d = int(x)
    h = int(round((x - d) * 24))
    if h == 24:
        d, h = d + 1, 0
    return f"{d}d {h:02d}h"


def main_rows(eng: UdayaAstaEngine, periods: List[Period]) -> Tuple[List[str], List[list]]:
    """Obj | Lop | Lop DateTime | Sun udaya | Sun asta | Darshan | Darshan DateTime |
    Sun udaya | Sun asta | Lop days"""
    head = ["Obj", "Lop", "Lop DateTime", "Sun udaya", "Sun asta",
            "Darshan", "Darshan DateTime", "Sun udaya", "Sun asta", "Lop days"]
    rows = []
    for p in periods:
        row = [p.name]
        for e in (p.lop, p.darshan):
            if e is None:
                row += ["-", "not found", "-", "-"]
            else:
                su, sa = eng.sun_times(e.jd)
                row += [e.direction, eng.fmt_dt(e.jd), eng.fmt_time(su), eng.fmt_time(sa)]
        row.append(_days(p.days))
        rows.append(row)
    return head, rows


def detail_rows(eng: UdayaAstaEngine, periods: List[Period], crosscheck: bool) -> Tuple[List[str], List[list]]:
    m = eng.st.method
    measure_names = {"siddhanta": "kalamsha", "siddhanta-longitude": "dLon from Sun",
                     "av-fixed": "Sun depression", "av-variable": "Sun depression",
                     "av-altaz": "mag margin"}
    show_cross = eng.st.is_siddhanta     # the interpolated crossing is a siddhantic idea
    head = (["Obj", "event", "anchor", "anchor DateTime"] + (["limit crossed"] if show_cross else [])
            + ["last/first seen", "measure", "value", "limit", "mag", "phase %", "elong",
               "Sun lon", "obj lon", "obj lat", "dLon", "obj alt", "Sun alt",
               "Az obj", "Az sun", "dAz"])
    if crosscheck:
        head += ["swe_heliacal_ut", "diff d"]
    rows = []
    for p in periods:
        for e in (p.lop, p.darshan):
            if e is None:
                continue
            g = e.geo
            what = f"{'Lop' if e.is_lop else 'Darshan'} {e.direction}"
            moon_rule = p.body == MOON and m.startswith("av")
            limit_txt = ">= 0" if (m == "av-altaz" and not moon_rule) else _n(e.limit)
            what_measure = "crescent rule" if moon_rule else measure_names[m]
            row = ([p.name, what, p.anchor_kind, eng.fmt_dt(p.anchor_jd)]
                   + ([eng.fmt_dt(e.cross_jd)] if show_cross else [])
                   + [eng.fmt_dt(e.seen_jd), what_measure, _n(e.value), limit_txt, _n(g.get("mag"), 1),
                   _n(g["phase"] * 100.0, 1), _n(g.get("elong"), 1),
                   _n(eng.out_lon(e.jd, g["sun_lon"])), _n(eng.out_lon(e.jd, g["obj_lon"])),
                   _n(g.get("obj_lat"), 2, True), _n(norm180(g["obj_lon"] - g["sun_lon"]), 2, True),
                   _n(g.get("alt_obj")), _n(g.get("sun_alt")),
                   _n(g.get("az_obj"), 1), _n(g.get("az_sun"), 1),
                   _n(norm180(g["az_obj"] - g["az_sun"]), 1, True)])
            if crosscheck:
                if e.swe_jd is None:
                    row += ["-", "-"]
                else:
                    ref = e.seen_jd if e.seen_jd is not None else e.jd
                    diff = round(eng.local_midnight(e.swe_jd) - eng.local_midnight(ref))
                    row += [eng.fmt_dt(e.swe_jd), f"{diff:+d}"]
            rows.append(row)
    return head, rows


def dm(x: Optional[float]) -> str:
    """Degrees and minutes, rounded to the nearest minute, as PLSV prints them."""
    if x is None:
        return "-"
    a = int(round(abs(x) * 60.0))
    sign = "-" if x < 0 and a > 0 else ""
    return f"{sign}{a // 60}° {a % 60:02d}'"


def hm_signed(hours: Optional[float]) -> str:
    """PLSV style 'd r/s': -0:42h."""
    if hours is None:
        return "-"
    m = int(abs(hours) * 60.0 + 1e-9)
    return f"{'-' if hours < 0 else ''}{m // 60}:{m % 60:02d}h"


def age_dh(days: float) -> str:
    h = int(round(abs(days) * 24.0))          # hours rounded, as PLSV
    return f"{'-' if days < 0 else ''}{h // 24}d {h % 24:02d}h"


PLSV_LABELS = {"ML": "morning visibility ends", "EF": "evening visibility begins",
               "EL": "evening visibility ends", "MF": "morning visibility begins",
               "L": "last visibility", "F": "first visibility"}


def plsv_moon_rows(eng: "UdayaAstaEngine", periods: List[Period]) -> Tuple[List[str], List[list]]:
    """The Moon in PLSV's 'first and last visibility of the Moon' layout."""
    head = ["", "date", "sun r/s", "moon r/s", "sun lon", "moon lon", "moon lat", "moon alt",
            "moon phs", "d az", "d lon"]
    rows = []
    evs = sorted([e for p in periods for e in (p.lop, p.darshan) if e and p.body == MOON],
                 key=lambda e: e.geo.get("t_sun") or e.jd)
    for e in evs:
        g = e.geo
        ts = g.get("t_sun")
        rows.append(["L vis" if e.is_lop else "F vis", eng.fmt_date(ts), eng.fmt_time(ts),
                     eng.fmt_time(g.get("t_obj")), dm(eng.out_lon(ts, g["sun_lon"])),
                     dm(eng.out_lon(ts, g["obj_lon"])), dm(g["obj_lat"]), dm(g["alt_obj"]),
                     f"{g['phase'] * 100.0:.1f}%", dm(norm180(g["az_obj"] - g["az_sun"])),
                     dm(norm180(g["obj_lon"] - g["sun_lon"]))])
    return head, rows


def plsv_planet_rows(eng: "UdayaAstaEngine", periods: List[Period], body: int) -> Tuple[List[str], List[list]]:
    """A planet in PLSV's 'heliacal dates' layout (more information columns included)."""
    head = ["", "date", "obj r/s", "sun r/s", "d r/s", "age", "mag", "sun alt", "sun lon",
            "obj lon", "obj lat", "d az", "d lon"]
    rows = []
    evs = sorted([e for p in periods for e in (p.lop, p.darshan) if e and p.body == body],
                 key=lambda e: e.geo.get("t_obj") or e.jd)
    for e in evs:
        g = e.geo
        to, ts = g.get("t_obj"), g.get("t_sun")
        rows.append([PLSV_LABELS[e.code], eng.fmt_date(to), eng.fmt_time(to), eng.fmt_time(ts),
                     hm_signed(None if to is None or ts is None else (to - ts) * 24.0),
                     age_dh((to or e.jd) - e.anchor_jd), f"{g['mag']:.1f}", dm(g["sun_alt"]),
                     dm(eng.out_lon(to, g["sun_lon"])), dm(eng.out_lon(to, g["obj_lon"])),
                     dm(g["obj_lat"]), dm(norm180(g["az_obj"] - g["az_sun"])),
                     dm(norm180(g["obj_lon"] - g["sun_lon"]))])
    return head, rows


def plsv_tables(eng: "UdayaAstaEngine", results: Dict[int, List[Period]]) -> List[Tuple[str, List[str], List[list]]]:
    """PLSV-format tables (AV methods only), one per body."""
    if eng.st.is_siddhanta:
        return []
    out = []
    for b, periods in results.items():
        if not periods:
            continue
        y = eng.local((periods[0].lop or periods[0].darshan).jd)[0]
        if b == MOON:
            out.append((f"first and last visibility of the Moon", *plsv_moon_rows(eng, periods)))
        else:
            out.append((f"heliacal dates for {BODY_NAMES[b]}", *plsv_planet_rows(eng, periods, b)))
    return out


def text_table(head: List[str], rows: List[list]) -> str:
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(head)]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(head))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def limits_text(eng: UdayaAstaEngine, b: int) -> str:
    st = eng.st
    if st.is_siddhanta:
        k = st.kalamsha[b]
        if b in INFERIOR and k["ML"] == k["EF"] and k["EL"] == k["MF"]:
            return f"kalamsha {k['ML']:g} (direct: ML, EF), {k['EL']:g} (retrograde: EL, MF)"
        vals = sorted(set(k.values()))
        if len(vals) == 1:
            return f"kalamsha {vals[0]:g}"
        return "kalamsha " + ", ".join(f"{c} {k[c]:g}" for c in VALID_CODES[b])
    if b == MOON:
        return f"crescent rule: lower limb + {st.moon.coef:.3f} x distance >= {st.moon.vis_arc:g} deg"
    p = st.planets[b]
    if st.method == "av-fixed":
        names = {"MF": "morning first", "ML": "morning last", "EF": "evening first",
                 "EL": "evening last", "F": "heliacal rising", "L": "heliacal setting"}
        txt = "AV " + ", ".join(f"{names[c]} {p.av_events[c]:g}" for c in VALID_CODES[b])
    elif st.method == "av-variable":
        txt = f"AV {p.c:g} + {p.k:g} x mag"
    else:
        txt = f"twilight step {st.step_minutes:g} min, moonlight {'on' if st.moonlight else 'off'}"
    return f"{txt}; critical altitude {p.critical_alt:g} deg"


def settings_summary(eng: UdayaAstaEngine, year: int, bodies: Sequence[int]) -> List[Tuple[str, str]]:
    st, loc = eng.st, eng.loc
    rows = [("Location", f"{loc.name}  {abs(loc.lat):.4f} {'N' if loc.lat >= 0 else 'S'}  "
                         f"{abs(loc.lon):.4f} {'E' if loc.lon >= 0 else 'W'}  {loc.elev:.0f} m"),
            ("Year / time zone", f"{year}  /  UTC{loc.tz:+g} h"
                                 + ("" if (year > 1582 or st.gregorian) else ", Julian calendar")),
            ("Method", METHOD_TEXT[st.method]
             + (f" - {AV_PRESET_TEXT.get(st.av_preset, st.av_preset)} values"
                if st.method == "av-fixed" else "")),
            ("Sun udaya / asta", "Hindu (centre of disc, no refraction)" if st.hindu_sunrise
             else "standard (upper limb, refraction)"),
            ("Lop / Darshan time", "moment the limit is crossed (siddhantic)" if st.report_crossing
             else "actual last / first sighting"),
            ("Magnitudes", "Mueller (as PLSV)" if st.magnitudes == "plsv" else "Swiss Ephemeris")]
    if st.method == "av-altaz":
        rows.append(("Atmosphere", f"{eng.press:.1f} hPa, {eng.temp:.1f} C, RH {eng.atm.humidity:.0f} %"))
    for b in bodies:
        rows.append((BODY_NAMES[b], limits_text(eng, b)))
    rows.append(("Ephemeris", f"Swiss Ephemeris {swe.version}"))
    return rows


# --------------------------------------------------------------------------------------
# Console (Windows safe: plain ASCII text, and never stop on an unprintable character)
# --------------------------------------------------------------------------------------
def make_console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def report_text(eng: UdayaAstaEngine, year: int, results: Dict[int, List[Period]],
                crosscheck: bool, more: bool) -> str:
    """The console report as one string (also written by --txt)."""
    out = io.StringIO()
    for k, v in settings_summary(eng, year, list(results)):
        print(f"{k:18s}: {v}", file=out)
    periods = [p for b in results for p in results[b]]
    print(f"\n=== Udaya / Asta {year} - {eng.loc.name} ===", file=out)
    print(text_table(*main_rows(eng, periods)) if periods else "(nothing found)", file=out)
    for title, head, rows in plsv_tables(eng, results):
        print(f"\n=== PLSV format: {title} ===", file=out)
        print(text_table(head, rows), file=out)
    if more and periods:
        print("\n=== Details ===", file=out)
        print(text_table(*detail_rows(eng, periods, crosscheck)), file=out)
    return out.getvalue()


def print_report(text: str) -> None:
    """Print line by line and flush each line: some Windows terminals (PyCharm, ConPTY)
    drop part of a very large single write."""
    try:
        text.encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        text = text.replace("°", " ")          # a console that cannot show the degree sign
    for line in text.splitlines():
        print(line, flush=True)


# --------------------------------------------------------------------------------------
# Diagram
# --------------------------------------------------------------------------------------
DIAGRAM_COLORS = {
    "below": "#fcfcfb", "sun": "#e6e4dc", "hidden": "#a19f96",
    "ramp": ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281"],
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
}


def draw_diagram(eng: UdayaAstaEngine, body: int, year: int, periods: List[Period],
                 filename: Optional[str] = None, step_min: int = 4) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    state, bright, sun_true, t0, step = eng.visibility_grid(body, year, step_min)
    ndays, ncol = state.shape
    C = DIAGRAM_COLORS
    tt = t0 + np.arange(ndays)[:, None] + (np.arange(ncol) * step / 1440.0)[None, :]
    for p in periods:     # between Lop and Darshan the body is not seen
        if p.lop and p.darshan and p.darshan.jd > p.lop.jd:
            hide = (tt > p.lop.jd) & (tt < p.darshan.jd) & (state == 3)
            state[hide] = 2
    ramp = np.array([to_rgb(c) for c in C["ramp"]])
    img = np.empty(state.shape + (3,))
    img[state == 0] = to_rgb(C["below"])
    img[state == 1] = to_rgb(C["sun"])
    img[state == 2] = to_rgb(C["hidden"])
    pos = np.linspace(0.0, 1.0, len(ramp))
    day_rgb = np.stack([np.interp(bright, pos, ramp[:, i]) for i in range(3)], axis=1)
    vis_rgb = day_rgb[:, None, :].repeat(ncol, axis=1)
    img[state == 3] = vis_rgb[state == 3]

    fig, ax = plt.subplots(figsize=(10.5, 9.0), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=(12.0, 36.0, ndays, 0))
    hours = 12.0 + np.arange(ncol) * step / 60.0
    ax.contour(hours, np.arange(ndays) + 0.5, sun_true, levels=[-0.833], colors=C["ink2"],
               linewidths=0.9, linestyles="solid")

    def mark(t, marker, label):
        y, m, d, h = eng.local(t)
        doy = eng.jd_local(y, m, d) - eng.jd_local(year, 1, 1)
        x = h if h >= 12.0 else h + 24.0
        yrow = doy + (0.5 if h >= 12.0 else -0.5)
        if not (0 <= yrow <= ndays):
            return
        ax.plot([x], [yrow], marker=marker, markersize=8, color=C["ink"], markeredgecolor="#fcfcfb",
                markeredgewidth=1.2, linestyle="none", zorder=5)
        if label and body != MOON:
            ax.annotate(label, (x, yrow), xytext=(7, -3), textcoords="offset points",
                        fontsize=8.5, color=C["ink"], zorder=6)

    for p in periods:
        for e in (p.lop, p.darshan):
            if e is None:
                continue
            t = e.seen_jd if e.seen_jd is not None else (e.geo.get("t_sun") or e.jd)
            mark(t, "v" if e.is_lop else "^", "Lop" if e.is_lop else "Darshan")

    ax.set_xlim(12, 36)
    ax.set_xticks(np.arange(12, 37, 2))
    ax.set_xticklabels([f"{int(h) % 24:02d}" for h in np.arange(12, 37, 2)])
    month_starts = [eng.jd_local(year, mo, 1) - eng.jd_local(year, 1, 1) for mo in range(1, 13)]
    ax.set_yticks(month_starts)
    ax.set_yticklabels([calendar.month_abbr[mo] for mo in range(1, 13)])
    ax.tick_params(colors=C["muted"], labelcolor=C["ink2"], labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for ms in month_starts[1:]:
        ax.axhline(ms, color=C["grid"], linewidth=0.5, zorder=3, alpha=0.6)
    ax.set_xlabel(f"local time (UTC{eng.loc.tz:+g} h), noon to noon", color=C["ink2"], fontsize=9)
    ax.set_title(f"{BODY_NAMES[body]} - Udaya / Asta {year} - {eng.loc.name}", loc="left",
                 color=C["ink"], fontsize=13, pad=22)
    ax.text(0.0, 1.012, METHOD_TEXT[eng.st.method], transform=ax.transAxes, fontsize=9, color=C["ink2"])
    seen_label = "above horizon, Sun down (darker = fuller)" if body == MOON else "seen (darker = brighter)"
    handles = [Patch(facecolor=C["below"], edgecolor=C["grid"], label="below horizon"),
               Patch(facecolor=C["sun"], label="above horizon with the Sun"),
               Patch(facecolor=C["hidden"], label="above horizon, not seen"),
               Patch(facecolor=C["ramp"][5], label=seen_label),
               Line2D([], [], color=C["ink2"], linewidth=0.9, label="Sun udaya / asta"),
               Line2D([], [], marker="v", color=C["ink"], linestyle="none", markersize=7,
                      label="Lop (last sighting)"),
               Line2D([], [], marker="^", color=C["ink"], linestyle="none", markersize=7,
                      label="Darshan (first sighting)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              frameon=False, fontsize=8.5, labelcolor=C["ink2"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    data = buf.getvalue()
    if filename:
        with open(filename, "wb") as fh:
            fh.write(data)
    return data


# --------------------------------------------------------------------------------------
# HTML and CSV
# --------------------------------------------------------------------------------------
HTML_CSS = """
body{margin:0;background:#f9f9f7;color:#0b0b0b;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 8px}
p.sub{color:#52514e;margin:0 0 16px}
.card{background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:12px 14px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;text-align:left;white-space:nowrap;border-bottom:1px solid #e1e0d9}
th{color:#52514e;font-weight:600;background:#f4f3ef}
.scroll{overflow-x:auto}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:0}
dt{color:#52514e} dd{margin:0}
img{max-width:100%;height:auto;display:block;margin:8px 0}
.note{color:#52514e;font-size:13px}
"""


def _html_table(head: List[str], rows: List[list]) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def html_report(eng: UdayaAstaEngine, year: int, results: Dict[int, List[Period]],
                crosscheck: bool, diagrams: bool) -> str:
    periods = [p for b in results for p in results[b]]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Udaya Asta {year} - {html.escape(eng.loc.name)}</title>"
             f"<style>{HTML_CSS}</style></head><body><main>",
             f"<h1>Udaya / Asta {year} - {html.escape(eng.loc.name)}</h1>",
             "<p class='sub'>Lop (asta) and Darshan (udaya) of the Moon and planets, "
             "computed with the Swiss Ephemeris.</p><div class='card'><dl>"]
    for k, v in settings_summary(eng, year, list(results)):
        parts.append(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>")
    parts.append("</dl></div><h2>Lop and Darshan</h2>")
    parts.append(_html_table(*main_rows(eng, periods)) if periods else "<p class='note'>Nothing found.</p>")
    if periods:
        for title, head, rows in plsv_tables(eng, results):
            parts.append(f"<h2>PLSV format: {html.escape(title)}</h2>")
            parts.append(_html_table(head, rows))
        parts.append("<h2>Details</h2>")
        parts.append(_html_table(*detail_rows(eng, periods, crosscheck)))
    if diagrams:
        for b, per in results.items():
            parts.append(f"<h2>{BODY_NAMES[b]} - visibility through the year</h2>")
            try:
                png = draw_diagram(eng, b, year, per)
                parts.append(f"<img alt='{BODY_NAMES[b]} visibility diagram {year}' "
                             f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>")
            except ImportError:
                parts.append("<p class='note'>Install matplotlib and numpy for diagrams.</p>")
            except Exception as exc:
                parts.append(f"<p class='note'>Diagram could not be drawn: "
                             f"{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>")
    parts.append(
        "<h2>Notes</h2><div class='card note'><dl>"
        "<dt>Lop / Darshan</dt><dd>Lop (asta): the body is lost in the Sun's rays. Darshan "
        "(udaya): it is seen again. Purva = east (morning), Pashchim = west (evening).</dd>"
        "<dt>DateTime</dt><dd>Either the moment the body crosses its visibility limit, "
        "interpolated between the last day it is seen and the first day it is not (siddhantic, "
        "default for the siddhanta methods), or the actual last / first sighting (default for "
        "the AV methods). The details table shows both.</dd>"
        "<dt>Sun udaya / asta</dt><dd>Sunrise and sunset of the date of the Lop or Darshan.</dd>"
        "<dt>Lop days</dt><dd>Time from Lop to Darshan. 'never lost' means the body stays "
        "visible across the conjunction (Venus far from the ecliptic).</dd>"
        "<dt>kalamsha</dt><dd>Degrees of time between the body's and the Sun's rising (morning) "
        "or setting (evening); 4 minutes = 1 degree.</dd>"
        "<dt>Geometry in the details</dt><dd>Planets: at the planet's rise/set (critical "
        "altitude). Moon with the AV methods: at sunrise/sunset, as in PLSV; 'last/first seen' "
        "is moonrise/moonset. Altitudes are topocentric and apparent; azimuths from north "
        "through east.</dd>"
        "<dt>Times</dt><dd>Seconds are dropped (09:41:52 shows as 09:41), as in PLSV; use "
        "--round-minutes to round.</dd></dl></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def write_csv(eng: UdayaAstaEngine, results: Dict[int, List[Period]], crosscheck: bool,
              filename: str) -> None:
    periods = [p for b in results for p in results[b]]
    with open(filename, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        head, rows = main_rows(eng, periods)
        w.writerow(head)
        w.writerows(rows)
        for title, head, rows in plsv_tables(eng, results):
            w.writerow([])
            w.writerow([f"PLSV format: {title}"])
            w.writerow(head)
            w.writerows(rows)
        w.writerow([])
        head, rows = detail_rows(eng, periods, crosscheck)
        w.writerow(head)
        w.writerows(rows)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def _parse_planet_values(items: Optional[List[str]], option: str, bodies: Sequence[int]):
    """PLANET=VALUE (all its events) or PLANET.EVENT=VALUE -> {body: {code or '*': value}}"""
    out: Dict[int, Dict[str, float]] = {}
    for it in items or []:
        try:
            key, val = it.split("=")
            value = float(val)
            if "." in key or ":" in key:
                name, code = key.replace(":", ".").split(".")
                b = BODY_BY_NAME[name.strip().lower()]
                code = next(c for c in VALID_CODES[b] if c.lower() == code.strip().lower())
            else:
                b, code = BODY_BY_NAME[key.strip().lower()], "*"
            if b not in bodies:
                raise KeyError(b)
            out.setdefault(b, {})[code] = value
        except (ValueError, KeyError, StopIteration):
            raise SystemExit(f"--{option} expects PLANET=DEG or PLANET.EVENT=DEG (events: ML EF EL MF "
                             f"for Mercury/Venus, L F for Mars/Jupiter/Saturn, ML EF for the Moon) "
                             f"- got {it!r}")
    return out


def _parse_crit(items: Optional[List[str]]) -> Dict[int, float]:
    out = {}
    for it in items or []:
        try:
            name, val = it.split("=")
            out[BODY_BY_NAME[name.strip().lower()]] = float(val)
        except (ValueError, KeyError):
            raise SystemExit(f"--crit expects PLANET=DEG, e.g. venus=1 (got {it!r})")
    return out


def default_argv() -> List[str]:
    import datetime
    here = os.path.dirname(os.path.abspath(__file__))
    d = DEFAULT_RUN
    year = d["year"] or datetime.date.today().year
    args = ["--name", str(d["name"]), "--lat", str(d["lat"]), "--lon", str(d["lon"]),
            "--elev", str(d["elev"]), "--tz", str(d["tz"]), "--year", str(year),
            "--method", d["method"], "--bodies", d["bodies"],
            "--crit-alt", str(d.get("crit_alt", 0.0))]
    for key, val in (d.get("kalamsha") or {}).items():
        args += ["--kalamsha", f"{key}={val}"]
    if d.get("av_preset"):
        args += ["--av-preset", d["av_preset"]]
    for key, val in (d.get("av") or {}).items():
        args += ["--av", f"{key}={val}"]
    if d.get("sunrise"):
        args += ["--sunrise", d["sunrise"]]
    if d.get("ayanamsa"):
        args += ["--ayanamsa", d["ayanamsa"]]
    if d.get("time"):
        args += ["--time", d["time"]]
    if d.get("ephe"):
        ephe = d["ephe"] if os.path.isabs(d["ephe"]) else os.path.join(here, d["ephe"])
        if os.path.isdir(ephe):
            args += ["--ephe", ephe]
        else:
            print(f"Note: ephemeris folder {ephe} not found - using the built-in Moshier "
                  f"ephemeris. Put sepl_18.se1 and semo_18.se1 there for full precision.\n")
    for key in ("html", "csv", "txt"):
        if d.get(key):
            args += [f"--{key}", d[key] if os.path.isabs(d[key]) else os.path.join(here, d[key])]
    if d.get("diagrams"):
        args.append("--diagrams")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    make_console_safe()
    if argv is None and len(sys.argv) == 1:
        argv = default_argv()
        print("No arguments given - using DEFAULT_RUN from the top of the script:")
        print("  " + " ".join(argv) + "\n")
    ap = argparse.ArgumentParser(description="Udaya / Asta (Darshan / Lop) of the Moon and planets "
                                             "on the Swiss Ephemeris.")
    ap.add_argument("--name", default="Location")
    ap.add_argument("--lat", type=float, required=True, help="latitude, north positive")
    ap.add_argument("--lon", type=float, required=True, help="longitude, east positive")
    ap.add_argument("--elev", type=float, default=0.0, help="elevation in metres")
    ap.add_argument("--tz", type=float, default=0.0, help="hours east of UTC (IST = 5.5)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--bodies", default="moon,mercury,venus,mars,jupiter,saturn")
    ap.add_argument("--method", default="siddhanta", choices=METHODS + tuple(METHOD_ALIASES),
                    help="siddhanta (default), siddhanta-longitude, av-fixed, av-variable, av-altaz")
    ap.add_argument("--kalamsha", action="append", metavar="PLANET[.EVENT]=DEG",
                    help="siddhanta: override, e.g. mercury=13 or venus.EL=9")
    ap.add_argument("--av-preset", choices=sorted(AV_PRESETS), default="plsv",
                    help="av-fixed starting values: plsv (Schoch 1928, PLSV default) or ptolemy")
    ap.add_argument("--av", action="append", metavar="PLANET[.EVENT]=DEG",
                    help="av-fixed overrides: venus=5 (all events) or mercury.MF=13")
    ap.add_argument("--crit-alt", type=float, default=0.0, help="AV methods: critical altitude")
    ap.add_argument("--crit", action="append", metavar="PLANET=DEG", help="per-planet critical altitude")
    ap.add_argument("--var-c", type=float, default=10.5, help="av-variable: AV = c + k*mag")
    ap.add_argument("--var-k", type=float, default=1.4)
    ap.add_argument("--sunrise", choices=("hindu", "standard"), default=None,
                    help="Sun udaya/asta convention (default: hindu for siddhanta, standard for AV)")
    ap.add_argument("--time", choices=("crossing", "sighting"), default=None,
                    help="Lop/Darshan moment: limit crossing (default for siddhanta) or actual "
                         "last/first sighting (default for AV methods)")
    ap.add_argument("--magnitude", choices=("plsv", "swe"), default=None,
                    help="magnitude formulas: plsv (Mueller, as PLSV; default) or swe "
                         "(Swiss Ephemeris; default for av-altaz)")
    ap.add_argument("--round-minutes", action="store_true",
                    help="round times to the nearest minute (default: drop the seconds, as PLSV)")
    ap.add_argument("--no-refraction", action="store_true", help="AV methods: geometric horizon")
    ap.add_argument("--step", type=float, default=2.0, help="av-altaz: twilight step in minutes")
    ap.add_argument("--moonlight", action="store_true", help="av-altaz: include moonlight")
    ap.add_argument("--moon-arc", type=float, default=11.3, help="AV methods: Moon crescent rule")
    ap.add_argument("--moon-coef", type=float, default=1.0 / 3.0)
    ap.add_argument("--pressure", type=float, default=None)
    ap.add_argument("--temp", type=float, default=15.0)
    ap.add_argument("--humidity", type=float, default=40.0)
    ap.add_argument("--extinction", type=float, default=0.0)
    ap.add_argument("--age", type=float, default=36.0)
    ap.add_argument("--snellen", type=float, default=1.0)
    ap.add_argument("--ayanamsa", choices=sorted(AYANAMSAS), default=None)
    ap.add_argument("--gregorian", action="store_true")
    ap.add_argument("--ephe", default=None, help="folder with Swiss Ephemeris .se1 files")
    ap.add_argument("--crosscheck", action="store_true", help="compare sightings with swe.heliacal_ut")
    ap.add_argument("--more", action="store_true", help="console: also print the detail table")
    ap.add_argument("--html", help="write an HTML report")
    ap.add_argument("--csv", help="write a CSV")
    ap.add_argument("--txt", help="write the console report (all tables) to a UTF-8 text file")
    ap.add_argument("--diagrams", action="store_true", help="add visibility diagrams to the HTML")
    ap.add_argument("--png-dir", help="also save the diagrams as PNG files")
    a = ap.parse_args(argv)

    try:
        bodies = [BODY_BY_NAME[n.strip().lower()] for n in a.bodies.split(",") if n.strip()]
        bodies = [b for b in bodies if b != SUN]
    except KeyError as exc:
        raise SystemExit(f"unknown body {exc}; use moon, mercury, venus, mars, jupiter, saturn")

    planets = default_planet_params(a.av_preset)
    crit = _parse_crit(a.crit)
    av = _parse_planet_values(a.av, "av", PLANETS)
    for b, p in planets.items():
        p.critical_alt = crit.get(b, a.crit_alt)
        p.c, p.k = a.var_c, a.var_k
        vals = av.get(b, {})
        if "*" in vals:
            p.av_events = {c: vals["*"] for c in p.av_events}
        p.av_events.update({c: v for c, v in vals.items() if c != "*"})
    kal = default_kalamsha()
    for b, vals in _parse_planet_values(a.kalamsha, "kalamsha", ALL_BODIES).items():
        if "*" in vals:
            kal[b] = {c: vals["*"] for c in kal[b]}
        kal[b].update({c: v for c, v in vals.items() if c != "*"})

    st = Settings(method=METHOD_ALIASES.get(a.method, a.method), planets=planets, kalamsha=kal,
                  moon=MoonParams(a.moon_arc, a.moon_coef), sunrise=a.sunrise,
                  refraction=not a.no_refraction, step_minutes=a.step, moonlight=a.moonlight,
                  ayanamsa=a.ayanamsa, gregorian=a.gregorian, time_mode=a.time,
                  av_preset=a.av_preset, round_minutes=a.round_minutes, magnitudes=a.magnitude)
    eng = UdayaAstaEngine(Location(a.name, a.lat, a.lon, a.elev, a.tz), st,
                          Atmosphere(a.pressure, a.temp, a.humidity, a.extinction),
                          Observer(a.age, a.snellen), ephe_path=a.ephe)
    results = eng.run(a.year, bodies, crosscheck=a.crosscheck)

    written = []                       # files first, then the console
    if a.html:
        path = os.path.abspath(a.html)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_report(eng, a.year, results, a.crosscheck, a.diagrams))
        written.append(f"HTML report written to {path}")
    if a.csv:
        path = os.path.abspath(a.csv)
        write_csv(eng, results, a.crosscheck, path)
        written.append(f"CSV written to {path}")
    if a.png_dir:
        folder = os.path.abspath(a.png_dir)
        os.makedirs(folder, exist_ok=True)
        for b, per in results.items():
            draw_diagram(eng, b, a.year, per,
                         filename=os.path.join(folder, f"{BODY_NAMES[b].lower()}_{a.year}.png"))
        written.append(f"Diagrams saved in {folder}")
    text = report_text(eng, a.year, results, a.crosscheck, a.more)
    if a.txt:
        path = os.path.abspath(a.txt)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(f"Text report written to {path}")
    print_report(text)
    if written:
        print("\n" + "\n".join(written))


if __name__ == "__main__":
    main()
