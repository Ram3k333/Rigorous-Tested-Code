"""
Panchanga boundaries in Julian Days (UT) -- LOCATION FREE
=========================================================
swisseph only SAMPLES the Sun and Moon (coarse steps, with speed).
All boundary times come from NumPy: cubic Hermite curve in each time cell + Newton.

EPHEMERIS
  "swieph" : Swiss Ephemeris .se1 files (sepl_*.se1 for the Sun, semo_*.se1 for the Moon)
  "jpl"    : JPL DE file (de440.eph / de441.eph ...) read directly by swisseph
  "moseph" : built-in Moshier, no files needed
  With REQUIRE_EPHEMERIS = True every single swisseph call is checked: if swisseph had to
  fall back (file missing, date outside the file's range) the run stops with a clear message.

ONE GRID PER LIMB (angle in "grid units", 1 unit = 1 cell)
  Moon - Sun (6 deg)          -> Karana ;  Tithi = every 2nd line ; Paksha = every 30th
  Moon sidereal (3 deg 20')   -> Pada   ;  Nakshatra = every 4th, Rashi = every 9th
  Moon + Sun sidereal (13d20')-> Yoga
  Sun sidereal (30 deg)       -> Sankranti, nirayana Ritu (60 deg)
  Sun tropical (90 / 60 deg)  -> Gola, Ayana, sayana Ritu   (separate call, no sidereal flag)

DERIVED (no extra ephemeris work, only Amavasya + Sankranti lists)
  Masa (amanta)     : Amavasya -> Amavasya, named by the Sankranti inside it
                      0 Sankranti = Adhika, 2 Sankranti = Kshaya
  Masa-Purn         : Purnimanta name (Krishna half takes the next month's name)
  Shaka, Vikram     : Vikram from Chaitra or Kartika
  Samvatsara        : north (Barhaspatya) or south cycle

FIXED CLOCK TIME
  Surya spashta at given local clock times (e.g. 05:30, 06:00 IST), one row per day per time.

NOT HERE (location dependent): sunrise, sunset, dinmaan, vaar.
"""
import math
import os
import time
import csv
import numpy as np
import swisseph as swe

# =====================================================================
#                              CONFIG
# =====================================================================
# ---- time range -------------------------------------------------------
# Give START and END as a date-time tuple OR as a Julian Day (UT). A JD wins if set.
START_DATE = (2024, 4, 8, 0.0)    # (year, month, day, hour)
END_DATE   = (2026, 4, 10, 0.0)    # (year, month, day, hour)  -> end is exclusive
START_JD   = None                  # e.g. 2461314.5  (overrides START_DATE)
END_JD     = None                  # e.g. 2461344.5  (overrides END_DATE)
DATES_ARE_LOCAL = True             # True -> START_DATE / END_DATE are local (TZ_OFFSET_HOURS).
                                   # START_JD / END_JD are always UT.
PAD        = 3.0                   # days computed outside the range (edge safety)

# ---- time display -----------------------------------------------------
SHOW_JD    = True
SHOW_IST   = True
TZ_OFFSET_HOURS = 5.5
TZ_LABEL   = "IST"

# ---- swisseph FLAGS ----------------------------------------------------
AYANAMSA   = "LAHIRI"              # any swe.SIDM_<name>
EPHEMERIS  = "jpl"              # "swieph" (SE .se1 files), "jpl" (JPL DE file) or "moseph" (built-in, ~1" less accurate)
EPHE_PATH  =  r"C:\Users\durva\PycharmProjects\Rigorous_Tested_Code"   # folder holding the .se1 files (sepl_18 + semo_18 for 1800-2400, *_24 for 2400-3000) and the JPL file if EPHEMERIS="jpl"
JPL_FILE   = "de440.eph"           # file name inside EPHE_PATH; de441 covers -13200..17191, de440 covers 1550..2650
REQUIRE_EPHEMERIS = True           # True -> stop if swisseph silently falls back to a lower-quality ephemeris
TRUE_POSITION      = False         # True -> geometric (no light-time)
NO_ABERRATION      = False         # True -> skip annual aberration
NO_GRAV_DEFLECTION = False         # True -> skip Sun light deflection
# (no TOPOCENTRIC option on purpose: it would make the result location dependent)

# ---- which events ------------------------------------------------------
SHOW_TITHI       = False
SHOW_KARANA      = False
SHOW_PAKSHA      = False
SHOW_NAKSHATRA   = False
SHOW_PADA        = False
SHOW_RASHI       = False             # Chandra rashi
SHOW_YOGA        = False
SHOW_SANKRANTI   = True             # Sun enters sidereal rashi
SHOW_GOLA        = True             # Uttara / Dakshina gola  (sayana)
SHOW_AYANA       = True             # Uttarayana / Dakshinayana (sayana)
SHOW_RITU_SAYANA   = True           # ritu from tropical Sun
SHOW_RITU_NIRAYANA = False          # ritu from sidereal Sun
RITU_SAYANA_START   = 330.0        # tropical longitude where Vasanta begins
RITU_NIRAYANA_START = 0.0          # sidereal longitude where Vasanta begins

# ---- lunar calendar: masa / year ---------------------------------------
SHOW_MASA            = True        # Masa (amanta) rows
SHOW_MASA_PURN       = False        # Purnimanta masa rows
SHOW_YEAR            = True        # Shaka, Vikram, Samvatsara rows
SAMVATSARA_CYCLE     = "north"     # "north" (Barhaspatya, mean Jupiter) or "south" (from Shaka)
VIKRAM_START         = "chaitra"   # "chaitra" or "kartika" (Gujarat)
YEAR_START_AT_ADHIKA = True        # True: an Adhika Chaitra/Kartika opens the year (shastra rule)
                                   # False: the year opens with the Nija month
NORTH_SHIFT_DAYS     = 3.9         # calibration of the north cycle (see samvatsara_north)
MASA_PAD_BACK = 400.0              # days before START: to find the Chaitra that opened the year
MASA_PAD_FWD  = 400.0              # days after END: to find the Chaitra that closes the year

# ---- Surya spashta at fixed local clock times ---------------------------
SUN_FIXED_TIMES = []   # local "HH:MM" (TZ_OFFSET_HOURS); [] = off

# ---- labels / output ---------------------------------------------------
EDGE_MODE      = "overlap"         # "overlap"  = every item that overlaps the range
                                   # "starts_in" = only items whose START is inside the range
NUMBER_LABELS  = False             # "Pratipada (1)", "Makara (10)"
RASHI_LANGUAGE = "sanskrit"        # "sanskrit" or "english"
SORT_BY        = "type"            # "time" or "type"
JD_DECIMALS    = 8
PRINT_TABLE    = True
PRINT_SUN      = True              # Surya spashta at SUN_FIXED_TIMES
PRINT_TIME     = True
PRINT_CALLS    = True
PRINT_EPHEM    = True              # print which ephemeris / file was actually used
SAVE_CSV       = True
CSV_NAME       = "panchanga_jd.csv"

# ---- sampling / accuracy ----------------------------------------------
MOON_STEP = 0.5                    # days. 1.0 = half the calls, ~1 s error;  0.5 ~ 0.05 s
SUN_STEP  = 4.0                    # days. Sun is smooth; interpolated onto the Moon grid
POLISH    = False                  # True -> re-solve every Moon-grid root on the REAL ephemeris
POLISH_MAX_SHIFT = 0.05            # days; bigger shift from the fitted root is rejected

# ---- verification ------------------------------------------------------
RUN_CHECK  = True                  # continuity + ephemeris audit + masa audit
AUDIT_STEP = 5.0                   # days between true-ephemeris audit samples (0 = off)
STRICT     = False                 # True -> any warning is a hard error
# ======================== END CONFIG ================================

SUN, MOON = swe.SUN, swe.MOON

# ---- FLAGS -------------------------------------------------------------
_EPH_FLAG = {"swieph": swe.FLG_SWIEPH, "jpl": swe.FLG_JPLEPH, "moseph": swe.FLG_MOSEPH}
if EPHEMERIS not in _EPH_FLAG:
    raise ValueError(f'EPHEMERIS must be "swieph", "jpl" or "moseph" (got {EPHEMERIS!r})')
EPH_WANT = _EPH_FLAG[EPHEMERIS]
EPH_MASK = swe.FLG_JPLEPH | swe.FLG_SWIEPH | swe.FLG_MOSEPH
EPH_NAME = {swe.FLG_JPLEPH: "JPL", swe.FLG_SWIEPH: "Swiss Ephemeris", swe.FLG_MOSEPH: "Moshier"}

BASE = EPH_WANT | swe.FLG_SPEED
if TRUE_POSITION:      BASE |= swe.FLG_TRUEPOS
if NO_ABERRATION:      BASE |= swe.FLG_NOABERR
if NO_GRAV_DEFLECTION: BASE |= swe.FLG_NOGDEFL
FLAGS_SID  = BASE | swe.FLG_SIDEREAL      # Moon, Sun (sidereal)
FLAGS_TROP = BASE                         # Sun (tropical, of date)

# ---- what is needed ----------------------------------------------------
NEED_ELONG = SHOW_TITHI or SHOW_KARANA or SHOW_PAKSHA or SHOW_MASA
NEED_MOON  = SHOW_NAKSHATRA or SHOW_PADA or SHOW_RASHI
NEED_YOGA  = SHOW_YOGA
NEED_MOON_SERIES = NEED_ELONG or NEED_MOON or NEED_YOGA
NEED_SANKRANTI = SHOW_SANKRANTI or SHOW_MASA
NEED_SUN_SID  = NEED_MOON_SERIES or NEED_SANKRANTI or SHOW_RITU_NIRAYANA
NEED_SUN_TROP = SHOW_GOLA or SHOW_AYANA or SHOW_RITU_SAYANA

# Slow Sun items need their start/end even when those lie outside the range, so the Sun
# series is extended by the longest possible length of the item (days).
SUN_EXTRA = max(33 if NEED_SANKRANTI else 0,
                66 if (SHOW_RITU_SAYANA or SHOW_RITU_NIRAYANA) else 0,
                190 if (SHOW_GOLA or SHOW_AYANA) else 0)

# ---- grid widths (degrees) --------------------------------------------
W_ELONG, W_PADA, W_YOGA = 6.0, 10.0 / 3.0, 40.0 / 3.0
GRID_N = {"elong": 60, "moon": 108, "yoga": 27, "sun_sid": 12,
          "sun_trop": 4, "ritu_say": 6, "ritu_nir": 6}

# ---- names -------------------------------------------------------------
NAKSHATRA = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
             "Punarvasu", "Pushya", "Ashlesha", "Magha", "P.Phalguni", "U.Phalguni",
             "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula",
             "P.Ashadha", "U.Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
             "P.Bhadrapada", "U.Bhadrapada", "Revati"]
RASHI_SA = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
            "Tula", "Vrishchika", "Dhanu", "Makara", "Kumbha", "Meena"]
RASHI_EN = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
RASHI = RASHI_SA if RASHI_LANGUAGE == "sanskrit" else RASHI_EN
TITHI = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
         "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
         "Trayodashi", "Chaturdashi"]            # 15 = Purnima, 30 = Amavasya
KARANA_MOVABLE = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]
YOGA = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda",
        "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata",
        "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan", "Parigha", "Shiva",
        "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
RITU = ["Vasanta", "Grishma", "Varsha", "Sharad", "Hemanta", "Shishira"]
# Masa index = index of the rashi the Sun ENTERS during the month (Mesha -> Chaitra)
MASA = ["Chaitra", "Vaishakha", "Jyeshtha", "Ashadha", "Shravana", "Bhadrapada",
        "Ashwin", "Kartika", "Margashirsha", "Pausha", "Magha", "Phalguna"]
SAMVATSARA = ["Prabhava", "Vibhava", "Shukla", "Pramoda", "Prajapati", "Angirasa",
              "Shrimukha", "Bhava", "Yuva", "Dhatri", "Ishvara", "Bahudhanya",
              "Pramathi", "Vikrama", "Vrisha", "Chitrabhanu", "Subhanu", "Tarana",
              "Parthiva", "Vyaya", "Sarvajit", "Sarvadhari", "Virodhi", "Vikriti",
              "Khara", "Nandana", "Vijaya", "Jaya", "Manmatha", "Durmukha",
              "Hemalamba", "Vilambi", "Vikari", "Sharvari", "Plava", "Shubhakrit",
              "Shobhakrit", "Krodhi", "Vishvavasu", "Parabhava", "Plavanga", "Kilaka",
              "Saumya", "Sadharana", "Virodhikrit", "Paridhavi", "Pramadi", "Ananda",
              "Rakshasa", "Nala", "Pingala", "Kalayukta", "Siddharthi", "Raudra",
              "Durmati", "Dundubhi", "Rudhirodgari", "Raktakshi", "Krodhana", "Akshaya"]


def lab(name, i):
    return f"{name} ({i + 1})" if NUMBER_LABELS else name


def tithi_label(i):                     # i = 0..29
    if i == 14: name = "Purnima"
    elif i == 29: name = "Amavasya"
    else: name = ("Shukla " if i < 15 else "Krishna ") + TITHI[i % 15]
    return lab(name, i)


def karana_label(i):                    # i = 0..59  (karana number = i + 1)
    n = i + 1
    if n == 1: name = "Kimstughna"
    elif n >= 58: name = ["Shakuni", "Chatushpada", "Naga"][n - 58]
    else: name = KARANA_MOVABLE[(n - 2) % 7]
    return lab(name, i)


def masa_name(idx, idx2=None, adhika=False, kshaya=False):
    if kshaya:
        return f"{MASA[idx]}-{MASA[idx2]} (Kshaya)"
    return ("Adhika " if adhika else "") + MASA[idx]


def jd_to_local(jd):
    """JD (UT) -> 'YYYY-MM-DD HH:MM:SS' local. Integer seconds: no 60-s carry bugs."""
    total = int(round((jd + TZ_OFFSET_HOURS / 24.0 + 0.5) * 86400.0))
    day, sod = divmod(total, 86400)
    y, m, d, _ = swe.revjul(float(day))
    return f"{y:04d}-{m:02d}-{d:02d} {sod // 3600:02d}:{sod % 3600 // 60:02d}:{sod % 60:02d}"


def local_year(jd):
    return swe.revjul(jd + TZ_OFFSET_HOURS / 24.0)[0]


# ----------------------------------------------------- ephemeris wrappers
CALLS = 0
WARN = []
EPH_USED = {}            # ephemeris actually used -> number of calls
_FALLBACK_SEEN = False


def _calc(jd, body, flags):
    """(longitude, speed). Counted. Works with old and new pyswisseph return shapes.
    Checks the RETURN flag: swisseph silently falls back JPL -> SWIEPH -> MOSEPH when a
    file is missing or the date is outside the file's range."""
    global CALLS, _FALLBACK_SEEN
    CALLS += 1
    try:
        r = swe.calc_ut(float(jd), body, flags)
    except swe.Error as e:
        raise RuntimeError(f"swisseph failed at JD {jd:.5f} (body {body}, EPHEMERIS={EPHEMERIS!r}): {e}") from None
    if isinstance(r[0], (tuple, list)):          # pyswisseph >= 2: (xx, retflag)
        xx, ret = r[0], r[1]
        got = ret & EPH_MASK
        EPH_USED[got] = EPH_USED.get(got, 0) + 1
        if got != EPH_WANT:
            msg = (f"{EPH_NAME[EPH_WANT]} requested but swisseph used {EPH_NAME.get(got, '?')} "
                   f"at JD {jd:.5f} (body {body}); check EPHE_PATH, the files in it"
                   + (f" and JPL_FILE={JPL_FILE!r}" if EPHEMERIS == "jpl" else "")
                   + " and the date range")
            if REQUIRE_EPHEMERIS:
                raise RuntimeError(msg)
            if not _FALLBACK_SEEN:
                WARN.append(msg)
                _FALLBACK_SEEN = True
    else:                                        # very old pyswisseph: xx only, no flag
        xx = r
        if not _FALLBACK_SEEN:
            WARN.append("old pyswisseph: ephemeris actually used cannot be verified")
            _FALLBACK_SEEN = True
    return xx[0], xx[3]


def sample(body, flags, t):
    lon = np.empty(len(t)); spd = np.empty(len(t))
    for i, jd in enumerate(t):
        lon[i], spd[i] = _calc(jd, body, flags)
    return lon, spd


def init():
    if EPHE_PATH:
        if not os.path.isdir(EPHE_PATH):
            raise FileNotFoundError(f"EPHE_PATH is not a folder: {EPHE_PATH}")
        swe.set_ephe_path(EPHE_PATH)
    if EPHEMERIS == "jpl":
        full = JPL_FILE if os.path.isabs(JPL_FILE) else os.path.join(EPHE_PATH or "", JPL_FILE)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"JPL file not found: {full}")
        swe.set_jpl_file(JPL_FILE)
    swe.set_sid_mode(getattr(swe, "SIDM_" + AYANAMSA))


def preflight(a, b):
    """Touch the ephemeris at both ends of the range that build() will sample, for the Sun
    and the Moon, so a missing file or an out-of-range date fails at once (not after
    thousands of calls). Returns a one-line description of what is in use."""
    ends = (a - SUN_EXTRA - 1.0, b + 2 * SUN_STEP + SUN_EXTRA + 1.0)
    bodies = [SUN] + ([MOON] if NEED_MOON_SERIES or POLISH else [])
    for jd in ends:
        for body in bodies:
            _calc(jd, body, FLAGS_SID)
    info = EPH_NAME[EPH_WANT]
    if EPHEMERIS == "jpl" and hasattr(swe, "get_current_file_data"):
        try:
            path, start, end, denum = swe.get_current_file_data(0)
            info += f" DE{denum}  {os.path.basename(path)}  covers JD {start:.1f} .. {end:.1f}"
        except Exception:
            info += f"  {JPL_FILE}"
    elif EPHEMERIS == "swieph":
        info += f"  path {EPHE_PATH or '(default)'}"
    return info


def wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


def unwrap(raw):
    """Continuous increasing angle. Starts at raw[0] (kept in 0..360) so k mod N is the real cell."""
    d = np.diff(raw) % 360.0
    return raw[0] + np.concatenate(([0.0], np.cumsum(d)))


# ------------------------------------------------- Hermite + crossings
def hermite(s, p0, p1, m0, m1):
    """Cubic Hermite value and d/ds. s in [0,1]; m0, m1 are slopes already times cell length."""
    s2, s3 = s * s, s * s * s
    H  = (2*s3 - 3*s2 + 1)*p0 + (s3 - 2*s2 + s)*m0 + (-2*s3 + 3*s2)*p1 + (s3 - s2)*m1
    dH = (6*s2 - 6*s)*p0 + (3*s2 - 4*s + 1)*m0 + (6*s - 6*s2)*p1 + (3*s2 - 2*s)*m1
    return H, dH


def hermite_eval(ts, f, v, tq):
    """Evaluate the (f, v) series at times tq. Returns value and slope per day."""
    i = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, len(ts) - 2)
    h = ts[i + 1] - ts[i]
    s = (tq - ts[i]) / h
    H, dH = hermite(s, f[i], f[i + 1], v[i] * h, v[i + 1] * h)
    return H, dH / h


def crossings(t, f, v, iters=4):
    """EVERY integer line of f (grid units) crossed by the increasing curve f(t).
    t: sample JDs; f: unwrapped angle in grid units; v: df/dt per day.
    Returns (jd, k) in time order."""
    h = np.diff(t); f0, f1 = f[:-1], f[1:]
    k_lo = np.floor(f0) + 1
    cnt = np.maximum(np.floor(f1) - k_lo + 1, 0).astype(int)
    cell = np.repeat(np.arange(len(h)), cnt)
    first = np.cumsum(cnt) - cnt
    k = k_lo[cell] + (np.arange(cnt.sum()) - first[cell])
    p0, p1 = f0[cell], f1[cell]
    m0, m1 = v[:-1][cell] * h[cell], v[1:][cell] * h[cell]
    s = (k - p0) / (p1 - p0)                       # linear first guess
    for _ in range(iters):
        H, dH = hermite(s, p0, p1, m0, m1)
        s = np.clip(s - (H - k) / dH, 0.0, 1.0)    # stays inside its cell
    return t[:-1][cell] + s * h[cell], k.astype(int)


# ----------------------------------------------------- root polishing
POLISH_W = {"elong": W_ELONG, "moon": W_PADA, "yoga": W_YOGA}


def polish(jd0, k, kind):
    """Re-solve on the REAL ephemeris (Newton with true speeds). 2 swisseph calls / iteration."""
    jd = jd0
    for _ in range(2):
        m, mv = _calc(jd, MOON, FLAGS_SID)
        s, sv = _calc(jd, SUN, FLAGS_SID)
        a, v = {"elong": (m - s, mv - sv), "moon": (m, mv), "yoga": (m + s, mv + sv)}[kind]
        diff = wrap180(a - k * POLISH_W[kind])
        if abs(diff) < 1e-9 or v == 0.0:
            break
        jd -= diff / v
    if abs(jd - jd0) > POLISH_MAX_SHIFT:
        WARN.append(f"polish moved {kind} k={k} by {jd - jd0:.4f} d; kept fitted root")
        return jd0
    return jd


# --------------------------------------------------------------- build
def build(a, b):
    """-> db: {grid_name: (jd array, k array)}"""
    db = {}
    mt = np.arange(a, b + MOON_STEP, MOON_STEP)
    st = np.arange(a - SUN_EXTRA, mt[-1] + 1.5 * SUN_STEP + SUN_EXTRA, SUN_STEP)

    s_u = sv = None
    if NEED_SUN_SID:
        s_raw, sv = sample(SUN, FLAGS_SID, st)
        s_u = unwrap(s_raw)

    if NEED_MOON_SERIES:
        m_raw, mv = sample(MOON, FLAGS_SID, mt)
        m_u = unwrap(m_raw)
        s_g, sv_g = hermite_eval(st, s_u, sv, mt)          # Sun on the Moon grid (no new calls)
        jobs = []
        if NEED_ELONG: jobs.append(("elong", m_u - s_g, mv - sv_g, W_ELONG))
        if NEED_MOON:  jobs.append(("moon", m_u, mv, W_PADA))
        if NEED_YOGA:  jobs.append(("yoga", m_u + s_g, mv + sv_g, W_YOGA))
        for name, f, v, w in jobs:
            jd, k = crossings(mt, f / w, v / w)
            if POLISH:
                jd = np.array([polish(j, int(kk), name) for j, kk in zip(jd, k)])
            db[name] = (jd, k)

    if NEED_SANKRANTI:
        db["sun_sid"] = crossings(st, s_u / 30.0, sv / 30.0)
    if SHOW_RITU_NIRAYANA:
        db["ritu_nir"] = crossings(st, (s_u - RITU_NIRAYANA_START) / 60.0, sv / 60.0)

    if NEED_SUN_TROP:                                       # separate call, no sidereal flag
        t_raw, tv = sample(SUN, FLAGS_TROP, st)
        t_u = unwrap(t_raw)
        if SHOW_GOLA or SHOW_AYANA:
            db["sun_trop"] = crossings(st, t_u / 90.0, tv / 90.0)
        if SHOW_RITU_SAYANA:
            db["ritu_say"] = crossings(st, (t_u - RITU_SAYANA_START) / 60.0, tv / 60.0)
    return db


# ------------------------------------------------- masa / paksha / year
# Surya Siddhanta mean Jupiter: 364,220 revolutions in 1,577,917,828 civil days of a
# Mahayuga, all planets at 0 deg at the Kali epoch.  One Barhaspatya year = 1/12 revolution.
KALI_EPOCH_JD = 588465.5
SS_JUP_REV, SS_CIVIL_DAYS = 364220, 1577917828


def samvatsara_north(jd):
    """Barhaspatya samvatsara running at jd (0 = Prabhava). The name current at the
    start of Chaitra Shukla Pratipada is used for the whole year (north convention).
    Calibrated (NORTH_SHIFT_DAYS, valid 2.8..5.0) so VS 2074..2083 give Sadharana,
    Virodhikrit, Paridhavi, Pramadi, Rakshasa (Ananda lupta),
    Nala, Pingala, Kalayukta, Siddharthi, Raudra."""
    n = math.floor((jd + NORTH_SHIFT_DAYS - KALI_EPOCH_JD) * 12 * SS_JUP_REV / SS_CIVIL_DAYS)
    return (n + 26) % 60


def samvatsara_south(shaka):
    """South cycle: fixed to the Shaka year (Shaka 1947 = Vishvavasu)."""
    return (shaka + 11) % 60


def build_months(db):
    """Amanta months from the Amavasya and Sankranti lists. No ephemeris calls."""
    jd, k = db["elong"]
    nm = jd[k % 60 == 0]                  # Amavasya ends = new moon
    fm = jd[k % 60 == 30]                 # Purnima ends = full moon
    sj, sk = db["sun_sid"]
    sr = sk % 12                          # rashi the Sun enters
    months = []
    for a, b in zip(nm[:-1], nm[1:]):
        i0, i1 = int(np.searchsorted(sj, a)), int(np.searchsorted(sj, b))
        n = i1 - i0
        if n == 1:
            idx, idx2, adh, ksh = int(sr[i0]), None, False, False
        elif n == 0:
            if i0 >= len(sj):
                continue                  # name unknown: next sankranti not computed
            idx, idx2, adh, ksh = int(sr[i0]), None, True, False   # takes the next month's name
        elif n == 2:
            idx, idx2, adh, ksh = int(sr[i0]), int(sr[i0 + 1]), False, True
        else:
            raise RuntimeError(f"{n} sankrantis in one lunar month at JD {a:.4f}")
        f = fm[(fm > a) & (fm < b)]   # Purnima instant
        months.append(dict(start=float(a), end=float(b), idx=idx, idx2=idx2, adh=adh,
                           ksh=ksh, n=n, full=float(f[0]) if len(f) else None))

    def has(m, target):
        return m["idx"] == target or (m["ksh"] and m["idx2"] == target)

    def opens(i, target):                 # does month i open a year that starts with `target`?
        m = months[i]
        if not has(m, target) or i == 0:
            return False
        if m["adh"]:
            return YEAR_START_AT_ADHIKA
        p = months[i - 1]
        return not (p["adh"] and p["idx"] == target and YEAR_START_AT_ADHIKA)

    shaka = samv = None
    after_kartika = False
    for i, m in enumerate(months):
        if opens(i, 0):
            shaka = local_year(m["start"]) - 78
            samv = (samvatsara_north(m["start"]) if SAMVATSARA_CYCLE == "north"
                    else samvatsara_south(shaka))
            after_kartika = False
        if opens(i, 7):
            after_kartika = True
        m["opens_shaka"] = opens(i, 0)
        m["opens_kartika"] = opens(i, 7)
        m["shaka"] = shaka
        m["samv"] = samv
        if shaka is None:
            m["vikram"] = None
        elif VIKRAM_START == "kartika":
            m["vikram"] = shaka + (135 if after_kartika else 134)
        else:
            m["vikram"] = shaka + 135
        m["name"] = masa_name(m["idx"], m["idx2"], m["adh"], m["ksh"])
        m["ritu"] = RITU[m["idx"] // 2]   # lunar ritu (kept for reference)
    return months


def build_pakshas(months):
    """Each half month with its Amanta and Purnimanta month name.
    Purnimanta: Shukla half = same name; Krishna half = name of the NEXT month
    (an Adhika month keeps its own name for both halves)."""
    out = []
    for i, m in enumerate(months):
        if m["full"] is None:
            continue
        nxt = months[i + 1] if i + 1 < len(months) else None
        if m["adh"]:
            p_name, p_adh = m["name"], True
        elif nxt is not None:
            p_name, p_adh = masa_name(nxt["idx"], nxt["idx2"], False, nxt["ksh"]), False
        else:
            p_name, p_adh = None, False
        out.append(dict(start=m["start"], end=m["full"], paksha="Shukla", m=m,
                        p_name=m["name"], p_adh=m["adh"]))
        out.append(dict(start=m["full"], end=m["end"], paksha="Krishna", m=m,
                        p_name=p_name, p_adh=p_adh))
    return out


def build_years(months):
    rows = []
    sh = [m for m in months if m["opens_shaka"]]
    for a, b in zip(sh, sh[1:] + [None]):
        rows.append(dict(system="shaka", number=a["shaka"], start=a["start"],
                         end=b["start"] if b else None,
                         samv=a["samv"]))
    vk = sh if VIKRAM_START == "chaitra" else [m for m in months if m["opens_kartika"]]
    for a, b in zip(vk, vk[1:] + [None]):
        if a["vikram"] is None:
            continue
        rows.append(dict(system="vikram_" + VIKRAM_START, number=a["vikram"], start=a["start"],
                         end=b["start"] if b else None,
                         samv=a["samv"] if VIKRAM_START == "chaitra" else None))
    return [r for r in rows if r["end"] is not None]


def overlap(items, lo, hi):
    if EDGE_MODE == "starts_in":
        return [r for r in items if lo <= r["start"] < hi]
    return [r for r in items if r["end"] > lo and r["start"] < hi]


# ------------------------------------------------- Surya spashta at fixed times
def dms_in_rashi(lon):
    sec = int(round((lon % 360.0) * 3600.0)) % (360 * 3600)
    deg, rem = divmod(sec, 3600)
    r = deg // 30
    return r, f"{RASHI[r]} {deg % 30:02d}°{rem // 60:02d}'{rem % 60:02d}\""


def sun_fixed(lo, hi):
    rows = []
    tz = TZ_OFFSET_HOURS / 24.0
    for d in range(math.floor(lo + tz + 0.5), math.floor(hi + tz + 0.5) + 1):
        y, mo, dd, _ = swe.revjul(float(d))
        for hhmm in SUN_FIXED_TIMES:
            hh, mm = map(int, hhmm.split(":"))
            jd = d - 0.5 + (hh + mm / 60.0) / 24.0 - tz
            if not (lo <= jd < hi):
                continue
            lon, spd = _calc(jd, SUN, FLAGS_SID)
            trop, _ = _calc(jd, SUN, FLAGS_TROP)
            r, txt = dms_in_rashi(lon)
            rows.append(dict(date=f"{y:04d}-{mo:02d}-{dd:02d}", time=hhmm, jd=jd, lon=lon,
                             speed=spd, rashi=r, spashta=txt, trop=trop))
    return rows


# ------------------------------------------------- verification layers
def check(db):
    """Crossings of one grid must chain k, k+1, k+2 ... in time order. A gap = skipped event."""
    for name, (jd, k) in db.items():
        assert np.all(np.diff(jd) > 0), f"{name}: times not increasing"
        assert np.all(np.diff(k) == 1), f"{name}: grid line skipped or repeated"


def audit(db, lo, hi):
    """Read the TRUE cell from swisseph every AUDIT_STEP days and compare with the cell
    the event list says we are in."""
    if AUDIT_STEP <= 0 or not db:
        return 0
    before = CALLS
    alljd = np.sort(np.concatenate([v[0] for v in db.values()]))
    for t in np.arange(lo, hi, AUDIT_STEP):
        t = float(t)
        for _ in range(5):                                  # step off exact event times
            i = int(np.searchsorted(alljd, t))
            near = (i < len(alljd) and alljd[i] - t < 2e-4) or (i > 0 and t - alljd[i - 1] < 2e-4)
            if not near:
                break
            t += 5e-4
        m = s = x = None
        if any(n in db for n in ("elong", "moon", "yoga", "sun_sid", "ritu_nir")):
            m = _calc(t, MOON, FLAGS_SID)[0]; s = _calc(t, SUN, FLAGS_SID)[0]
        if any(n in db for n in ("sun_trop", "ritu_say")):
            x = _calc(t, SUN, FLAGS_TROP)[0]
        true = {
            "elong":   lambda: ((m - s) % 360.0) / W_ELONG,
            "moon":    lambda: (m % 360.0) / W_PADA,
            "yoga":    lambda: ((m + s) % 360.0) / W_YOGA,
            "sun_sid": lambda: (s % 360.0) / 30.0,
            "ritu_nir": lambda: ((s - RITU_NIRAYANA_START) % 360.0) / 60.0,
            "sun_trop": lambda: (x % 360.0) / 90.0,
            "ritu_say": lambda: ((x - RITU_SAYANA_START) % 360.0) / 60.0,
        }
        for name, (jd, k) in db.items():
            i = int(np.searchsorted(jd, t, side="right")) - 1
            if i < 0:
                continue
            n = GRID_N[name]
            assert int(k[i]) % n == int(math.floor(true[name]())) % n, \
                f"audit {name}: event list says cell {int(k[i]) % n} but ephemeris disagrees at JD {t:.5f}"
    return CALLS - before


def audit_months(months):
    """Independent masa check (the classic drik method): Sun's rashi read from swisseph just
    after the opening and just before the closing new moon."""
    before = CALLS
    for m in months:
        r0 = int((_calc(m["start"] + 1e-5, SUN, FLAGS_SID)[0] % 360.0) // 30)
        r1 = int((_calc(m["end"] - 1e-5, SUN, FLAGS_SID)[0] % 360.0) // 30)
        assert (r1 - r0) % 12 == m["n"], f"masa audit: sankranti count wrong at JD {m['start']:.4f}"
        want = r1 if m["n"] == 1 else (r0 + 1) % 12
        assert m["idx"] == want, f"masa audit: name wrong at JD {m['start']:.4f}"
    return CALLS - before


# ---------------------------------------------------------- tables
def _segs(typ, jd, k, keep, label):
    """Items of one limb: START = its grid line, END = start of the next item of the same limb."""
    sel = [(float(j), label(int(kk))) for j, kk in zip(jd, k) if keep(int(kk))]
    return [(sel[i][0], sel[i + 1][0], typ, sel[i][1]) for i in range(len(sel) - 1)]


def calendar_rows(months):
    """Masa, Masa-Purn, Shaka, Vikram, Samvatsara as (start, end, type, label) rows."""
    rows = []
    if SHOW_MASA:
        rows += [(m["start"], m["end"], "Masa", m["name"]) for m in months]
    if SHOW_MASA_PURN:
        seg = []                                   # merge pakshas with the same purnimanta name
        for p in build_pakshas(months):
            if p["p_name"] is None:
                continue
            name = ("Adhika " if p["p_adh"] and not p["p_name"].startswith("Adhika") else "") + p["p_name"]
            if seg and seg[-1][3] == name and abs(seg[-1][1] - p["start"]) < 1e-9:
                seg[-1] = (seg[-1][0], p["end"], "Masa-Purn", name)
            else:
                seg.append((p["start"], p["end"], "Masa-Purn", name))
        rows += seg
    if SHOW_YEAR:
        for y in build_years(months):
            if y["system"] == "shaka":
                rows.append((y["start"], y["end"], "Shaka", str(y["number"])))
                rows.append((y["start"], y["end"], "Samvatsara", lab(SAMVATSARA[y["samv"]], y["samv"])))
            else:
                rows.append((y["start"], y["end"], "Vikram", str(y["number"])))
    return rows


def make_rows(db, months, lo, hi):
    """-> [(start_jd, end_jd, type, label)]"""
    rows = []
    always = lambda x: True
    for name, (jd, k) in db.items():
        if name == "elong":
            if SHOW_KARANA:
                rows += _segs("Karana", jd, k, always, lambda x: karana_label(x % 60))
            if SHOW_TITHI:
                rows += _segs("Tithi", jd, k, lambda x: x % 2 == 0, lambda x: tithi_label((x // 2) % 30))
            if SHOW_PAKSHA:
                rows += _segs("Paksha", jd, k, lambda x: x % 30 == 0,
                              lambda x: "Shukla" if x % 60 == 0 else "Krishna")
        elif name == "moon":
            if SHOW_PADA:
                rows += _segs("Pada", jd, k, always,
                              lambda x: f"{NAKSHATRA[(x % 108) // 4]}-{x % 4 + 1}")
            if SHOW_NAKSHATRA:
                rows += _segs("Nakshatra", jd, k, lambda x: x % 4 == 0,
                              lambda x: lab(NAKSHATRA[(x % 108) // 4], (x % 108) // 4))
            if SHOW_RASHI:
                rows += _segs("Rashi", jd, k, lambda x: x % 9 == 0,
                              lambda x: lab(RASHI[(x % 108) // 9], (x % 108) // 9))
        elif name == "yoga":
            rows += _segs("Yoga", jd, k, always, lambda x: lab(YOGA[x % 27], x % 27))
        elif name == "sun_sid" and SHOW_SANKRANTI:
            rows += _segs("Sun Rashi", jd, k, always, lambda x: lab(RASHI[x % 12], x % 12))
        elif name == "sun_trop":
            if SHOW_GOLA:
                rows += _segs("Gola", jd, k, lambda x: x % 4 in (0, 2),
                              lambda x: "Uttara Gola" if x % 4 == 0 else "Dakshina Gola")
            if SHOW_AYANA:
                rows += _segs("Ayana", jd, k, lambda x: x % 4 in (1, 3),
                              lambda x: "Dakshinayana" if x % 4 == 1 else "Uttarayana")
        elif name == "ritu_say":
            rows += _segs("Ritu-Say", jd, k, always, lambda x: lab(RITU[x % 6], x % 6))
        elif name == "ritu_nir":
            rows += _segs("Ritu-Nir", jd, k, always, lambda x: lab(RITU[x % 6], x % 6))
    rows += calendar_rows(months)
    if EDGE_MODE == "starts_in":
        rows = [r for r in rows if lo <= r[0] < hi]
    else:
        rows = [r for r in rows if r[1] > lo and r[0] < hi]
    rows.sort(key=(lambda r: (r[0], r[1])) if SORT_BY == "time" else (lambda r: (r[2], r[0])))
    return rows


def show(rows):
    w = 18 + JD_DECIMALS - 8
    head = f"{'event':<11}{'name':<26}"
    head += f"{'start JD':>{w}}{'end JD':>{w + 2}}" if SHOW_JD else ""
    head += f"  {'start ' + TZ_LABEL:<19}  {'end ' + TZ_LABEL:<19}" if SHOW_IST else ""
    print(head)
    for a, b, typ, label in rows:
        line = f"{typ:<11}{label:<26}"
        line += f"{a:>{w}.{JD_DECIMALS}f}{b:>{w + 2}.{JD_DECIMALS}f}" if SHOW_JD else ""
        line += f"  {jd_to_local(a):<19}  {jd_to_local(b):<19}" if SHOW_IST else ""
        print(line)


def show_sun(suns):
    w = 18 + JD_DECIMALS - 8
    print(f"\n{'Surya spashta':<15}{'date ' + TZ_LABEL:<12}{'time':<7}{'JD (UT)':>{w}}"
          f"  {'sidereal lon':>14}  {'speed':>9}  spashta")
    for s in suns:
        print(f"{'':<15}{s['date']:<12}{s['time']:<7}{s['jd']:>{w}.{JD_DECIMALS}f}"
              f"  {s['lon']:>14.8f}  {s['speed']:>9.6f}  {s['spashta']}")


def save_csv(rows, suns, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["event", "name", "start JD", "end JD", "start " + TZ_LABEL, "end " + TZ_LABEL])
        for a, b, typ, label in rows:
            wr.writerow([typ, label, f"{a:.{JD_DECIMALS}f}", f"{b:.{JD_DECIMALS}f}",
                         jd_to_local(a), jd_to_local(b)])
        for s in suns:                              # at an instant: end columns left empty
            wr.writerow([f"Surya {s['time']}", s["spashta"], f"{s['jd']:.{JD_DECIMALS}f}", "",
                         jd_to_local(s["jd"]), ""])


# --------------------------------------------------------------- main
if __name__ == "__main__":
    def _to_jd(jd, date):
        if jd is not None:
            return float(jd)
        j = swe.julday(*date)
        return j - TZ_OFFSET_HOURS / 24.0 if DATES_ARE_LOCAL else j

    lo, hi = _to_jd(START_JD, START_DATE), _to_jd(END_JD, END_DATE)
    if hi <= lo:
        raise ValueError(f"END must be after START (got JD {lo:.5f} .. {hi:.5f})")
    init()
    t0 = time.perf_counter()
    need_cal = SHOW_MASA or SHOW_MASA_PURN or SHOW_YEAR
    pad_back = max(PAD, MASA_PAD_BACK if need_cal else 0.0)
    pad_fwd = max(PAD, MASA_PAD_FWD if need_cal else 0.0)
    eph_info = preflight(lo - pad_back, hi + pad_fwd)
    if PRINT_EPHEM:
        print(f"Ephemeris      : {eph_info}")
    db = build(lo - pad_back, hi + pad_fwd)
    months = build_months(db) if need_cal else []
    suns = sun_fixed(lo, hi) if SUN_FIXED_TIMES else []
    elapsed = time.perf_counter() - t0

    if RUN_CHECK:
        check(db)
        n = audit(db, lo, hi)
        if need_cal:
            n += audit_months(overlap(months, lo, hi))
        print(f"Verification passed: continuity check + ephemeris audit ({n:,} audit calls)")

    rows = make_rows(db, months, lo, hi)
    if need_cal:
        if any(m["shaka"] is None for m in overlap(months, lo, hi)):
            WARN.append("some months have no year: increase MASA_PAD_BACK")
    if WARN:
        msg = f"{len(WARN)} warning(s); first: {WARN[0]}"
        if STRICT:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    if PRINT_TABLE:
        show(rows)
    if PRINT_SUN and suns:
        show_sun(suns)
    if SAVE_CSV:
        save_csv(rows, suns, CSV_NAME)
    if PRINT_TIME:
        print(f"Execution time : {elapsed:.3f} s (excl. audit)")
    if PRINT_CALLS:
        used = ", ".join(f"{EPH_NAME.get(k, '?')} {v:,}" for k, v in EPH_USED.items())
        print(f"Ephemeris calls: {CALLS:,} (incl. audit)" + (f"  [{used}]" if used else ""))
    swe.close()