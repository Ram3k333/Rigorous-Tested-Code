"""
Planet events in Julian Days (Sun..Pluto, Rahu, Ketu) -- RIGOROUS edition + ABHIJIT
====================================================================================
Flag-controlled design (swisseph only samples; roots come from NumPy Chebyshev
polynomials), hardened so that NO event can be skipped.

LAYERS OF RIGOR
 1. FIT VALIDATION  - window accepted only if tail coefficients are small AND the
    polynomial matches swisseph at 7 held-out points.
 2. MONOTONIC PIECES - each window is cut at every stationary point; inside a piece
    "does it cross level L?" is decided by comparing two endpoint values.
 3. SEAM OVERLAP    - windows evaluated on [-1-EXT, 1+EXT]; duplicates merged.
 4. POLISH          - each root re-solved against the REAL ephemeris.
 5. CONTINUITY CHECK- crossings must chain; stations must alternate;
                      Abhijit enter/leave must alternate.
 6. INDEPENDENT AUDIT- true sector / Abhijit state read from swisseph every
    AUDIT_STEP days and the event list replayed between samples.

ABHIJIT
 SHOW_ABHIJIT reports when a body enters/leaves Abhijit (276°40' .. 280°53'20",
 i.e. Makara 6°40' .. 10°53'20", sidereal). It has its own level list because its
 end edge is not on the 108-pada grid. Use with SIDEREAL = True.

ABHIJIT PADA
 SHOW_ABHIJIT_PADA reports when a body enters Abhijit pada 1..4. Abhijit is split
 into 4 equal padas of 1°03'20" each (5 edges: start, 3 inner, end).
"""
import math
import time
import bisect
import csv
import numpy as np
import swisseph as swe
from numpy.polynomial import chebyshev as C
from concurrent.futures import ProcessPoolExecutor

# =====================================================================
#                              CONFIG
# =====================================================================
# ---- time range (Julian Day, UT; no datetime used) --------------------
START_DATE = (1900, 1, 1, 0.0)     # (year, month, day, hour) -> converted with swe.julday
START_JD   = None                  # set a JD here to override START_DATE
DAYS       = 219146                   # length of range
PAD        = 3.0                   # compute this many days outside the range (edge safety)
START_DATE_IS_LOCAL = True         # True -> START_DATE is IST (TZ_OFFSET_HOURS), not UT

# ---- time display -----------------------------------------------------
SHOW_JD    = False                  # JD (UT) column
SHOW_IST   = True                  # local time column: YYYY-MM-DD HH:MM:SS
TZ_OFFSET_HOURS = 5.5              # IST = UT + 5:30 (change for another zone)
TZ_LABEL   = "IST"                 # column heading

# ---- coordinate system ------------------------------------------------
SIDEREAL   = True                  # Abhijit needs sidereal. False -> tropical longitudes
AYANAMSA   = "LAHIRI"              # any swe.SIDM_<name> (used only if SIDEREAL)
EQUINOX_J2000 = False              # tropical only: fixed J2000 ecliptic instead of "of date"
NODE_TYPE  = "mean"                # "mean" (smooth) or "true" (wobbling, has stations)

# ---- ephemeris / position model ----------------------------------------
EPHEMERIS  = "swieph"              # "swieph" (SE .se1 files), "jpl" (JPL DE file) or "moseph" (built-in, ~1" less accurate)
EPHE_PATH  = r"C:\Users\durva\PycharmProjects\Rigorous_Tested_Code"   # folder holding sepl_18.se1 + sepl_24.se1 (and the JPL file if EPHEMERIS="jpl")
JPL_FILE   = "de440.eph"           # file name inside EPHE_PATH; de441 covers -13200..17191, de440 covers 1550..2650
REQUIRE_EPHEMERIS = True           # True -> stop if swisseph silently falls back to a lower-quality ephemeris
TOPOCENTRIC = False                # True -> observer-based (Moon parallax up to ~1 deg)
TOPO_LON, TOPO_LAT, TOPO_ALT = 75.8577, 22.7196, 553.0   # used only if TOPOCENTRIC
TRUE_POSITION      = False         # True -> geometric (no light-time)
NO_ABERRATION      = False         # True -> skip annual aberration
NO_GRAV_DEFLECTION = True          # True -> skip Sun light deflection (needed: makes planet paths smooth near Sun conjunction)

# ---- which bodies ------------------------------------------------------
BODY_ON = {
    "Sun": True, "Moon": False, "Mercury": True, "Venus": True, "Mars": True,
    "Jupiter": True, "Saturn": True, "Uranus": True, "Neptune": True, "Pluto": True,
    "Rahu": True, "Ketu": True,
}

# ---- which events ------------------------------------------------------
SHOW_RASHI           = True
SHOW_NAKSHATRA       = True        # (tropical mode: still computed, but not meaningful)
SHOW_PADA            = True
SHOW_ABHIJIT         = True        # Abhijit entry/exit (use with SIDEREAL = True)
SHOW_ABHIJIT_PADA    = True         # when entering Abhijit pada 1..4 (use with SIDEREAL = True)
ABHIJIT_START_DEG    = 276 + 40/60             # 6°40' Makara (start of U.Ashadha pada 4)
ABHIJIT_END_DEG      = 280 + 53/60 + 20/3600   # 10°53'20" Makara
SHOW_VAKRI           = True        # retrograde stations
SHOW_MARGI           = True        # direct stations
SHOW_RETRO_REENTRY   = True        # False -> hide backward (retrograde) boundary crossings

# ---- output ------------------------------------------------------------
PRINT_TABLE          = True
PRINT_TIME           = True
PRINT_CALLS          = True
PRINT_LEGEND         = True
SHOW_MOTION_COLUMN   = True        # the D/R/S column
SHOW_STATION_LONGITUDE = False     # "28.341 deg in Kumbha" on Vakri/Margi rows
NUMBER_LABELS        = False       # "Makara (10)", "Shravana (22)"
RASHI_LANGUAGE       = "english"   # "auto" (sanskrit if sidereal else english) / "sanskrit" / "english"
SORT_BY              = "time"      # "body" (grouped) or "time" (one chronological list)
JD_DECIMALS          = 8

# ---- rigor / verification ---------------------------------------------
RUN_CHECK       = True             # continuity of crossings + station alternation
AUDIT_STEP      = 5.0              # days between true-ephemeris audit samples (0 = off)
STRICT          = False             # True -> any warning is a hard error

# ---- performance / accuracy -------------------------------------------
WORKERS   = 4
CHUNK     = 180.0
DEG       = 12
TOL       = 1e-7                   # max tail coefficient (deg) before a window is split
VAL_TOL   = 1e-6                   # max error (deg) at held-out validation points
MIN_WIN   = 0.05
WINDOW_OVERRIDE = {}
POLISH    = True                   # re-solve every root against the true ephemeris
# ======================== END CONFIG ================================

# ---- internal constants ------------------------------------------------
EXT = 1e-7                         # window overlap in x (seam safety)
DEDUPE_DAYS = 1e-3                 # same (sector,direction) closer than this = duplicate
POLISH_RESID = 1e-6                # deg; accept Newton result below this residual
POLISH_MAX_SHIFT = 0.05            # days; larger shift from fit root -> use bracket instead
VAL_X = np.array([-0.93, -0.71, -0.37, 0.11, 0.43, 0.77, 0.96])
SCAN_X = np.linspace(-1.0, 1.0, 129)

NODE = swe.MEAN_NODE if NODE_TYPE == "mean" else swe.TRUE_NODE
ORDER = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
         "Uranus", "Neptune", "Pluto", "Rahu", "Ketu"]
ID = {"Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY, "Venus": swe.VENUS,
      "Mars": swe.MARS, "Jupiter": swe.JUPITER, "Saturn": swe.SATURN,
      "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO,
      "Rahu": NODE, "Ketu": NODE}
NO_STATIONS = {"Sun", "Moon"} | ({"Rahu", "Ketu"} if NODE_TYPE == "mean" else set())
_win = {"Sun": 10.0, "Moon": 4.0, "Mercury": 8.0, "Venus": 12.0, "Mars": 24.0,
        "Jupiter": 48.0, "Saturn": 60.0, "Uranus": 90.0, "Neptune": 120.0,
        "Pluto": 120.0, "Rahu": 30.0 if NODE_TYPE == "mean" else 5.0}
_win.update(WINDOW_OVERRIDE)

NEED_STATIONS  = SHOW_VAKRI or SHOW_MARGI
NEED_CROSSINGS = SHOW_RASHI or SHOW_NAKSHATRA or SHOW_PADA or SHOW_ABHIJIT or SHOW_ABHIJIT_PADA
AB_START, AB_END = ABHIJIT_START_DEG % 360.0, ABHIJIT_END_DEG % 360.0
AB_LEVELS = [AB_START, AB_END]     # index 0 = start edge, 1 = end edge
AP_W = (AB_END - AB_START) / 4.0   # 1°03'20" per Abhijit pada
AP_LEVELS = [AB_START + k * AP_W for k in range(5)]   # edges of pada 1|2|3|4; index 0 = start, 4 = end
if SHOW_PADA or (SHOW_RASHI and SHOW_NAKSHATRA):
    GRID = 108
elif SHOW_NAKSHATRA:
    GRID = 27
elif SHOW_RASHI:
    GRID = 12
else:
    GRID = None
R_PER = GRID // 12 if GRID and GRID % 12 == 0 else None
N_PER = GRID // 27 if GRID and GRID % 27 == 0 else None


def _needed(nm):
    if not BODY_ON[nm]:
        return False
    return NEED_CROSSINGS or (NEED_STATIONS and nm not in NO_STATIONS)


ACTIVE = [nm for nm in ORDER if _needed(nm)]
FIT_BODIES = [nm for nm in ACTIVE if nm != "Ketu"]
if "Ketu" in ACTIVE and "Rahu" not in FIT_BODIES:
    FIT_BODIES.append("Rahu")

EPH_FLAG = {"swieph": swe.FLG_SWIEPH, "jpl": swe.FLG_JPLEPH, "moseph": swe.FLG_MOSEPH}[EPHEMERIS]
EPH_NAME = {swe.FLG_SWIEPH: "Swiss Ephemeris files", swe.FLG_JPLEPH: "JPL " + JPL_FILE,
            swe.FLG_MOSEPH: "Moshier (built-in)"}
FLAGS = EPH_FLAG
if SIDEREAL:
    FLAGS |= swe.FLG_SIDEREAL
elif EQUINOX_J2000:
    FLAGS |= swe.FLG_J2000
if TOPOCENTRIC:        FLAGS |= swe.FLG_TOPOCTR
if TRUE_POSITION:      FLAGS |= swe.FLG_TRUEPOS
if NO_ABERRATION:      FLAGS |= swe.FLG_NOABERR
if NO_GRAV_DEFLECTION: FLAGS |= swe.FLG_NOGDEFL

_k = np.arange(DEG + 1)
NODES = -np.cos(np.pi * (_k + 0.5) / (DEG + 1))
FIT_M = np.linalg.inv(C.chebvander(NODES, DEG))

NAKSHATRA = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
             "Punarvasu", "Pushya", "Ashlesha", "Magha", "P.Phalguni", "U.Phalguni",
             "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula",
             "P.Ashadha", "U.Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
             "P.Bhadrapada", "U.Bhadrapada", "Revati"]

RASHI_SA = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
            "Tula", "Vrishchika", "Dhanu", "Makara", "Kumbha", "Meena"]

RASHI_EN = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

_lang = RASHI_LANGUAGE if RASHI_LANGUAGE != "auto" else ("sanskrit" if SIDEREAL else "english")
RASHI = RASHI_SA if _lang == "sanskrit" else RASHI_EN


def rashi_label(i):
    return f"{RASHI[i]} ({i + 1})" if NUMBER_LABELS else RASHI[i]


def nak_label(i):
    return f"{NAKSHATRA[i]} ({i + 1})" if NUMBER_LABELS else NAKSHATRA[i]


def jd_to_local(jd):
    """JD (UT) -> 'YYYY-MM-DD HH:MM:SS' local. Integer seconds: no 60-s carry bugs."""
    total = int(round((jd + TZ_OFFSET_HOURS / 24.0 + 0.5) * 86400.0))
    day, sod = divmod(total, 86400)
    y, m, d, _ = swe.revjul(float(day))
    return f"{y:04d}-{m:02d}-{d:02d} {sod // 3600:02d}:{sod % 3600 // 60:02d}:{sod % 60:02d}"


# ----------------------------------------------------- ephemeris wrappers
CALLS = 0
WARN = []


def _used_eph(ret):
    """Which ephemeris swisseph actually used, from its return flag."""
    for f in (swe.FLG_JPLEPH, swe.FLG_SWIEPH, swe.FLG_MOSEPH):
        if ret & f:
            return f
    return swe.FLG_MOSEPH


def _calc(jd, body, flags):
    """swe.calc_ut + guard: swisseph silently falls back (JPL -> SE -> Moshier)
    when a file is missing or out of range. ~1 arcsec difference can create or
    remove a grazing crossing near a station, so stop instead of continuing."""
    xx, ret = swe.calc_ut(jd, body, flags)
    if REQUIRE_EPHEMERIS and not (ret & EPH_FLAG):
        y, m, d, _ = swe.revjul(jd)
        raise RuntimeError(
            f"Ephemeris fallback at JD {jd:.5f} ({y:04d}-{m:02d}-{d:02d}): asked for "
            f"{EPH_NAME[EPH_FLAG]}, got {EPH_NAME[_used_eph(ret)]}. "
            f"Check EPHE_PATH={EPHE_PATH!r} contains the file covering this date.")
    return xx


def lon(jd, body):
    """Longitude only. Counted."""
    global CALLS
    CALLS += 1
    return _calc(jd, body, FLAGS)[0]


def speed(jd, body):
    """True swisseph speed (deg/day), used only for station polishing. Counted."""
    global CALLS
    CALLS += 1
    return _calc(jd, body, FLAGS | swe.FLG_SPEED)[3]


def wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


def init_worker():
    if EPHE_PATH:
        swe.set_ephe_path(EPHE_PATH)
    if EPHEMERIS == "jpl":
        swe.set_jpl_file(JPL_FILE)
    if SIDEREAL:
        swe.set_sid_mode(getattr(swe, "SIDM_" + AYANAMSA))
    if TOPOCENTRIC:
        swe.set_topo(TOPO_LON, TOPO_LAT, TOPO_ALT)


# ------------------------------------------------------------- fitting
def fit(body, a, b):
    """Adaptive Chebyshev fit. A window is accepted only if the tail is small AND the
    polynomial reproduces swisseph at held-out points (anti-aliasing validation)."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    v = np.unwrap([lon(mid + half * x, body) for x in NODES], period=360.0)
    c = FIT_M @ v
    c[0] -= 360.0 * np.floor(c[0] / 360.0)

    ok = max(abs(c[-1]), abs(c[-2])) <= TOL
    if ok:
        truth = np.array([lon(mid + half * x, body) for x in VAL_X])
        err = np.max(np.abs(wrap180(truth - C.chebval(VAL_X, c))))
        ok = err <= VAL_TOL
    if ok:
        return [(a, b, c)]
    if (b - a) > MIN_WIN:
        return fit(body, a, mid) + fit(body, mid, b)
    WARN.append(f"fit not converged for body {body} on [{a:.4f},{b:.4f}] at MIN_WIN")
    return [(a, b, c)]


# ------------------------------------------------- polynomial utilities
def _bisect_poly(p, xa, xb):
    neg_a = C.chebval(xa, p) < 0
    for _ in range(80):
        xm = 0.5 * (xa + xb)
        if (C.chebval(xm, p) < 0) == neg_a:
            xa = xm
        else:
            xb = xm
    return 0.5 * (xa + xb)


def stationary_x(d):
    """All candidate stationary points of the fitted curve (roots of derivative d),
    from TWO independent methods: eigenvalue roots AND a dense sign-change scan.
    Extra candidates are harmless (they only add piece boundaries)."""
    cand = []
    if len(d) > 1:
        r = C.chebroots(d)
        r = r[np.abs(r.imag) < 1e-6].real
        cand = [float(x) for x in r if -1.0 - EXT <= x <= 1.0 + EXT]
    g = C.chebval(SCAN_X, d)
    neg = g < 0
    for i in np.nonzero(neg[:-1] != neg[1:])[0]:
        xa, xb = SCAN_X[i], SCAN_X[i + 1]
        if not any(xa - 1e-9 <= x <= xb + 1e-9 for x in cand):
            cand.append(_bisect_poly(d, xa, xb))
    out = []
    for x in sorted(cand):
        if not out or x - out[-1] > 1e-9:
            out.append(x)
    return out


def _solve_piece(c, d, xa, xb, L, ga, gb):
    """Root of c(x)=L inside a MONOTONIC piece [xa,xb] whose end values already
    straddle L (ga = c(xa)-L, gb = c(xb)-L). Safeguarded Newton + bisection."""
    lo, hi, glo = xa, xb, ga
    up_lo = glo >= 0
    x = lo - glo * (hi - lo) / (gb - glo)
    for _ in range(80):
        g = float(C.chebval(x, c)) - L
        if g == 0.0:
            return x
        if (g >= 0) == up_lo:
            lo, glo = x, g
        else:
            hi = x
        dg = float(C.chebval(x, d))
        xn = x - g / dg if dg != 0.0 else float("nan")
        if not (lo < xn < hi):
            xn = 0.5 * (lo + hi)
        if abs(xn - x) < 1e-14:
            return xn
        x = xn
    return x


# ----------------------------------------------------- root polishing
def _polish_cross(lonfn, L, jd_fit, slope):
    """Re-solve lon(jd) == L against the REAL ephemeris. Newton with the (accurate)
    polynomial slope; falls back to a true bracket if Newton is unhappy."""
    jd = jd_fit
    f = wrap180(lonfn(jd) - L)
    for _ in range(3):
        if abs(f) < 1e-9:
            break
        if abs(slope) < 1e-9:
            break
        jd -= f / slope
        f = wrap180(lonfn(jd) - L)
    if abs(f) < POLISH_RESID and abs(jd - jd_fit) < POLISH_MAX_SHIFT:
        return jd
    # bracketed fallback (rare: near a station or a bad slope)
    h = 1e-3
    for _ in range(8):
        fl = wrap180(lonfn(jd_fit - h) - L)
        fr = wrap180(lonfn(jd_fit + h) - L)
        if (fl >= 0) != (fr >= 0):
            lo, hi = jd_fit - h, jd_fit + h
            for _ in range(45):
                if hi - lo < 1e-9:
                    break
                mid = 0.5 * (lo + hi)
                fm = wrap180(lonfn(mid) - L)
                if (fm >= 0) == (fl >= 0):
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        h *= 4.0
    WARN.append(f"polish bracket failed near JD {jd_fit:.5f}; kept fit root")
    return jd_fit


def _polish_station(body, jd_fit, vakri):
    """Bisection on the true swisseph speed. Vakri: speed + -> -. Margi: - -> +."""
    for h in (0.05, 0.2, 0.8):
        sl, sr = speed(jd_fit - h, body), speed(jd_fit + h, body)
        if vakri and sl > 0 > sr or (not vakri) and sl < 0 < sr:
            lo, hi = jd_fit - h, jd_fit + h
            while hi - lo > 1e-7:
                mid = 0.5 * (lo + hi)
                if (speed(mid, body) > 0) == (sl > 0):
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
    WARN.append(f"station polish bracket failed near JD {jd_fit:.5f}; kept fit root")
    return jd_fit


# ------------------------------------------------------------- solvers
def _levels_in(lo, hi, den, levels):
    """Yield (index, absolute level) for every level inside [lo, hi].
    Grid mode: index n means level n*360/den.  Custom mode: index into `levels`,
    repeated every 360 deg because the fitted polynomial is unwrapped."""
    if levels is None:
        lev = 360.0 / den
        for n in range(int(math.ceil(lo / lev)), int(math.floor(hi / lev)) + 1):
            yield n, n * 360.0 / den                         # multiply first: exact labels
    else:
        for i, base in enumerate(levels):
            for k in range(int(math.ceil((lo - base) / 360.0)),
                           int(math.floor((hi - base) / 360.0)) + 1):
                yield i, base + 360.0 * k


def crossings(c, a, b, den, lonfn, levels=None):
    """EVERY crossing of the grid levels (or of `levels`) in the window
    -> [(n_or_index, jd, +1/-1)]. Cut at all stationary points into monotonic
    pieces; a piece crosses level L iff its two end values straddle L."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    d = C.chebder(c)
    xs = [x for x in stationary_x(d) if -1.0 - EXT < x < 1.0 + EXT]
    xk = np.array([-1.0 - EXT] + xs + [1.0 + EXT])
    vk = C.chebval(xk, c)
    out = []
    for i in range(len(xk) - 1):
        v0, v1 = float(vk[i]), float(vk[i + 1])
        for n, L in _levels_in(min(v0, v1), max(v0, v1), den, levels):
            if (v0 >= L) == (v1 >= L):
                continue
            x = _solve_piece(c, d, xk[i], xk[i + 1], L, v0 - L, v1 - L)
            jd = mid + half * x
            direction = 1 if v1 >= L else -1
            if POLISH:
                jd = _polish_cross(lonfn, L, jd, float(C.chebval(x, d)) / half)
            out.append((n, jd, direction))
    return out


def stations(c, a, b, body):
    """Verified sign changes of velocity -> [(jd, 'Vakri'|'Margi', longitude)]."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    d = C.chebder(c)
    out = []
    for x in stationary_x(d):
        dl = float(C.chebval(x - 1e-6, d))
        dr = float(C.chebval(x + 1e-6, d))
        if (dl > 0) == (dr > 0):
            continue                                         # touched zero, no sign change
        vakri = dl > 0
        jd = mid + half * x
        if POLISH:
            jd = _polish_station(body, jd, vakri)
        L = lon(jd, body) if POLISH else float(C.chebval(x, c)) % 360.0
        out.append((jd, "Vakri" if vakri else "Margi", L % 360.0))
    return out


# --------------------------------------------------------------- jobs
def job(args):
    """One fitted body over one time chunk (runs in a worker process)."""
    global CALLS, WARN
    CALLS = 0
    WARN = []
    init_worker()
    nm, c0, c1 = args
    body = ID[nm]
    L = _win[nm]
    ev = {}
    add = lambda k, rows: ev.setdefault(k, []).extend(rows)
    for i in range(int(np.ceil((c1 - c0) / L))):
        a, b = c0 + i * L, min(c0 + (i + 1) * L, c1)
        for wa, wb, c in fit(body, a, b):
            want = ["Rahu", "Ketu"] if nm == "Rahu" else [nm]
            st = stations(c, wa, wb, body) if (NEED_STATIONS and nm not in NO_STATIONS) else []
            for who in want:
                if who not in ACTIVE:
                    continue
                shift = 180.0 if who == "Ketu" else 0.0
                cw = c
                if who == "Ketu":                              # Ketu = Rahu + 180: no extra fit
                    cw = c.copy(); cw[0] += shift
                if NEED_CROSSINGS:
                    lf = (lambda jd, _b=body, _s=shift: lon(jd, _b) + _s)
                    if GRID:
                        add(who, crossings(cw, wa, wb, GRID, lf))
                    if SHOW_ABHIJIT:
                        add(who + ":ab", crossings(cw, wa, wb, None, lf, AB_LEVELS))
                    if SHOW_ABHIJIT_PADA:
                        add(who + ":ap", crossings(cw, wa, wb, None, lf, AP_LEVELS))
                add(who + ":st", [(j, k, (l + shift) % 360.0) for j, k, l in st])
    return ev, CALLS, WARN


def _dedupe(rows, key, tpos):
    """Merge seam duplicates: identical key closer than DEDUPE_DAYS."""
    rows = sorted(rows, key=lambda r: r[tpos])
    last, out = {}, []
    for r in rows:
        k = key(r)
        if k in last and r[tpos] - last[k] < DEDUPE_DAYS:
            continue
        last[k] = r[tpos]
        out.append(r)
    return out


def build(start, end):
    edges = list(np.arange(start, end, CHUNK)) + [end]
    tasks = [(nm, c0, c1) for c0, c1 in zip(edges[:-1], edges[1:]) for nm in FIT_BODIES]
    if WORKERS > 1:
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(job, tasks))
    else:
        res = [job(t) for t in tasks]
    db, calls, warns = {}, 0, []
    for ev, n, w in res:
        calls += n
        warns += w
        for k, rows in ev.items():
            db.setdefault(k, []).extend(rows)
    for k in db:
        if k.endswith(":st"):
            db[k] = _dedupe(db[k], key=lambda r: r[1], tpos=0)
        else:                                                # grid crossings, :ab and :ap share layout
            db[k] = _dedupe(db[k], key=lambda r: (r[0], r[2]), tpos=1)
    return db, calls, warns


# ------------------------------------------------- verification layers
def check(db):
    """Layer 5: crossings chain continuously (incl. retrograde back-steps),
    Vakri/Margi alternate, Abhijit enter/leave alternate."""
    for k in ACTIVE:
        if GRID:
            ev = db.get(k, [])
            for (n0, t0, d0), (n1, t1, d1) in zip(ev[:-1], ev[1:]):
                step = (n1 - n0) % GRID
                ok = (step == (1 if d0 > 0 else GRID - 1) and d1 == d0) or (step == 0 and d1 == -d0)
                assert ok, f"discontinuity in {k} between JD {t0:.5f} and {t1:.5f}: crossing skipped/duplicated"
        if NEED_STATIONS and k not in NO_STATIONS:
            st = db.get(k + ":st", [])
            for (t0, k0, _), (t1, k1, _) in zip(st[:-1], st[1:]):
                assert k0 != k1, f"{k}: two consecutive {k0} stations (JD {t0:.5f}, {t1:.5f}): station skipped"
        if SHOW_ABHIJIT:                                     # enter/leave must alternate
            inside = None
            for i, tj, d in db.get(k + ":ab", []):
                enter = (i == 0) == (d > 0)                  # start edge up / end edge down
                if inside is not None:
                    assert inside != enter, (f"{k}: Abhijit {'enter' if enter else 'leave'} twice "
                                             f"in a row at JD {tj:.5f}: event skipped")
                inside = enter


def audit(db, lo, hi):
    """Layer 6: independent proof against the real ephemeris. Read the true sector
    every AUDIT_STEP days, and replay the event list in between; every event must
    start from the current sector and the replay must land on the true sector."""
    if not GRID or AUDIT_STEP <= 0:
        return 0
    global CALLS
    before = CALLS
    for k in ACTIVE:
        ev = db.get(k, [])
        jds = [e[1] for e in ev]
        body, off = ID[k], (180.0 if k == "Ketu" else 0.0)

        def cell(t):
            return int(math.floor(((lon(t, body) + off) % 360.0) * GRID / 360.0)) % GRID

        prev_t = prev_c = None
        for t in np.arange(lo, hi, AUDIT_STEP):
            t = float(t)
            for _ in range(3):                                # step off exact event times
                i = bisect.bisect_left(jds, t - 1e-5)
                if i < len(jds) and jds[i] < t + 1e-5:
                    t += 3e-5
            c_true = cell(t)
            if prev_t is not None:
                cur = prev_c
                for n, tj, d in ev[bisect.bisect_right(jds, prev_t):bisect.bisect_right(jds, t)]:
                    frm = (n - 1) % GRID if d > 0 else n % GRID
                    assert cur == frm, f"audit {k}: event at JD {tj:.5f} does not follow sector {cur}"
                    cur = n % GRID if d > 0 else (n - 1) % GRID
                assert cur == c_true, (f"audit {k}: replay says sector {cur} but ephemeris says "
                                       f"{c_true} at JD {t:.5f} (window {prev_t:.5f}..{t:.5f}): event missed")
            prev_t, prev_c = t, c_true
    return CALLS - before


def audit_abhijit(db, lo, hi):
    """Independent check: true inside/outside state from swisseph vs replayed events."""
    if not SHOW_ABHIJIT or AUDIT_STEP <= 0:
        return 0
    global CALLS
    before = CALLS
    for k in ACTIVE:
        ev = db.get(k + ":ab", [])
        jds = [e[1] for e in ev]
        body, off = ID[k], (180.0 if k == "Ketu" else 0.0)

        def inside(t):
            return AB_START <= (lon(t, body) + off) % 360.0 < AB_END

        prev_t = prev_in = None
        for t in np.arange(lo, hi, AUDIT_STEP):
            t = float(t)
            for _ in range(3):                               # step off exact event times
                i = bisect.bisect_left(jds, t - 1e-5)
                if i < len(jds) and jds[i] < t + 1e-5:
                    t += 3e-5
            now = inside(t)
            if prev_t is not None:
                cur = prev_in
                for i, tj, d in ev[bisect.bisect_right(jds, prev_t):bisect.bisect_right(jds, t)]:
                    enter = (i == 0) == (d > 0)
                    assert cur != enter, f"audit {k}: Abhijit event at JD {tj:.5f} inconsistent with state"
                    cur = enter
                assert cur == now, (f"audit {k}: Abhijit replay says {'inside' if cur else 'outside'} "
                                    f"but ephemeris disagrees at JD {t:.5f}: event missed")
            prev_t, prev_in = t, now
    return CALLS - before


# ---------------------------------------------------------- tables
def entering(n, per, d, count):
    """Sector being ENTERED. A backward crossing enters the previous sector."""
    return (n // per - (1 if d < 0 else 0)) % count


def body_rows(db, name, lo, hi):
    rows = []
    for n, jd, d in db.get(name, []):
        if not lo <= jd < hi or (d < 0 and not SHOW_RETRO_REENTRY):
            continue
        mv = "RETROGRADE" if d < 0 else "DIRECT"
        if SHOW_RASHI and R_PER and n % R_PER == 0:
            rows.append((jd, name, "Rashi", rashi_label(entering(n, R_PER, d, 12)), mv))
        if SHOW_NAKSHATRA and N_PER and n % N_PER == 0:
            rows.append((jd, name, "Nakshatra", nak_label(entering(n, N_PER, d, 27)), mv))
        if SHOW_PADA and GRID == 108:
            pada_idx = entering(n, 1, d, 108)                 # sector entered, 0..107
            rows.append((jd, name, "Pada", f"{NAKSHATRA[pada_idx // 4]}-{pada_idx % 4 + 1}", mv))
    for i, jd, d in db.get(name + ":ab", []):
        if not lo <= jd < hi or (d < 0 and not SHOW_RETRO_REENTRY):
            continue
        mv = "RETROGRADE" if d < 0 else "DIRECT"
        enter = (i == 0) == (d > 0)
        rows.append((jd, name, "Abhijit", "Enters Abhijit" if enter else "Leaves Abhijit", mv))
    for i, jd, d in db.get(name + ":ap", []):
        if not lo <= jd < hi or (d < 0 and not SHOW_RETRO_REENTRY):
            continue
        if d > 0 and i < 4:            # moving forward over edge i -> enters pada i+1
            p = i + 1
        elif d < 0 and i > 0:          # moving backward over edge i -> enters pada i
            p = i
        else:                          # edge 0 backward / edge 4 forward = leaving Abhijit
            continue
        mv = "RETROGRADE" if d < 0 else "DIRECT"
        rows.append((jd, name, "AbhPada", f"Abhijit-{p}", mv))
    for jd, kind, l in db.get(name + ":st", []):
        if not lo <= jd < hi or (kind == "Vakri" and not SHOW_VAKRI) \
                or (kind == "Margi" and not SHOW_MARGI):
            continue
        detail = f"{l % 30:6.3f} deg in {RASHI[int(l // 30)]}" if SHOW_STATION_LONGITUDE else ""
        rows.append((jd, name, kind, detail, "S"))
    return rows


def _all_rows(db, lo, hi):
    rows = []
    for name in ACTIVE:
        rows += sorted(body_rows(db, name, lo, hi))
    if SORT_BY == "time":
        rows.sort()
    return rows


def show(db, lo, hi):
    rows = _all_rows(db, lo, hi)
    if PRINT_LEGEND and SHOW_MOTION_COLUMN:
        print("motion: D direct, R retrograde crossing, S station")
    w = 18 + JD_DECIMALS - 8
    head = f"{'body':<9}{'event':<11}{'entering / detail':<28}"
    head += f"{'JD':>{w}}" if SHOW_JD else ""
    head += f"  {TZ_LABEL:<19}" if SHOW_IST else ""
    print(head + ("  mv" if SHOW_MOTION_COLUMN else ""))
    for jd, name, ev, label, mv in rows:
        line = f"{name:<9}{ev:<11}{label:<28}"
        line += f"{jd:>{w}.{JD_DECIMALS}f}" if SHOW_JD else ""
        line += f"  {jd_to_local(jd):<19}" if SHOW_IST else ""
        print(line + (f"  {mv}" if SHOW_MOTION_COLUMN else ""))


def save_csv(db, lo, hi, filename="events.csv"):
    rows = _all_rows(db, lo, hi)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["body", "event", "entering / detail", "JD", TZ_LABEL, "motion"])
        for jd, name, ev, label, mv in rows:
            wr.writerow([name, ev, label, f"{jd:.{JD_DECIMALS}f}", jd_to_local(jd), mv])


# --------------------------------------------------------------- main
if __name__ == "__main__":
    s_jd = START_JD if START_JD is not None else swe.julday(*START_DATE)
    if START_JD is None and START_DATE_IS_LOCAL:
        s_jd -= TZ_OFFSET_HOURS / 24.0
    lo, hi = s_jd, s_jd + DAYS
    init_worker()

    # ---- ephemeris pre-flight: every active body, every ~10 years, both ends
    probe = list(np.arange(lo - PAD, hi + PAD, 3652.5)) + [hi + PAD]
    for nm in FIT_BODIES:
        for t in probe:
            lon(float(t), ID[nm])                  # raises if the file is missing/out of range
    print(f"Ephemeris: {EPH_NAME[EPH_FLAG]}"
          + (f"  (path: {EPHE_PATH})" if EPHE_PATH else "")
          + ("  [fallback guard ON]" if REQUIRE_EPHEMERIS else "  [fallback guard OFF]"))

    t0 = time.perf_counter()
    db, calls, warns = build(lo - PAD, hi + PAD)
    elapsed = time.perf_counter() - t0

    if warns:
        msg = f"{len(warns)} warning(s); first: {warns[0]}"
        if STRICT:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    audit_calls = 0
    if RUN_CHECK and (GRID or SHOW_ABHIJIT):
        check(db)
        audit_calls = audit(db, lo, hi) + audit_abhijit(db, lo, hi)
        print(f"Verification passed: continuity check + ephemeris audit "
              f"({audit_calls:,} audit calls)")
    elif RUN_CHECK and NEED_STATIONS:
        check(db)

    if PRINT_TABLE:
        show(db, lo, hi)
        save_csv(db, lo, hi, "planet_test.csv")
    if PRINT_TIME:
        print(f"Execution time : {elapsed:.3f} s (excl. audit)")
    if PRINT_CALLS:
        print(f"Ephemeris calls: {calls:,}")