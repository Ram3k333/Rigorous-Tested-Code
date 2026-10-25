"""
Planet ASPECTS with orb entry / exact / orb exit times  (Major, Minor, Custom)
==============================================================================
Same engine as the planet-events script: swisseph only samples, roots come from
Chebyshev polynomials, every root is polished against the real ephemeris.

HOW IT WORKS
 * For every planet pair (A,B) the RELATIVE longitude d(t) = lon(A) - lon(B) is fitted.
 * An aspect of angle T with orb O is active while d is within O of T or of 360-T
   (the two mirror positions; for 0 and 180 they are the same point).
 * Levels per aspect side:  T-O (orb edge),  T (exact),  T+O (orb edge).
   Cut at every stationary point -> monotonic pieces -> no crossing can be skipped.
 * A state machine turns the crossings into episodes:
       orb start  ->  exact (one or MORE times if a retrograde turn re-crosses)  ->  orb exit
 * Audit: true inside/outside orb state is read from swisseph on a time grid and compared
   with the replayed event list.

OUTPUT
   Aspect | Planet (sign) Aspect angle Planet (sign) | Orb start | Exact | Orb exit
"""
import math
import time
import bisect
import csv
import itertools
import numpy as np
import swisseph as swe
from numpy.polynomial import chebyshev as C
from concurrent.futures import ProcessPoolExecutor

# =====================================================================
#                              CONFIG
# =====================================================================
# ---- time range -------------------------------------------------------
START_DATE = (2026, 1, 1, 0.0)    # (year, month, day, hour)
START_JD   = None                  # set a JD to override START_DATE
DAYS       = 365
PAD        = 3.0                   # days computed outside the range (edge safety)
START_DATE_IS_LOCAL = True         # True -> START_DATE is local time (TZ_OFFSET_HOURS)

# ---- time display -----------------------------------------------------
TZ_OFFSET_HOURS = 5.5
TZ_LABEL   = "IST"
SHOW_JD    = False

# ---- coordinate system ------------------------------------------------
SIDEREAL   = True
AYANAMSA   = "LAHIRI"
EQUINOX_J2000 = False
NODE_TYPE  = "mean"                # "mean" or "true"

# ---- ephemeris --------------------------------------------------------
EPHEMERIS  = "jpl"                 # "swieph" / "jpl" / "moseph"
EPHE_PATH  = r"C:\Users\durva\PycharmProjects\Rigorous_Tested_Code"
JPL_FILE   = "de440.eph"
REQUIRE_EPHEMERIS = True
TOPOCENTRIC = False
TOPO_LON, TOPO_LAT, TOPO_ALT = 75.8577, 22.7196, 553.0
TRUE_POSITION      = False
NO_ABERRATION      = False
NO_GRAV_DEFLECTION = True

# ---- WHICH PLANETS (include / exclude) -------------------------------
BODY_ON = {
    "Sun": True, "Moon": False, "Mercury": True, "Venus": True, "Mars": True,
    "Jupiter": True, "Saturn": True, "Uranus": False, "Neptune": False, "Pluto": False,
    "Rahu": False, "Ketu": False,
}
# Optional: only keep pairs that contain at least one of these (empty = no filter)
#   e.g. FOCUS = ["Saturn"]  -> only aspects that involve Saturn
FOCUS = []
# Optional: pairs to drop, e.g. [("Sun", "Mercury"), ("Rahu", "Ketu")]
EXCLUDE_PAIRS = [("Rahu", "Ketu")]          # Rahu-Ketu is always 180 deg, so always excluded

# ---- WHICH ASPECTS -----------------------------------------------------
# (name, angle, orb, influence, enabled)
ASPECTS = [
    # ---- major
    ("Conjunction",   0,         7,   1.0,  True),
    ("Opposition",    180,       7,   0.8,  True),
    ("Square",        90,        7,   0.8,  True),
    ("Trine",         120,       7,   0.6,  True),
    ("Sextile",       60,        6,   0.6,  True),
    # ---- minor
    ("Inconjunct",    150,       3,   0.4,  False),
    ("SemiSextile",   30,        3,   0.4,  False),
    ("SemiSquare",    45,        3,   0.2,  False),
    ("SesquiQuadrate", 135,      3,   0.2,  False),
    ("Quintile",      72,        2,   0.2,  False),
    ("BiQuintile",    144,       2,   0.2,  False),
    ("SemiQuintile",  36,        1,   0.1,  False),
    ("Septile",       360 / 7,   1,   0.1,  False),
    ("Novile",        40,        1,   0.1,  False),
    ("BiNovile",      80,        1,   0.1,  False),
    ("BiSeptile",     720 / 7,   1,   0.1,  False),
    ("TriSeptile",    1080 / 7,  1,   0.1,  False),
    ("QuatroNovile",  160,       1,   0.1,  False),
    ("TriDecile",     108,       0.5, 0.05, False),
    ("Undecile",      360 / 11,  0.5, 0.05, False),
    ("BiUndecile",    720 / 11,  0.5, 0.05, False),
    ("TriUndecile",   1080 / 11, 0.5, 0.05, False),
    ("QuadUndecile",  1440 / 11, 0.5, 0.05, False),
    ("QuinUndecile",  1800 / 11, 0.5, 0.05, False),
]
# ---- YOUR OWN aspects: (name, angle, orb)   orb = None or 0 -> exact time only
CUSTOM_ASPECTS = [
    # ("MyAspect", 17.5, 1.0),
    # ("Exact 100", 100, None),
]
USE_ORB        = True      # False -> every aspect reports the exact time only
ORB_MULTIPLIER = 1.0       # scale all orbs (e.g. 0.5 = half the orbs)
MIN_INFLUENCE  = 0.0       # drop aspects whose influence is below this (custom aspects = 1.0)

# ---- output ------------------------------------------------------------
PRINT_TABLE  = True
PRINT_TIME   = True
PRINT_CALLS  = True
SHOW_SIGNS   = True        # "Sun (Mesha) Trine 120 Moon (Simha)" - signs taken at the exact time
RASHI_LANGUAGE = "english" # "auto" / "sanskrit" / "english"
SORT_BY      = "time"      # "time" or "aspect"
JD_DECIMALS  = 8
CSV_FILE     = "aspects.csv"

# ---- rigor --------------------------------------------------------------
RUN_CHECK  = True
AUDIT_STEP = 5.0           # max days between audit samples (automatically denser for fast pairs)
STRICT     = False

# ---- performance / accuracy ---------------------------------------------
WORKERS   = 4
CHUNK     = 180.0
DEG       = 12
TOL       = 1e-7
VAL_TOL   = 1e-6
MIN_WIN   = 0.05
WINDOW_OVERRIDE = {}
POLISH    = True
# ======================== END CONFIG ================================

EXT = 1e-7
DEDUPE_DAYS = 1e-3
POLISH_RESID = 1e-6
POLISH_MAX_SHIFT = 0.05
VAL_X = np.array([-0.93, -0.71, -0.37, 0.11, 0.43, 0.77, 0.96])
SCAN_X = np.linspace(-1.0, 1.0, 129)

NODE = swe.MEAN_NODE if NODE_TYPE == "mean" else swe.TRUE_NODE
ORDER = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
         "Uranus", "Neptune", "Pluto", "Rahu", "Ketu"]
ID = {"Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY, "Venus": swe.VENUS,
      "Mars": swe.MARS, "Jupiter": swe.JUPITER, "Saturn": swe.SATURN,
      "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO,
      "Rahu": NODE, "Ketu": NODE}
SHIFT = {nm: 0.0 for nm in ORDER}
SHIFT["Ketu"] = 180.0
_win = {"Sun": 10.0, "Moon": 4.0, "Mercury": 8.0, "Venus": 12.0, "Mars": 24.0,
        "Jupiter": 48.0, "Saturn": 60.0, "Uranus": 90.0, "Neptune": 120.0,
        "Pluto": 120.0, "Rahu": 30.0 if NODE_TYPE == "mean" else 5.0}
_win["Ketu"] = _win["Rahu"]
_win.update(WINDOW_OVERRIDE)
if "Rahu" in WINDOW_OVERRIDE:
    _win["Ketu"] = _win["Rahu"]

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

RASHI_SA = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
            "Tula", "Vrishchika", "Dhanu", "Makara", "Kumbha", "Meena"]
RASHI_EN = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
_lang = RASHI_LANGUAGE if RASHI_LANGUAGE != "auto" else ("sanskrit" if SIDEREAL else "english")
RASHI = RASHI_SA if _lang == "sanskrit" else RASHI_EN

# ---------------------------------------------------------- aspect table
ASP = []        # (name, angle, orb)
for nm, ang, orb, infl, on in ASPECTS:
    if on and infl >= MIN_INFLUENCE:
        ASP.append((nm, float(ang) % 360.0, float(orb or 0.0)))
for nm, ang, orb in CUSTOM_ASPECTS:
    ASP.append((nm, float(ang) % 360.0, float(orb or 0.0)))
ASP = [(n, a, (o * ORB_MULTIPLIER if USE_ORB else 0.0)) for n, a, o in ASP]

# TARGETS[ai] = list of centre angles (angle and its mirror 360-angle; one if 0/180)
TARGETS = []
for _n, _a, _o in ASP:
    t = [_a]
    if (_a % 180.0) != 0.0:
        t.append((360.0 - _a) % 360.0)
    TARGETS.append(t)

# LEVELS: one flat list of crossing levels; META[i] = (aspect idx, target idx, kind)
# kind 0 = lower orb edge, 1 = exact, 2 = upper orb edge
LEVELS, META = [], []
for ai, (_n, _a, _o) in enumerate(ASP):
    for ti, L in enumerate(TARGETS[ai]):
        if _o > 0:
            LEVELS.append((L - _o) % 360.0); META.append((ai, ti, 0))
        LEVELS.append(L % 360.0);            META.append((ai, ti, 1))
        if _o > 0:
            LEVELS.append((L + _o) % 360.0); META.append((ai, ti, 2))
KEYS = sorted({(m[0], m[1]) for m in META})


def _pairs():
    act = [nm for nm in ORDER if BODY_ON.get(nm)]
    ex = {frozenset(p) for p in EXCLUDE_PAIRS} | {frozenset(("Rahu", "Ketu"))}
    out = []
    for a, b in itertools.combinations(act, 2):
        if frozenset((a, b)) in ex:
            continue
        if FOCUS and a not in FOCUS and b not in FOCUS:
            continue
        out.append((a, b))
    return out


PAIRS = _pairs()
ACTIVE = sorted({n for p in PAIRS for n in p}, key=ORDER.index)


def jd_to_local(jd):
    total = int(round((jd + TZ_OFFSET_HOURS / 24.0 + 0.5) * 86400.0))
    day, sod = divmod(total, 86400)
    y, m, d, _ = swe.revjul(float(day))
    return f"{y:04d}-{m:02d}-{d:02d} {sod // 3600:02d}:{sod % 3600 // 60:02d}:{sod % 60:02d}"


# ----------------------------------------------------- ephemeris wrappers
CALLS = 0
WARN = []


def _used_eph(ret):
    for f in (swe.FLG_JPLEPH, swe.FLG_SWIEPH, swe.FLG_MOSEPH):
        if ret & f:
            return f
    return swe.FLG_MOSEPH


def _calc(jd, body, flags):
    xx, ret = swe.calc_ut(jd, body, flags)
    if REQUIRE_EPHEMERIS and not (ret & EPH_FLAG):
        y, m, d, _ = swe.revjul(jd)
        raise RuntimeError(
            f"Ephemeris fallback at JD {jd:.5f} ({y:04d}-{m:02d}-{d:02d}): asked for "
            f"{EPH_NAME[EPH_FLAG]}, got {EPH_NAME[_used_eph(ret)]}. "
            f"Check EPHE_PATH={EPHE_PATH!r}.")
    return xx


def lon(jd, nm):
    """Longitude of a named body (Ketu = Rahu + 180). Counted."""
    global CALLS
    CALLS += 1
    return (_calc(jd, ID[nm], FLAGS)[0] + SHIFT[nm]) % 360.0


def rel(jd, a, b):
    return lon(jd, a) - lon(jd, b)


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
def fit(a_nm, b_nm, a, b):
    """Adaptive Chebyshev fit of the relative longitude, validated at held-out points."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    v = np.unwrap([rel(mid + half * x, a_nm, b_nm) for x in NODES], period=360.0)
    c = FIT_M @ v
    c[0] -= 360.0 * np.floor(c[0] / 360.0)
    ok = max(abs(c[-1]), abs(c[-2])) <= TOL
    if ok:
        truth = np.array([rel(mid + half * x, a_nm, b_nm) for x in VAL_X])
        err = np.max(np.abs(wrap180(truth - C.chebval(VAL_X, c))))
        ok = err <= VAL_TOL
    if ok:
        return [(a, b, c)]
    if (b - a) > MIN_WIN:
        return fit(a_nm, b_nm, a, mid) + fit(a_nm, b_nm, mid, b)
    WARN.append(f"fit not converged for {a_nm}-{b_nm} on [{a:.4f},{b:.4f}] at MIN_WIN")
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


def _polish_cross(lonfn, L, jd_fit, slope):
    jd = jd_fit
    f = wrap180(lonfn(jd) - L)
    for _ in range(3):
        if abs(f) < 1e-9 or abs(slope) < 1e-9:
            break
        jd -= f / slope
        f = wrap180(lonfn(jd) - L)
    if abs(f) < POLISH_RESID and abs(jd - jd_fit) < POLISH_MAX_SHIFT:
        return jd
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


def crossings(c, a, b, lonfn):
    """EVERY crossing of every level in LEVELS in one window -> [(level idx, jd, +1/-1)]."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    d = C.chebder(c)
    xs = [x for x in stationary_x(d) if -1.0 - EXT < x < 1.0 + EXT]
    xk = np.array([-1.0 - EXT] + xs + [1.0 + EXT])
    vk = C.chebval(xk, c)
    out = []
    for i in range(len(xk) - 1):
        v0, v1 = float(vk[i]), float(vk[i + 1])
        lo, hi = min(v0, v1), max(v0, v1)
        for idx, base in enumerate(LEVELS):
            for k in range(int(math.ceil((lo - base) / 360.0)),
                           int(math.floor((hi - base) / 360.0)) + 1):
                L = base + 360.0 * k
                if (v0 >= L) == (v1 >= L):
                    continue
                x = _solve_piece(c, d, xk[i], xk[i + 1], L, v0 - L, v1 - L)
                jd = mid + half * x
                direction = 1 if v1 >= L else -1
                if POLISH:
                    jd = _polish_cross(lonfn, base, jd, float(C.chebval(x, d)) / half)
                out.append((idx, jd, direction))
    return out


# --------------------------------------------------------------- jobs
def job(args):
    global CALLS, WARN
    CALLS = 0
    WARN = []
    init_worker()
    a_nm, b_nm, c0, c1 = args
    L = min(_win[a_nm], _win[b_nm])
    rows = []
    lf = lambda jd, _a=a_nm, _b=b_nm: rel(jd, _a, _b)
    for i in range(int(np.ceil((c1 - c0) / L))):
        a, b = c0 + i * L, min(c0 + (i + 1) * L, c1)
        for wa, wb, c in fit(a_nm, b_nm, a, b):
            rows += crossings(c, wa, wb, lf)
    return (a_nm, b_nm), rows, CALLS, WARN


def _dedupe(rows):
    rows = sorted(rows, key=lambda r: r[1])
    last, out = {}, []
    for r in rows:
        k = (r[0], r[2])
        if k in last and r[1] - last[k] < DEDUPE_DAYS:
            continue
        last[k] = r[1]
        out.append(r)
    return out


def build(start, end):
    edges = list(np.arange(start, end, CHUNK)) + [end]
    tasks = [(a, b, c0, c1) for c0, c1 in zip(edges[:-1], edges[1:]) for a, b in PAIRS]
    if WORKERS > 1:
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(job, tasks))
    else:
        res = [job(t) for t in tasks]
    db, calls, warns = {}, 0, []
    for pair, rows, n, w in res:
        calls += n
        warns += w
        db.setdefault(pair, []).extend(rows)
    for k in db:
        db[k] = _dedupe(db[k])
    return db, calls, warns


# ------------------------------------------------------- episodes
def is_entry(kind, d):
    """Entering orb: lower edge going up, or upper edge going down."""
    return (kind == 0 and d > 0) or (kind == 2 and d < 0)


def initial_states(t0):
    """True inside/outside-orb state of every (pair, aspect, target) at t0."""
    st = {}
    for a, b in PAIRS:
        r = rel(t0, a, b)
        for ai, ti in KEYS:
            o = ASP[ai][2]
            if o > 0:
                st[(a, b, ai, ti)] = abs(wrap180(r - TARGETS[ai][ti])) <= o
    return st


def episodes(db, init, problems):
    """-> list of dict(pair, ai, start, exacts, end). start/end None = outside computed range."""
    eps = []
    for (a, b), rows in db.items():
        grp = {}
        for idx, jd, d in rows:
            ai, ti, kind = META[idx]
            grp.setdefault((ai, ti), []).append((jd, kind, d))
        for ai, ti in KEYS:
            ev = sorted(grp.get((ai, ti), []))
            o = ASP[ai][2]
            if o <= 0:                                        # exact only
                for jd, kind, d in ev:
                    eps.append(dict(pair=(a, b), ai=ai, start=None, exacts=[jd], end=None, orb0=True))
                continue
            cur = dict(pair=(a, b), ai=ai, start=None, exacts=[], end=None, orb0=False) \
                if init[(a, b, ai, ti)] else None
            for jd, kind, d in ev:
                if kind == 1:
                    if cur is None:
                        problems.append(f"{a}-{b} {ASP[ai][0]}: exact outside orb at JD {jd:.5f}")
                        cur = dict(pair=(a, b), ai=ai, start=None, exacts=[], end=None, orb0=False)
                    cur["exacts"].append(jd)
                elif is_entry(kind, d):
                    if cur is not None:
                        problems.append(f"{a}-{b} {ASP[ai][0]}: orb entered twice at JD {jd:.5f}")
                        eps.append(cur)
                    cur = dict(pair=(a, b), ai=ai, start=jd, exacts=[], end=None, orb0=False)
                else:
                    if cur is None:
                        problems.append(f"{a}-{b} {ASP[ai][0]}: orb left twice at JD {jd:.5f}")
                        cur = dict(pair=(a, b), ai=ai, start=None, exacts=[], end=None, orb0=False)
                    cur["end"] = jd
                    eps.append(cur)
                    cur = None
            if cur is not None:
                eps.append(cur)
    return eps


# ------------------------------------------------- audit
def audit(db, init, t0, t1):
    """Independent proof: true in-orb state from swisseph vs replayed event parity."""
    if AUDIT_STEP <= 0:
        return 0
    global CALLS
    before = CALLS
    for (a, b), rows in db.items():
        toggles = {}
        for idx, jd, d in rows:
            ai, ti, kind = META[idx]
            if kind != 1:
                toggles.setdefault((ai, ti), []).append(jd)
        for k in toggles:
            toggles[k].sort()
        step = min(AUDIT_STEP, min(_win[a], _win[b]) / 8.0)
        for t in np.arange(t0, t1, step):
            t = float(t)
            r = rel(t, a, b)
            for ai, ti in KEYS:
                o = ASP[ai][2]
                if o <= 0:
                    continue
                tg = toggles.get((ai, ti), [])
                i = bisect.bisect_left(tg, t - 1e-5)
                if i < len(tg) and tg[i] < t + 1e-5:
                    continue                                   # too close to an event
                diff = abs(wrap180(r - TARGETS[ai][ti]))
                if abs(diff - o) < 1e-6:
                    continue                                   # sitting on an orb edge
                true_in = diff <= o
                replay = init[(a, b, ai, ti)] ^ (bisect.bisect_right(tg, t) % 2 == 1)
                assert replay == true_in, (
                    f"audit {a}-{b} {ASP[ai][0]}: replay says "
                    f"{'inside' if replay else 'outside'} orb but ephemeris disagrees at JD {t:.5f}: "
                    f"event missed")
    return CALLS - before


# ---------------------------------------------------------- tables
def _angle_txt(x):
    return f"{round(x, 3):g}"


def make_rows(eps, lo, hi):
    """One row per EXACT PASS (an orb episode with 2 retrograde re-crossings -> 2 rows)."""
    out = []
    for e in eps:
        s, ex, en = e["start"], e["exacts"], e["end"]
        t_lo = s if s is not None else (ex[0] if ex else lo)
        t_hi = en if en is not None else (ex[-1] if ex else hi)
        if not (t_lo < hi and t_hi >= lo):                      # must overlap the requested range
            continue
        a, b = e["pair"]
        name, ang, orb = ASP[e["ai"]]
        if e["orb0"]:
            s_txt = e_txt = "-"
        else:
            s_txt = jd_to_local(s) if s is not None else "before range"
            e_txt = jd_to_local(en) if en is not None else "after range"
        passes = ex if ex else [None]
        for n, x in enumerate(passes, 1):
            t_ref = x if x is not None else (s if s is not None else lo)
            if SHOW_SIGNS:
                sa = RASHI[int(lon(t_ref, a) // 30)]
                sb = RASHI[int(lon(t_ref, b) // 30)]
                label = f"{a} ({sa}) {name} {_angle_txt(ang)} {b} ({sb})"
            else:
                label = f"{a} {name} {_angle_txt(ang)} {b}"
            ex_txt = jd_to_local(x) if x is not None else "no exact (orb only)"
            note = f"pass {n}/{len(passes)}" if len(passes) > 1 else ""
            out.append((t_ref, name, label, s_txt, [ex_txt], e_txt,
                        [x] if x is not None else [], s, en, note))
    if SORT_BY == "time":
        out.sort(key=lambda r: r[0])
    else:
        out.sort(key=lambda r: (r[1], r[0]))
    return out


def show(rows):
    w_l = max([len(r[2]) for r in rows] + [20])
    w_n = max([len(r[1]) for r in rows] + [6])
    head = f"{'Aspect':<{w_n}} | {'Planets':<{w_l}} | {'Orb start ' + TZ_LABEL:<19} | " \
           f"{'Exact ' + TZ_LABEL:<19} | {'Orb exit ' + TZ_LABEL:<19}"
    print(head)
    print("-" * len(head))
    for _, name, label, s_txt, ex_txt, e_txt, ex, s, en, note in rows:
        tail = f"  [{note}]" if note else ""
        print(f"{name:<{w_n}} | {label:<{w_l}} | {s_txt:<19} | {ex_txt[0]:<19} | {e_txt:<19}{tail}")


def save_csv(rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Aspect", "Planets", f"Orb start {TZ_LABEL}", f"Exact {TZ_LABEL}",
                     f"Orb exit {TZ_LABEL}", "Orb start JD", "Exact JD", "Orb exit JD", "Note"])
        for _, name, label, s_txt, ex_txt, e_txt, ex, s, en, note in rows:
            f6 = lambda x: "" if x is None else f"{x:.{JD_DECIMALS}f}"
            wr.writerow([name, label, s_txt, ex_txt[0], e_txt,
                         f6(s), f6(ex[0] if ex else None), f6(en), note])


# --------------------------------------------------------------- main
if __name__ == "__main__":
    if not PAIRS or not ASP:
        raise SystemExit("Nothing to do: select at least 2 planets and 1 aspect.")
    s_jd = START_JD if START_JD is not None else swe.julday(*START_DATE)
    if START_JD is None and START_DATE_IS_LOCAL:
        s_jd -= TZ_OFFSET_HOURS / 24.0
    lo, hi = s_jd, s_jd + DAYS
    init_worker()

    probe = list(np.arange(lo - PAD, hi + PAD, 3652.5)) + [hi + PAD]
    for nm in ACTIVE:
        for t in probe:
            lon(float(t), nm)
    print(f"Ephemeris: {EPH_NAME[EPH_FLAG]}  |  {len(PAIRS)} pairs, {len(ASP)} aspects")

    t0 = time.perf_counter()
    db, calls, warns = build(lo - PAD, hi + PAD)
    elapsed = time.perf_counter() - t0

    if warns:
        msg = f"{len(warns)} warning(s); first: {warns[0]}"
        if STRICT:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    init = initial_states(lo - PAD)
    problems = []
    eps = episodes(db, init, problems)
    if problems:
        msg = f"{len(problems)} sequence problem(s); first: {problems[0]}"
        if STRICT or RUN_CHECK:
            raise RuntimeError(msg)
        print("WARNING:", msg)
    if RUN_CHECK:
        n = audit(db, init, lo - PAD, hi + PAD)
        print(f"Verification passed: enter/exit alternation + ephemeris audit ({n:,} audit calls)")

    rows = make_rows(eps, lo, hi)
    if PRINT_TABLE:
        show(rows)
        save_csv(rows, CSV_FILE)
    if PRINT_TIME:
        print(f"Execution time : {elapsed:.3f} s (excl. audit)")
    if PRINT_CALLS:
        print(f"Ephemeris calls: {calls:,}")
    swe.close()