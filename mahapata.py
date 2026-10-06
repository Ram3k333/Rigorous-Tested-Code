"""
Mahapata (Vyatipata / Vaidhriti) engine -- RIGOROUS edition
============================================================
Kranti-samya of Sun and Moon, found as Sun-Moon declination aspects:

  PARALLEL        g(t) = dec_sun - dec_moon = 0
  CONTRAPARALLEL  g(t) = dec_sun + dec_moon = 0

  Vyatipata = PARALLEL crossing       with Sun and Moon in DIFFERENT ayana
              (same gola automatically)
  Vaidhriti = CONTRAPARALLEL crossing with Sun and Moon in the SAME ayana
              (different gola automatically)

  gola  : sign of declination        (+ Uttara gola,  - Dakshina gola)
  ayana : sign of declination speed  (+ Uttarayana,   - Dakshinayana)
  This is identical to mahapat.py (Sun: tropical longitude 0-180 / 270-90,
  Moon: from its declination zero-crossings and max/min), but needs no
  transit scans: the signs are read directly at the exact madhya instant.

  Madhya   = instant of kranti-samya (g = 0)
  Prarambh = last time before madhya with |g| = yogardh
  Samapti  = first time after madhya with |g| = yogardh
  yogardh  = Sun semi-diameter + Moon semi-diameter (from true distances,
             identical to swe.pheno_ut diameter/2 -- no pheno_ut calls)

WHY A MAHAPATA CAN NEVER BE A "GRAZE"
  At a Vyatipata the declination speeds have opposite signs, so
  |g'| = |v_sun| + |v_moon| > 0; at a Vaidhriti they have the same sign,
  so again |g'| = |v_sun| + |v_moon| > 0. Every Mahapata is a clean,
  transversal crossing. Tangential touches (g' = 0) are always of the
  non-Mahapata kind.

DECLINATION IS NOT MONOTONIC
  Longitudes of Sun/Moon are monotonic; declinations are not (Moon turns
  every ~13.6 days, Sun at the solstices). So, exactly as in Planet Events:

LAYERS OF RIGOR
 1. FIT VALIDATION   - Chebyshev fit of each declination accepted only if the
                       tail is small AND it matches swisseph at 7 held-out points.
 2. MONOTONIC PIECES - g is cut at every stationary point; in each piece a root
                       exists iff the two end values have opposite signs
                       (at most one root per piece -> nothing can be skipped).
 3. SEAM OVERLAP     - pieces evaluated on [-1-EXT, 1+EXT]; duplicates merged.
 4. POLISH           - madhya and both edges re-solved against the REAL
                       ephemeris (Newton with analytic declination speed,
                       bracketed bisection fallback).
 5. CONTINUITY CHECK - consecutive crossings of each kind must alternate
                       direction (+ then -), otherwise one was skipped.
 6. INDEPENDENT AUDIT- true sign of g read from swisseph every AUDIT_STEP days;
                       the number of found crossings in between must match.

All time is computed in JD (UT). Positions are GEOCENTRIC, TRUE EQUATOR OF
DATE (no sidereal flag in the declination calculation), so every result is
universal (location-independent). Local time is display only.

FLAG column (empty for normal events)
  NO_EDGE          |g| did not reach yogardh within EDGE_MAX_DAYS on one side
  TURN             g changes direction inside the pata (asymmetric pata)
  SUN_AYAN_CHG     Sun's ayana changes inside the pata (solstice)
  MOON_AYAN_CHG    Moon's ayana changes inside the pata (Moon's max/min dec)
  SUN_GOL_CHG      Sun crosses the equator inside the pata
  MOON_GOL_CHG     Moon crosses the equator inside the pata
"""
import math
import os
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
START_DATE = (2025, 1, 1, 0.0)     # (year, month, day, hour) -> converted with swe.julday
START_JD   = None                  # set a JD here to override START_DATE
DAYS       = 365                # length of range (1900-01-01 .. 2500-01-01)
START_DATE_IS_LOCAL = True         # True -> START_DATE is local (TZ_OFFSET_HOURS), not UT

# ---- time display -----------------------------------------------------
SHOW_JD    = False                  # Start/Middle/End JD (UT) columns
SHOW_IST   = True                  # local time columns: YYYY-MM-DD HH:MM:SS
TZ_OFFSET_HOURS = 5.5              # IST = UT + 5:30 (change for another zone)
TZ_LABEL   = "IST"                 # column heading suffix
JD_DECIMALS = 8

# ---- ephemeris / position model ----------------------------------------
EPHEMERIS  = "jpl"              # "swieph" (SE .se1 files), "jpl" (JPL DE file) or "moseph" (built-in)
EPHE_PATH  =  r"C:\Users\durva\PycharmProjects\Rigorous_Tested_Code" # folder holding sepl_18/semo_18 (+ _24) .se1 (and the JPL file if "jpl")
JPL_FILE   = "de440.eph"           # file name inside EPHE_PATH
REQUIRE_EPHEMERIS = True           # True -> stop if swisseph silently falls back to a lower-quality ephemeris
TRUE_POSITION      = False         # True -> geometric (no light-time)
NO_ABERRATION      = False         # True -> skip annual aberration
NO_GRAV_DEFLECTION = False         # True -> skip light deflection (negligible for Sun/Moon)
NO_NUTATION        = False         # True -> mean equator of date instead of true equator
                                   # (keep False: traditional kranti = true declination)

# ---- longitude display (does NOT affect the calculation) ---------------
LON_MODE   = "tropical"            # "tropical", "sidereal" or "both"
AYANAMSA   = "LAHIRI"              # any swe.SIDM_<name> (used only for sidereal display)
ANGLE_FORMAT   = "dms"             # "dms" (23°26'12.3") or "decimal"
ANGLE_DECIMALS = 6                 # decimals in "decimal" mode
LANGUAGE   = "english"             # "english" or "hindi" (paat / ayan / gol names)

# ---- mahapata rules -----------------------------------------------------
EDGE_METHOD   = "exact"            # "exact" (solve |g| = yogardh) or "linear" (Surya Siddhanta style)
SEMI_DIAMETER = "true"             # "true" (from distance, = pheno_ut) or "mean" (fixed values below)
SUN_RADIUS_KM  = 696000.0          # Swiss Ephemeris value (diameter 1392000 km)
MOON_RADIUS_KM = 1737.4
MEAN_SUN_SD_DEG  = 15 / 60 + 59.6 / 3600   # 15'59.6"  (used only if SEMI_DIAMETER = "mean")
MEAN_MOON_SD_DEG = 15 / 60 + 32.6 / 3600   # 15'32.6"
EDGE_MAX_DAYS = 5.0                # search limit for prarambh / samapti on each side
                                   # (longest side seen 1990-2030: 2.4 days; NO_EDGE flag if exceeded)
SHOW_FLAG_COLUMN    = True         # False -> no Flag column at all

# Each condition, one of:
#   "show"    -> keep the event, write the flag in the Flag column
#   "hide"    -> keep the event, condition is NOT calculated, no flag
#   "exclude" -> event is REMOVED from the list (its edges/longitudes are not polished)
FLAG_MODE = {
    "NO_EDGE":       "show",       # edge not found within EDGE_MAX_DAYS
    "TURN":          "show",       # Sun-Moon declination gap reverses inside the pata
    "SUN_AYAN_CHG":  "show",       # solstice inside the pata
    "MOON_AYAN_CHG": "show",       # Moon's max/min declination inside the pata
    "SUN_GOL_CHG":   "show",       # Sun crosses the equator inside the pata
    "MOON_GOL_CHG":  "show",       # Moon crosses the equator inside the pata
}
SHOW_OTHER_CROSSINGS = False       # True -> also list non-Mahapata parallel / contraparallel crossings

# ---- output ------------------------------------------------------------
PRINT_TABLE  = True
PRINT_TIME   = True
PRINT_CALLS  = True
CSV_FILE     = "mahapata.csv"      # None -> no CSV (written as UTF-8 with BOM, opens in Excel)

# ---- rigor / verification ---------------------------------------------
RUN_CHECK   = True                 # crossing-direction alternation check
AUDIT_STEP  = 2.0                  # days between true-ephemeris audit samples (0 = off)
STRICT      = False                # True -> any warning is a hard error

# ---- performance / accuracy -------------------------------------------
WORKERS   = 4
CHUNK     = 365.25                 # days per worker task
WIN_SUN,  DEG_SUN  = 20.0, 16      # fit window (days) / Chebyshev degree
WIN_MOON, DEG_MOON = 8.0, 20
TOL       = 1e-7                   # max tail coefficient (deg) before a window is split
VAL_TOL   = 1e-6                   # max error (deg) at held-out validation points
MIN_WIN   = 0.05
POLISH    = True                   # re-solve madhya and edges against the true ephemeris
# ======================== END CONFIG ================================

# ---- internal constants ------------------------------------------------
EXT = 1e-7                         # piece overlap in x (seam safety)
DEDUPE_DAYS = 1e-4                 # same-kind crossings closer than this = seam duplicate
SPEED_MARGIN = 1e-3                # deg/day; below this the polynomial's ayana sign is re-checked on the ephemeris
POLISH_TOL_DEG = 1e-9              # stop Newton when |residual| below this
POLISH_MAX_SHIFT = 0.05            # days; larger Newton shift -> bracketed bisection
AU_KM = 149597870.7
VAL_X = np.array([-0.93, -0.71, -0.37, 0.11, 0.43, 0.77, 0.96])
SCAN_X = np.linspace(-1.0, 1.0, 129)
PAR, CON = "PARALLEL", "CONTRAPARALLEL"
SIGN = {PAR: -1.0, CON: +1.0}      # g = dec_sun + SIGN * dec_moon

EPH_FLAG = {"swieph": swe.FLG_SWIEPH, "jpl": swe.FLG_JPLEPH, "moseph": swe.FLG_MOSEPH}[EPHEMERIS]
EPH_NAME = {swe.FLG_SWIEPH: "Swiss Ephemeris files", swe.FLG_JPLEPH: "JPL " + JPL_FILE,
            swe.FLG_MOSEPH: "Moshier (built-in)"}
MODS = 0
if TRUE_POSITION:      MODS |= swe.FLG_TRUEPOS
if NO_ABERRATION:      MODS |= swe.FLG_NOABERR
if NO_GRAV_DEFLECTION: MODS |= swe.FLG_NOGDEFL
if NO_NUTATION:        MODS |= swe.FLG_NONUT
FIT_FLAGS = EPH_FLAG | MODS | swe.FLG_EQUATORIAL                # declination only
POL_FLAGS = FIT_FLAGS | swe.FLG_SPEED                           # + declination speed
LON_FLAGS = EPH_FLAG | MODS                                     # ecliptic, tropical
SID_FLAGS = LON_FLAGS | swe.FLG_SIDEREAL


def _cheb_setup(deg):
    k = np.arange(deg + 1)
    nodes = -np.cos(np.pi * (k + 0.5) / (deg + 1))
    return nodes, np.linalg.inv(C.chebvander(nodes, deg))


FIT = {swe.SUN: (WIN_SUN, DEG_SUN) + _cheb_setup(DEG_SUN),
       swe.MOON: (WIN_MOON, DEG_MOON) + _cheb_setup(DEG_MOON)}
DEG_G = max(DEG_SUN, DEG_MOON)
NODES_G, FIT_M_G = _cheb_setup(DEG_G)

NAMES = {
    "english": {"Vyatipata": "Vyatipata", "Vaidhriti": "Vaidhriti",
                PAR: "Parallel", CON: "Contraparallel",
                "N_ayan": "Uttarayan", "S_ayan": "Dakshinayan",
                "N_gol": "Uttar Gol", "S_gol": "Dakshin Gol"},
    "hindi": {"Vyatipata": "व्यतिपात", "Vaidhriti": "वैधृति",
              PAR: "समक्रान्ति", CON: "विपरीतक्रान्ति",
              "N_ayan": "उत्तरायण", "S_ayan": "दक्षिणायण",
              "N_gol": "उत्तरगोल", "S_gol": "दक्षिणगोल"},
}[LANGUAGE]


# ----------------------------------------------------- ephemeris wrappers
CALLS = 0
WARN = []


def _used_eph(ret):
    for f in (swe.FLG_JPLEPH, swe.FLG_SWIEPH, swe.FLG_MOSEPH):
        if ret & f:
            return f
    return swe.FLG_MOSEPH


def _calc(jd, body, flags):
    """swe.calc_ut + fallback guard. Counted."""
    global CALLS
    CALLS += 1
    xx, ret = swe.calc_ut(jd, body, flags)
    if REQUIRE_EPHEMERIS and not (ret & EPH_FLAG):
        y, m, d, _ = swe.revjul(jd)
        raise RuntimeError(
            f"Ephemeris fallback at JD {jd:.5f} ({y:04d}-{m:02d}-{d:02d}): asked for "
            f"{EPH_NAME[EPH_FLAG]}, got {EPH_NAME[_used_eph(ret)]}. "
            f"Check EPHE_PATH={EPHE_PATH!r} contains the file covering this date.")
    return xx


def dec(jd, body):
    return _calc(jd, body, FIT_FLAGS)[1]


class State:
    """True Sun/Moon equatorial state at one instant (2 calls)."""
    __slots__ = ("jd", "ds", "dm", "vs", "vm", "rs", "rm")

    def __init__(self, jd):
        s = _calc(jd, swe.SUN, POL_FLAGS)
        m = _calc(jd, swe.MOON, POL_FLAGS)
        self.jd = jd
        self.ds, self.dm = s[1], m[1]          # declination (deg)
        self.vs, self.vm = s[4], m[4]          # declination speed (deg/day)
        self.rs, self.rm = s[2], m[2]          # distance (AU)

    def g(self, kind):
        return self.ds + SIGN[kind] * self.dm

    def gdot(self, kind):
        return self.vs + SIGN[kind] * self.vm

    def yogardh(self):
        if SEMI_DIAMETER == "mean":
            return MEAN_SUN_SD_DEG + MEAN_MOON_SD_DEG
        return (math.degrees(math.asin(SUN_RADIUS_KM / (self.rs * AU_KM))) +
                math.degrees(math.asin(MOON_RADIUS_KM / (self.rm * AU_KM))))


def init_worker():
    if EPHE_PATH:
        swe.set_ephe_path(EPHE_PATH)
    if EPHEMERIS == "jpl":
        swe.set_jpl_file(JPL_FILE)
    swe.set_sid_mode(getattr(swe, "SIDM_" + AYANAMSA))


# ------------------------------------------------------------- fitting
def fit(body, a, b):
    """Adaptive, validated Chebyshev fit of declination on [a, b]."""
    _, deg, nodes, fit_m = FIT[body]
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    c = fit_m @ np.array([dec(mid + half * x, body) for x in nodes])
    ok = max(abs(c[-1]), abs(c[-2])) <= TOL
    if ok:
        truth = np.array([dec(mid + half * x, body) for x in VAL_X])
        ok = np.max(np.abs(truth - C.chebval(VAL_X, c))) <= VAL_TOL
    if ok:
        return [(a, b, c)]
    if (b - a) > MIN_WIN:
        return fit(body, a, mid) + fit(body, mid, b)
    WARN.append(f"fit not converged for body {body} on [{a:.4f},{b:.4f}] at MIN_WIN")
    return [(a, b, c)]


def fit_range(body, lo, hi):
    win = FIT[body][0]
    out = []
    for i in range(int(math.ceil((hi - lo) / win))):
        out += fit(body, lo + i * win, min(lo + (i + 1) * win, hi))
    return out


def _restrict(windows, starts, a, b):
    """Re-express the fitted polynomial on the sub-interval [a,b] (no ephemeris calls;
    exact up to rounding because the polynomial degree <= DEG_G)."""
    wa, wb, c = windows[bisect.bisect_right(starts, 0.5 * (a + b)) - 1]
    t = 0.5 * (a + b) + 0.5 * (b - a) * NODES_G
    x = (t - 0.5 * (wa + wb)) / (0.5 * (wb - wa))
    return FIT_M_G @ C.chebval(x, c)


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
    """All candidate stationary points (roots of derivative d) from TWO methods:
    eigenvalue roots AND a dense sign-change scan."""
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


def true_turns(c):
    """Stationary points where the derivative really changes sign -> [x]."""
    d = C.chebder(c)
    out = []
    for x in stationary_x(d):
        if -1.0 <= x <= 1.0:
            if (C.chebval(x - 1e-6, d) > 0) != (C.chebval(x + 1e-6, d) > 0):
                out.append(x)
    return out


def _solve_piece(c, d, xa, xb, L, ga, gb):
    """Root of c(x)=L inside a MONOTONIC piece whose ends straddle L.
    Safeguarded Newton + bisection."""
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


def level_roots(c, L):
    """EVERY root of c(x) = L on [-1-EXT, 1+EXT] -> [(x, direction)].
    Cut at all stationary points; a monotonic piece has a root iff its ends straddle L."""
    d = C.chebder(c)
    xs = [x for x in stationary_x(d) if -1.0 - EXT < x < 1.0 + EXT]
    xk = [-1.0 - EXT] + xs + [1.0 + EXT]
    vk = C.chebval(np.array(xk), c)
    out = []
    for i in range(len(xk) - 1):
        v0, v1 = float(vk[i]), float(vk[i + 1])
        if (v0 >= L) == (v1 >= L):
            continue
        x = _solve_piece(c, d, xk[i], xk[i + 1], L, v0 - L, v1 - L)
        out.append((x, 1 if v1 >= L else -1))
    return out


# ----------------------------------------------------- polishing (true ephemeris)
def polish(jd_fit, kind, level_sign=0.0, y_fixed=None):
    """Solve g(t) - level_sign * yogardh(t) = 0 on the REAL ephemeris.
    level_sign = 0 -> madhya (g = 0); +1 / -1 -> edge (g = +Y / -Y).
    Newton with the analytic declination speed; bracketed bisection fallback.
    Returns (jd, State)."""
    def f(st):
        y = y_fixed if y_fixed is not None else st.yogardh()
        return st.g(kind) - level_sign * y

    st = State(jd_fit)
    jd, fv = jd_fit, f(st)
    for _ in range(6):
        if abs(fv) < POLISH_TOL_DEG:
            break
        slope = st.gdot(kind)
        if abs(slope) < 1e-9:
            break
        jd_new = jd - fv / slope
        if abs(jd_new - jd_fit) > POLISH_MAX_SHIFT:
            break
        st = State(jd_new)
        jd, fv = jd_new, f(st)
    if abs(fv) < 1e-7 and abs(jd - jd_fit) < POLISH_MAX_SHIFT:
        return jd, st
    # bracketed fallback (rare)
    h = 1e-3
    for _ in range(8):
        sl, sr = State(jd_fit - h), State(jd_fit + h)
        fl, fr = f(sl), f(sr)
        if (fl >= 0) != (fr >= 0):
            lo, hi = jd_fit - h, jd_fit + h
            while hi - lo > 1e-8:
                mid = 0.5 * (lo + hi)
                if (f(State(mid)) >= 0) == (fl >= 0):
                    lo = mid
                else:
                    hi = mid
            jd = 0.5 * (lo + hi)
            return jd, State(jd)
        h *= 4.0
    WARN.append(f"polish bracket failed near JD {jd_fit:.5f} ({kind}, level {level_sign:+.0f}); kept fit root")
    return jd_fit, State(jd_fit)


# --------------------------------------------------------------- engine
class Piecewise:
    """Sun & Moon declination as a list of common sub-intervals, each with
    Chebyshev series S (Sun), M (Moon), on x in [-1, 1]."""

    def __init__(self, lo, hi):
        sun = fit_range(swe.SUN, lo, hi)
        moon = fit_range(swe.MOON, lo, hi)
        s_st, m_st = [w[0] for w in sun], [w[0] for w in moon]
        cuts = sorted(set(s_st + m_st + [hi]))
        self.iv = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a > 1e-9:
                self.iv.append((a, b, _restrict(sun, s_st, a, b), _restrict(moon, m_st, a, b)))
        self.starts = [v[0] for v in self.iv]

    @staticmethod
    def g_series(S, M, kind):
        return S + SIGN[kind] * M

    def overlapping(self, t0, t1):
        i = max(bisect.bisect_right(self.starts, t0) - 1, 0)
        while i < len(self.iv) and self.iv[i][0] < t1:
            yield self.iv[i]
            i += 1

    def times(self, t0, t1, fn):
        """Collect fn(series-tuple) x-values in [t0, t1] as times."""
        out = []
        for a, b, S, M in self.overlapping(t0, t1):
            half, mid = 0.5 * (b - a), 0.5 * (a + b)
            for x in fn(S, M):
                t = mid + half * (x[0] if isinstance(x, tuple) else x)
                if t0 < t < t1:
                    out.append(t)
        return out


def _dedupe(rows, key):
    rows.sort(key=lambda r: r["fit_jd"])
    out, last = [], {}
    for r in rows:
        k = key(r)
        if k in last and r["fit_jd"] - last[k] < DEDUPE_DAYS:
            continue
        last[k] = r["fit_jd"]
        out.append(r)
    return out


def classify(kind, vs, vm):
    if kind == PAR and vs * vm < 0:
        return "Vyatipata"
    if kind == CON and vs * vm > 0:
        return "Vaidhriti"
    return None


def job(args):
    """All crossings with madhya in [c0, c1). Runs in a worker process."""
    global CALLS, WARN
    CALLS = 0
    WARN = []
    init_worker()
    c0, c1 = args
    pw = Piecewise(c0 - EDGE_MAX_DAYS - 1.0, c1 + EDGE_MAX_DAYS + 1.0)

    # ---- 1. every crossing of g = 0 (both kinds) from monotonic pieces
    cands = []
    for a, b, S, M in pw.iv:
        half, mid = 0.5 * (b - a), 0.5 * (a + b)
        dS, dM = C.chebder(S), C.chebder(M)
        for kind in (PAR, CON):
            for x, direction in level_roots(pw.g_series(S, M, kind), 0.0):
                t = mid + half * x
                if c0 <= t < c1:
                    cands.append({"kind": kind, "fit_jd": t, "dir": direction,
                                  "vs": float(C.chebval(x, dS)) / half,
                                  "vm": float(C.chebval(x, dM)) / half})
    cands = _dedupe(cands, key=lambda r: r["kind"])

    # ---- 2. classify, polish, edges, flags
    rows = []
    for r in cands:
        kind = r["kind"]
        paat = classify(kind, r["vs"], r["vm"])
        sure = min(abs(r["vs"]), abs(r["vm"])) > SPEED_MARGIN
        if paat is None and sure and not SHOW_OTHER_CROSSINGS:
            rows.append({"kind": kind, "paat": None, "jd": r["fit_jd"], "dir": r["dir"]})
            continue
        if POLISH:
            jd_m, st = polish(r["fit_jd"], kind)
        else:
            jd_m, st = r["fit_jd"], State(r["fit_jd"])
        paat = classify(kind, st.vs, st.vm)
        row = {"kind": kind, "paat": paat, "jd": jd_m, "dir": r["dir"], "st": st}
        if paat is not None or SHOW_OTHER_CROSSINGS:
            row.update(edges_and_flags(pw, kind, jd_m, st))
            if not row["excluded"]:
                row.update(longitudes(jd_m))
        rows.append(row)
    return rows, CALLS, WARN


def edges_and_flags(pw, kind, jd_m, st):
    """1. edges from the fitted polynomials, 2. conditions (no calls),
    3. if an "exclude" condition holds -> stop here (no polishing calls),
    otherwise polish the edges on the true ephemeris."""
    y0 = st.yogardh()
    if EDGE_METHOD == "linear":
        half_dur = y0 / abs(st.gdot(kind))
        pr = (jd_m - half_dur, None)
        sm = (jd_m + half_dur, None)
    else:
        lo, hi = jd_m - EDGE_MAX_DAYS, jd_m + EDGE_MAX_DAYS
        before, after = [], []
        for sgn in (+1.0, -1.0):
            for a, b, S, M in pw.overlapping(lo, hi):
                half, mid = 0.5 * (b - a), 0.5 * (a + b)
                for x, _ in level_roots(pw.g_series(S, M, kind), sgn * y0):
                    t = mid + half * x
                    if lo < t < jd_m:
                        before.append((t, sgn))
                    elif jd_m < t < hi:
                        after.append((t, sgn))
        pr = max(before) if before else None
        sm = min(after) if after else None

    # ---- conditions, only those not set to "hide" (no ephemeris calls)
    found = []
    if FLAG_MODE["NO_EDGE"] != "hide" and (pr is None or sm is None):
        found.append("NO_EDGE")
    t0 = pr[0] if pr is not None else jd_m - EDGE_MAX_DAYS
    t1 = sm[0] if sm is not None else jd_m + EDGE_MAX_DAYS
    tests = (("TURN",          lambda S, M: true_turns(pw.g_series(S, M, kind))),
             ("SUN_AYAN_CHG",  lambda S, M: true_turns(S)),
             ("MOON_AYAN_CHG", lambda S, M: true_turns(M)),
             ("SUN_GOL_CHG",   lambda S, M: level_roots(S, 0.0)),
             ("MOON_GOL_CHG",  lambda S, M: level_roots(M, 0.0)))
    for name, fn in tests:
        if FLAG_MODE[name] != "hide" and pw.times(t0, t1, fn):
            found.append(name)
    excluded = [n for n in found if FLAG_MODE[n] == "exclude"]
    shown = [n for n in found if FLAG_MODE[n] == "show"]
    if excluded:
        return {"excluded": excluded, "flags": ""}

    # ---- polish the edges on the true ephemeris
    y_fix = y0 if SEMI_DIAMETER == "mean" else None
    edge = []
    for e in (pr, sm):
        if e is None:
            edge.append(None)
        elif EDGE_METHOD == "linear" or not POLISH:
            edge.append(e[0])
        else:
            edge.append(polish(e[0], kind, e[1], y_fix)[0])
    return {"prarambh": edge[0], "samapti": edge[1], "yogardh": y0,
            "excluded": [], "flags": " ".join(shown) if SHOW_FLAG_COLUMN else ""}


def longitudes(jd):
    out = {}
    if LON_MODE in ("tropical", "both"):
        out["sun_lon"] = _calc(jd, swe.SUN, LON_FLAGS)[0] % 360.0
        out["moon_lon"] = _calc(jd, swe.MOON, LON_FLAGS)[0] % 360.0
    if LON_MODE in ("sidereal", "both"):
        out["sun_lon_sid"] = _calc(jd, swe.SUN, SID_FLAGS)[0] % 360.0
        out["moon_lon_sid"] = _calc(jd, swe.MOON, SID_FLAGS)[0] % 360.0
    return out


def build(lo, hi):
    edges = list(np.arange(lo, hi, CHUNK)) + [hi]
    tasks = list(zip(edges[:-1], edges[1:]))
    if WORKERS > 1:
        with ProcessPoolExecutor(WORKERS, initializer=init_worker) as ex:
            res = list(ex.map(job, tasks))
    else:
        res = [job(t) for t in tasks]
    rows, calls, warns = [], 0, []
    for r, n, w in res:
        rows += r
        calls += n
        warns += w
    rows.sort(key=lambda r: r["jd"])
    return rows, calls, warns


# ------------------------------------------------- verification layers
def check(rows):
    """Layer 5: consecutive crossings of one kind must alternate direction."""
    for kind in (PAR, CON):
        ev = [r for r in rows if r["kind"] == kind]
        for e0, e1 in zip(ev[:-1], ev[1:]):
            assert e0["dir"] != e1["dir"], (
                f"{kind}: two consecutive crossings in the same direction "
                f"(JD {e0['jd']:.5f}, {e1['jd']:.5f}): crossing skipped/duplicated")


def audit(rows, lo, hi):
    """Layer 6: independent proof against the real ephemeris. The true sign of g
    is sampled every AUDIT_STEP days; the parity of found crossings in between
    must equal the observed sign change, and every crossing must start from the
    current sign."""
    if AUDIT_STEP <= 0:
        return 0
    global CALLS
    before = CALLS
    jds = {k: [r["jd"] for r in rows if r["kind"] == k] for k in (PAR, CON)}
    dirs = {k: [r["dir"] for r in rows if r["kind"] == k] for k in (PAR, CON)}
    prev = None
    for t in np.arange(lo, hi, AUDIT_STEP):
        t = float(t)
        for _ in range(3):                                  # step off exact event times
            for k in (PAR, CON):
                i = bisect.bisect_left(jds[k], t - 1e-5)
                if i < len(jds[k]) and jds[k][i] < t + 1e-5:
                    t += 3e-5
        d_s, d_m = dec(t, swe.SUN), dec(t, swe.MOON)
        now = {k: (d_s + SIGN[k] * d_m) >= 0 for k in (PAR, CON)}
        if prev is not None:
            t_prev, s_prev = prev
            for k in (PAR, CON):
                i0, i1 = bisect.bisect_right(jds[k], t_prev), bisect.bisect_right(jds[k], t)
                cur = s_prev[k]
                for j in range(i0, i1):
                    assert cur == (dirs[k][j] < 0), (
                        f"audit {k}: crossing at JD {jds[k][j]:.5f} does not follow the current sign")
                    cur = dirs[k][j] > 0
                assert cur == now[k], (
                    f"audit {k}: replay disagrees with ephemeris at JD {t:.5f} "
                    f"(window {t_prev:.5f}..{t:.5f}): crossing missed")
        prev = (t, now)
    return CALLS - before


# ---------------------------------------------------------- formatting
def jd_to_local(jd):
    """JD (UT) -> 'YYYY-MM-DD HH:MM:SS' local. Integer seconds: no 60-s carry bugs."""
    if jd is None:
        return "-"
    total = int(round((jd + TZ_OFFSET_HOURS / 24.0 + 0.5) * 86400.0))
    day, sod = divmod(total, 86400)
    y, m, d, _ = swe.revjul(float(day))
    return f"{y:04d}-{m:02d}-{d:02d} {sod // 3600:02d}:{sod % 3600 // 60:02d}:{sod % 60:02d}"


def fmt_angle(v, signed=False):
    if ANGLE_FORMAT == "decimal":
        return f"{v:+.{ANGLE_DECIMALS}f}" if signed else f"{v:.{ANGLE_DECIMALS}f}"
    sgn = "-" if v < 0 else ("+" if signed else "")
    tenths = int(round(abs(v) * 36000.0))                # 0.1 arcsec units
    d, rem = divmod(tenths, 36000)
    mnt, sec = divmod(rem, 600)
    return f"{sgn}{d}°{mnt:02d}'{sec // 10:02d}.{sec % 10}\""


def fmt_jd(jd):
    return "-" if jd is None else f"{jd:.{JD_DECIMALS}f}"


def table_rows(rows, lo, hi):
    """-> (header, list of row lists) for the reported events."""
    head = ["Paat"]
    if SHOW_IST:
        head += [f"Start ({TZ_LABEL})", f"Middle ({TZ_LABEL})", f"End ({TZ_LABEL})"]
    if SHOW_JD:
        head += ["Start JD (UT)", "Middle JD (UT)", "End JD (UT)"]
    head += ["Sun Dec", "Moon Dec"]
    if LON_MODE in ("tropical", "both"):
        head += ["Sun Lon", "Moon Lon"] if LON_MODE == "tropical" else ["Sun Lon (Trop)", "Moon Lon (Trop)"]
    if LON_MODE in ("sidereal", "both"):
        head += ["Sun Lon (Sid)", "Moon Lon (Sid)"]
    head += ["Sun Ayan", "Sun Gol", "Moon Ayan", "Moon Gol"]
    if SHOW_FLAG_COLUMN:
        head.append("Flag")

    out = []
    for r in rows:
        if not lo <= r["jd"] < hi or "st" not in r or "prarambh" not in r:
            continue
        if r["paat"] is None and not SHOW_OTHER_CROSSINGS:
            continue
        st = r["st"]
        name = NAMES[r["paat"]] if r["paat"] else NAMES[r["kind"]]
        line = [name]
        if SHOW_IST:
            line += [jd_to_local(r["prarambh"]), jd_to_local(r["jd"]), jd_to_local(r["samapti"])]
        if SHOW_JD:
            line += [fmt_jd(r["prarambh"]), fmt_jd(r["jd"]), fmt_jd(r["samapti"])]
        line += [fmt_angle(st.ds, True), fmt_angle(st.dm, True)]
        if LON_MODE in ("tropical", "both"):
            line += [fmt_angle(r["sun_lon"]), fmt_angle(r["moon_lon"])]
        if LON_MODE in ("sidereal", "both"):
            line += [fmt_angle(r["sun_lon_sid"]), fmt_angle(r["moon_lon_sid"])]
        line += [NAMES["N_ayan"] if st.vs >= 0 else NAMES["S_ayan"],
                 NAMES["N_gol"] if st.ds >= 0 else NAMES["S_gol"],
                 NAMES["N_ayan"] if st.vm >= 0 else NAMES["S_ayan"],
                 NAMES["N_gol"] if st.dm >= 0 else NAMES["S_gol"]]
        if SHOW_FLAG_COLUMN:
            line.append(r["flags"])
        out.append(line)
    return head, out


def show(head, body):
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)).rstrip())
    for r in body:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def save_csv(head, body, filename):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(head)
        wr.writerows(body)


# --------------------------------------------------------------- main
if __name__ == "__main__":
    s_jd = START_JD if START_JD is not None else swe.julday(*START_DATE)
    if START_JD is None and START_DATE_IS_LOCAL:
        s_jd -= TZ_OFFSET_HOURS / 24.0
    lo, hi = s_jd, s_jd + DAYS
    init_worker()

    # ---- ephemeris pre-flight: both bodies, every ~10 years, both ends
    pad = EDGE_MAX_DAYS + 1.0
    probe = list(np.arange(lo - pad, hi + pad, 3652.5)) + [hi + pad]
    for body in (swe.SUN, swe.MOON):
        for t in probe:
            dec(float(t), body)                     # raises if the file is missing/out of range
    print(f"Ephemeris: {EPH_NAME[EPH_FLAG]}"
          + (f"  (path: {EPHE_PATH})" if EPHE_PATH else "")
          + ("  [fallback guard ON]" if REQUIRE_EPHEMERIS else "  [fallback guard OFF]"))
    print(f"Edges: {EDGE_METHOD}, semi-diameters: {SEMI_DIAMETER}, positions: geocentric, "
          f"{'mean' if NO_NUTATION else 'true'} equator of date")

    t0 = time.perf_counter()
    rows, calls, warns = build(lo, hi)
    elapsed = time.perf_counter() - t0

    if warns:
        msg = f"{len(warns)} warning(s); first: {warns[0]}"
        if STRICT:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    CALLS = 0
    if RUN_CHECK:
        check(rows)
        audit_calls = audit(rows, lo, hi)
        print("Verification passed: direction-alternation check"
              + (f" + ephemeris audit ({audit_calls:,} audit calls)" if AUDIT_STEP > 0 else ""))

    head, body = table_rows(rows, lo, hi)
    in_range = [r for r in rows if lo <= r["jd"] < hi and r["paat"]]
    kept = [r for r in in_range if not r.get("excluded")]
    n_vy = sum(1 for r in kept if r["paat"] == "Vyatipata")
    n_va = sum(1 for r in kept if r["paat"] == "Vaidhriti")
    n_fl = sum(1 for r in kept if r.get("flags"))
    n_ex = {}
    for r in in_range:
        for n in r.get("excluded") or []:
            n_ex[n] = n_ex.get(n, 0) + 1
    if PRINT_TABLE:
        show(head, body)
    if CSV_FILE:
        save_csv(head, body, CSV_FILE)
    print(f"Vyatipata: {n_vy:,}   Vaidhriti: {n_va:,}   flagged: {n_fl:,}   "
          f"excluded: {len(in_range) - len(kept):,}"
          + (f" {n_ex}" if n_ex else "")
          + f"   (all Sun-Moon declination crossings examined: {len(rows):,})")
    if PRINT_TIME:
        print(f"Execution time : {elapsed:.3f} s (excl. audit)")
    if PRINT_CALLS:
        print(f"Ephemeris calls: {calls:,}  (~{calls / (DAYS / 365.25):,.0f} per year)")
    swe.close()