"""
Lagna Pravesh (Ascendant entering Rashi / Nakshatra / Pada) -- RIGOROUS edition
==============================================================================
Same style as the planet engine: flag-controlled, longitude-only, verified.

DESIGN
 * Ascendant only moves forward (latitude is limited by LAT_LIMIT), so there is
   no scanning, no speed, no station or retrograde handling.
 * ONE grid is solved: the 108 pada boundaries, strictly in sequence
   (k, k+1, k+2 ...). Rashi (every 9th) and Nakshatra (every 4th) are derived
   from the same root, so all three levels always agree to the last digit.
 * Each root is seeded from a low-order extrapolation of the previous roots,
   then solved against the REAL swisseph Ascendant (Newton with estimated
   pace -> Illinois once bracketed). Convergence test is on residual (deg).
 * Ascendant is read from ascmc[0] (independent of house system).
 * 1 houses_ex call per evaluation (about 3-5 calls per event).

VERIFICATION
 1. Chain check   - pada index +1 each event, times strictly increasing.
 2. Sidereal-day  - event k+108 must be one sidereal day after event k.
 3. Audit         - true pada read from swisseph every AUDIT_STEP days and the
                    event list replayed between samples.
 4. Event audit   - (optional) true pada just before / after every event.
"""
import math
import time
import bisect
import csv
import swisseph as swe

# =====================================================================
#                              CONFIG
# =====================================================================
# ---- locations: (lat, lon, "label") -- lon east positive --------------

LOCATIONS = [
    (24.5859, 74.8566, "Location"),
]
LAT_LIMIT  = 60.0                  # refuse |lat| above this (Ascendant stays forward-only)

# ---- time range (Julian Day, UT; no datetime used) --------------------
START_DATE = (2026, 1, 1, 0.0)     # (year, month, day, hour) -> swe.julday
START_JD   = None                  # set a JD here to override START_DATE
DAYS       = 1.0                   # length of range
PAD        = 0.05                  # days solved outside the range (seeding + edge safety)
START_DATE_IS_LOCAL = True         # True -> START_DATE is local (TZ_OFFSET_HOURS), not UT

# ---- time display -----------------------------------------------------
SHOW_JD    = False
SHOW_IST   = True
TZ_OFFSET_HOURS = 5.5
TZ_LABEL   = "IST"

# ---- coordinate system ------------------------------------------------
SIDEREAL   = True
AYANAMSA   = "LAHIRI"              # any swe.SIDM_<name> (used only if SIDEREAL)
EQUINOX_J2000 = False              # tropical only: fixed J2000 ecliptic
NO_NUTATION   = False              # True -> mean (not true) equinox
HOUSE_SYSTEM  = b'A'               # only used to call houses_ex; ascmc[0] is system-free

# ---- ephemeris ---------------------------------------------------------
EPHEMERIS  = "swieph"              # "swieph" or "moseph"
EPHE_PATH  = None                  # folder with .se1 files, or None

# ---- which events ------------------------------------------------------
SHOW_RASHI     = True
SHOW_NAKSHATRA = False              # (tropical mode: computed, but not meaningful)
SHOW_PADA      = False

# ---- output ------------------------------------------------------------
PRINT_TABLE    = True
PRINT_TIME     = True
PRINT_CALLS    = True
SAVE_CSV       = False
CSV_FILE       = "lagna_rigorous.csv"
SHOW_LOCATION_COLUMN = None        # None -> only when more than one location
NUMBER_LABELS  = False             # "Makara (10)", "Shravana (22)"
RASHI_LANGUAGE = "english"         # "auto" / "sanskrit" / "english"
SORT_BY        = "location"            # "location" (grouped) or "time"
JD_DECIMALS    = 8

# ---- rigor / verification ---------------------------------------------
RUN_CHECK        = True            # chain + sidereal-day checks (0 ephemeris calls)
AUDIT_STEP       = 0.25            # days between true-ephemeris audit samples (0 = off, max 0.4)
AUDIT_EACH_EVENT = False           # True -> 2 extra calls per event, proves root to EVENT_EPS_SEC
EVENT_EPS_SEC    = 0.05
SIDEREAL_DAY_TOL = 1e-4            # days; allowed error in the k / k+108 spacing (~8.6 s)
STRICT           = True            # True -> any warning is a hard error

# ---- solver accuracy ---------------------------------------------------
TIME_TOL_SEC = 0.001               # target root accuracy (JD float floor is ~5e-5 s)
MAX_ITER     = 40
EXTRAP_POINTS = 4                  # roots used to predict the next one (2..4)
STEP_CLAMP   = 0.05                # days; largest single Newton move before a bracket exists
# ======================== END CONFIG ================================

# ---- internal constants ------------------------------------------------
GRID = 108
STEP_DEG = 360.0 / GRID            # 3d20'
SIDEREAL_DAY = 0.99726957
FLAGS = swe.FLG_MOSEPH if EPHEMERIS == "moseph" else swe.FLG_SWIEPH
if SIDEREAL:
    FLAGS |= swe.FLG_SIDEREAL
elif EQUINOX_J2000:
    FLAGS |= swe.FLG_J2000
if NO_NUTATION:
    FLAGS |= swe.FLG_NONUT

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

WARN = []


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


def wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


def init_swe():
    if EPHE_PATH:
        swe.set_ephe_path(EPHE_PATH)
    if SIDEREAL:
        swe.set_sid_mode(getattr(swe, "SIDM_" + AYANAMSA))


# ------------------------------------------------------------- site
class Site:
    """One location. Counts every houses_ex call."""

    def __init__(self, lat, lon, label):
        if abs(lat) > LAT_LIMIT:
            raise ValueError(f"{label}: latitude {lat} beyond LAT_LIMIT {LAT_LIMIT}")
        self.lat, self.lon, self.label = lat, lon, label
        self.calls = 0

    def asc(self, jd):
        """True Ascendant longitude (deg) from ascmc[0]."""
        self.calls += 1
        return swe.houses_ex(jd, self.lat, self.lon, HOUSE_SYSTEM, FLAGS)[1][0]

    def level(self, k):
        """Exact longitude of pada boundary k (multiply first, then wrap)."""
        return ((k % GRID) * 360.0) / GRID

    def cell(self, jd):
        return int(math.floor((self.asc(jd) % 360.0) * GRID / 360.0)) % GRID


# ------------------------------------------------------------- solver
def solve_boundary(site, k, seed, pace):
    """Find t with asc(t) == level(k). f = wrap180(asc - level) is negative before
    the boundary and >= 0 after it (forward-only motion).
    Newton with estimated pace until bracketed, then Illinois. Residual-based stop."""
    L = site.level(k)
    F = lambda t: wrap180(site.asc(t) - L)
    tol_time = TIME_TOL_SEC / 86400.0
    tol_deg = max(1e-9, tol_time * pace)

    t = seed
    f = F(t)
    a = b = fa = fb = None
    side = 0
    prev = None
    for _ in range(MAX_ITER):
        if abs(f) <= tol_deg:
            return t
        if f < 0:
            if side == -1 and fb is not None:
                fb *= 0.5                                   # Illinois
            a, fa, side = t, f, -1
        else:
            if side == 1 and fa is not None:
                fa *= 0.5
            b, fb, side = t, f, 1
        if a is not None and b is not None:
            if b - a <= tol_time:
                return 0.5 * (a + b)
            tn = (a * fb - b * fa) / (fb - fa)
            if not (a < tn < b):
                tn = 0.5 * (a + b)
        else:
            slope = pace
            if prev is not None and t != prev[0]:
                s = (f - prev[1]) / (t - prev[0])
                if 0.2 * pace < s < 5.0 * pace:
                    slope = s
            step = -f / slope
            step = max(-STEP_CLAMP, min(STEP_CLAMP, step))
            tn = t + step
        prev = (t, f)
        t, f = tn, F(tn)
    WARN.append(f"{site.label}: pada {k % GRID}: solver hit MAX_ITER, residual {f:.2e} deg")
    return t


def extrapolate(ts):
    """Lagrange extrapolation of the next root time from consecutive roots."""
    m = len(ts)
    base = ts[-1]
    off = [x - base for x in ts]
    total = 0.0
    for j in range(m):
        w = 1.0
        for i in range(m):
            if i != j:
                w *= (m - i) / (j - i)
        total += w * off[j]
    return base + total


# ------------------------------------------------------------- sequence
def build_site(site, t_lo, t_hi):
    """Roots [(k, jd)] for every pada boundary in [t_lo, t_hi], strictly sequential."""
    dt = 60.0 / 86400.0
    L0 = site.asc(t_lo) % 360.0
    L1 = site.asc(t_lo + dt) % 360.0
    pace = wrap180(L1 - L0) / dt                            # deg/day
    if pace <= 0:
        raise RuntimeError(f"{site.label}: Ascendant not moving forward -- latitude too high?")
    k = int(math.floor(L0 * GRID / 360.0)) + 1              # first boundary above start
    dist = (site.level(k) - L0) % 360.0
    seed = t_lo + dist / pace

    roots = []
    while True:
        t = solve_boundary(site, k, seed, pace)
        if roots and t <= roots[-1][1]:
            raise RuntimeError(f"{site.label}: non-increasing root at pada {k % GRID}")
        roots.append((k, t))
        if t > t_hi:
            break
        k += 1
        ts = [r[1] for r in roots[-EXTRAP_POINTS:]]
        if len(ts) >= 2:
            seed = extrapolate(ts)
            pace = STEP_DEG / max(seed - t, 1e-9)
        else:
            seed = t + STEP_DEG / pace
    return roots


# ------------------------------------------------- verification layers
def check(site, roots):
    """Chain check + sidereal-day check. Zero ephemeris calls."""
    for (k0, t0), (k1, t1) in zip(roots[:-1], roots[1:]):
        assert k1 == k0 + 1, f"{site.label}: pada index jump {k0}->{k1}: event skipped"
        assert t1 > t0, f"{site.label}: times not increasing at JD {t0:.6f}"
    for i in range(GRID, len(roots)):
        dt = roots[i][1] - roots[i - GRID][1]
        assert abs(dt - SIDEREAL_DAY) <= SIDEREAL_DAY_TOL, (
            f"{site.label}: pada {roots[i][0] % GRID} repeats after {dt:.6f} d, "
            f"expected {SIDEREAL_DAY}: root wrong or missing")


def audit(site, roots, lo, hi):
    """Independent proof against the real ephemeris. Read true pada every AUDIT_STEP
    days; replay events in between; the replay must land on the true pada."""
    if AUDIT_STEP <= 0:
        return
    assert AUDIT_STEP <= 0.4, "AUDIT_STEP must be under half a revolution"
    jds = [r[1] for r in roots]
    prev_t = prev_c = None
    t = lo
    while t < hi:
        for _ in range(3):                                  # step off exact event times
            i = bisect.bisect_left(jds, t - 1e-5)
            if i < len(jds) and jds[i] < t + 1e-5:
                t += 3e-5
        c_true = site.cell(t)
        if prev_t is not None:
            cur = prev_c
            for k, tj in roots[bisect.bisect_right(jds, prev_t):bisect.bisect_right(jds, t)]:
                assert k % GRID == (cur + 1) % GRID, (
                    f"audit {site.label}: event at JD {tj:.6f} does not follow pada {cur}")
                cur = k % GRID
            assert cur == c_true, (f"audit {site.label}: replay says pada {cur} but ephemeris says "
                                   f"{c_true} at JD {t:.6f} (window {prev_t:.6f}..{t:.6f}): event missed")
        prev_t, prev_c = t, c_true
        t += AUDIT_STEP


def audit_events(site, roots, lo, hi):
    """Optional: true pada just before and just after every event."""
    if not AUDIT_EACH_EVENT:
        return
    eps = EVENT_EPS_SEC / 86400.0
    for k, t in roots:
        if lo <= t < hi:
            assert site.cell(t - eps) == (k - 1) % GRID and site.cell(t + eps) == k % GRID, (
                f"event audit {site.label}: root at JD {t:.8f} not within {EVENT_EPS_SEC}s of true boundary")


# ---------------------------------------------------------- tables
def site_rows(site, roots, lo, hi):
    rows = []
    for k, jd in roots:
        if not lo <= jd < hi:
            continue
        if SHOW_RASHI and k % 9 == 0:
            rows.append((jd, site.label, "Rashi", rashi_label((k // 9) % 12)))
        if SHOW_NAKSHATRA and k % 4 == 0:
            rows.append((jd, site.label, "Nakshatra", nak_label((k // 4) % 27)))
        if SHOW_PADA:
            p = k % GRID
            rows.append((jd, site.label, "Pada", f"{NAKSHATRA[p // 4]}-{p % 4 + 1}"))
    return rows


def _order(rows):
    if SORT_BY == "time":
        rows.sort(key=lambda r: r[0])
    return rows


def show(rows, multi):
    show_loc = multi if SHOW_LOCATION_COLUMN is None else SHOW_LOCATION_COLUMN
    w = 18 + JD_DECIMALS - 8
    head = (f"{'location':<12}" if show_loc else "") + f"{'event':<11}{'entering':<28}"
    head += f"{'JD':>{w}}" if SHOW_JD else ""
    head += f"  {TZ_LABEL}" if SHOW_IST else ""
    print(head)
    for jd, loc, ev, label in rows:
        line = (f"{loc:<12}" if show_loc else "") + f"{ev:<11}{label:<28}"
        line += f"{jd:>{w}.{JD_DECIMALS}f}" if SHOW_JD else ""
        line += f"  {jd_to_local(jd)}" if SHOW_IST else ""
        print(line)


def save_csv(rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["location", "event", "entering", "JD", TZ_LABEL])
        for jd, loc, ev, label in rows:
            wr.writerow([loc, ev, label, f"{jd:.{JD_DECIMALS}f}", jd_to_local(jd)])


# --------------------------------------------------------------- main
if __name__ == "__main__":
    s_jd = START_JD if START_JD is not None else swe.julday(*START_DATE)
    if START_JD is None and START_DATE_IS_LOCAL:
        s_jd -= TZ_OFFSET_HOURS / 24.0
    lo, hi = s_jd, s_jd + DAYS
    init_swe()

    sites = [Site(*loc) for loc in LOCATIONS]
    all_rows, total_calls, audit_calls = [], 0, 0
    t0 = time.perf_counter()
    solved = []
    for site in sites:
        roots = build_site(site, lo - PAD, hi + PAD)
        solved.append((site, roots))
        total_calls += site.calls
    elapsed = time.perf_counter() - t0

    if WARN:
        msg = f"{len(WARN)} warning(s); first: {WARN[0]}"
        if STRICT:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    if RUN_CHECK:
        for site, roots in solved:
            check(site, roots)
            before = site.calls
            audit(site, roots, lo, hi)
            audit_events(site, roots, lo, hi)
            audit_calls += site.calls - before
        print(f"Verification passed: chain + sidereal-day check + ephemeris audit "
              f"({audit_calls:,} audit calls)")

    for site, roots in solved:
        all_rows += _order(site_rows(site, roots, lo, hi))
    _order(all_rows)

    if PRINT_TABLE:
        show(all_rows, len(sites) > 1)
    if SAVE_CSV:
        save_csv(all_rows, CSV_FILE)
    if PRINT_TIME:
        print(f"Execution time : {elapsed:.3f} s (excl. audit)")
    if PRINT_CALLS:
        n_ev = sum(1 for r in all_rows if r[2] == "Pada") or 1
        print(f"Ephemeris calls: {total_calls:,}"
              + (f"  (~{total_calls / n_ev:.1f} per pada event)" if SHOW_PADA else ""))