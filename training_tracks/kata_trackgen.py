#!/usr/bin/env python
"""Kata curriculum track generator for F1TENTH.

Generates a graded set of closed circuits, ordered like the Shotokan kata syllabus
(Taikyoku Shodan -> Heian Shodan..Godan -> Tekki Shodan -> Bassai Dai). Every stage
introduces exactly one new driving competence, the same way each kata adds one new
technique on top of the previous ones:

  stage  kata               new competence                          form factor
  -----  -----------------  --------------------------------------  ----------------------------------
   1     Taikyoku Shodan    accelerate, hold lane, one gentle curve  stadium, wide, R>=6 m, one turn side
   2     Taikyoku Nidan     first direction change                   stadium + one S-shift, tighter R
   3     Taikyoku Sandan    chained direction changes at speed       stadium + chicane / two S-shifts
   4     Heian Shodan       ~90 deg corners, uneven straights        convex rounded polygon (3-5 corners)
   5     Heian Nidan        turning the other way (reflex corner)    concave "kidney" polygon
   6     Heian Sandan       S-sequences, alternating corners         polygon with 2-3 reflex corners
   7     Heian Yondan       hairpin: hard braking + 180 deg           polygon + one hairpin finger
   8     Heian Godan        combination at speed                     long straight + hairpins + S
   9     Tekki Shodan       precision in a narrow corridor           narrow slalom track
  10     Bassai Dai         everything, unknown layout               random circuit, hairpins, narrow

Difficulty knobs that decrease monotonically over the stages: track width, minimum corner
radius, straight fraction. Knobs that increase: number of direction changes, track length,
curvature per metre.

Every track is emitted in the exact layout ``f1tenth_gym`` expects under ``maps/``::

    maps/<name>/<name>_map.png          occupancy image (white = drivable)
    maps/<name>/<name>_map.yaml         resolution / origin metadata
    maps/<name>/<name>_centerline.csv   x_m, y_m, w_tr_right_m, w_tr_left_m

plus a manifest ``maps/kata_curriculum.yaml`` (stage -> tracks, form factors, metrics) that
the curriculum wrapper reads, a metrics CSV and a contact-sheet gallery PNG.

Geometry engine: every track is a polygon with a per-corner fillet radius. Families only
differ in how they build the polygon (stadium, S-shift, chicane, convex/concave polygon,
hairpin "finger", slalom, random polar polygon). The fillet engine turns the polygon into a
C1 centerline made of straights and circular arcs, so min corner radius is controlled
analytically, not by smoothing. A candidate is rejected unless:

  * the centerline is a simple closed curve,
  * the corridor (centerline buffered by width/2 + wall margin) has exactly one hole and
    no self-overlap (area == length * total width within tolerance),
  * every fillet fits between its neighbours and respects the stage's radius floor,
  * total length and bounding box fall in the stage's range.

Usage::

    python training_tracks/kata_trackgen.py                 # 10 stages x (10 train + 1 eval)
    python training_tracks/kata_trackgen.py --per-stage 10 --eval-per-stage 1 --seed 2026
    python training_tracks/kata_trackgen.py --only-gallery  # re-render gallery from manifest
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import shapely.geometry as shp
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MAPS_DIR = REPO_ROOT / "maps"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"

RESOLUTION = 0.05          # m / pixel (same as oval_small)
WALL_MARGIN = 0.6          # minimum wall thickness between two corridor sections (m)
HAIRPIN_WALL = 0.8         # wall thickness between the two legs of a hairpin (m)
IMG_MARGIN = 2.0           # free border around the corridor bounding box (m)
ARC_DS = 0.10              # sampling step along arcs (m)
CSV_DS = 0.50              # centerline waypoint spacing (m)
MAX_BBOX = 70.0            # reject anything larger than this (m)
LAP_GATE_HALF_WIDTH = 2.0  # F110Env._check_done lateral tolerance -> keep width <= 4.4 m


# --------------------------------------------------------------------------------------
# Stage definitions (form factors)
# --------------------------------------------------------------------------------------
@dataclass
class Stage:
    stage: int
    kata: str
    slug: str
    family: str
    competence: str
    width: tuple[float, float]        # corridor width range (m)
    rmin: tuple[float, float]         # corner radius floor range (m)  -> sampled per track
    length: tuple[float, float]       # total centerline length range (m)
    max_speed: float                  # curriculum speed cap for this stage (m/s)
    params: dict = field(default_factory=dict)


STAGES: list[Stage] = [
    Stage(1, "Taikyoku Shodan", "taikyoku_shodan", "stadium",
          "acelerar, mantener carril, una curva suave (un solo sentido de giro)",
          width=(4.0, 4.4), rmin=(6.0, 9.0), length=(55, 80), max_speed=5.0,
          params=dict(r_frac=(0.85, 1.0))),
    Stage(2, "Taikyoku Nidan", "taikyoku_nidan", "stadium_s",
          "primer cambio de sentido: una S suave en la recta",
          width=(3.8, 4.2), rmin=(4.5, 6.0), length=(70, 100), max_speed=5.0,
          params=dict(n_shifts=1, shift=(1.5, 3.0), shift_len=(7.0, 10.0))),
    Stage(3, "Taikyoku Sandan", "taikyoku_sandan", "stadium_chicane",
          "encadenar cambios de sentido a velocidad: chicane o doble S",
          width=(3.6, 4.0), rmin=(4.0, 5.5), length=(80, 115), max_speed=6.0,
          params=dict(shift=(2.0, 3.5), shift_len=(6.0, 9.0))),
    Stage(4, "Heian Shodan", "heian_shodan", "convex_polygon",
          "curvas de ~90 grados con rectas desiguales: frenar antes del vertice",
          width=(3.4, 3.8), rmin=(3.0, 4.5), length=(70, 110), max_speed=6.0,
          params=dict(n_vertices=(3, 5), aspect=(1.0, 1.6), r_mult=(1.0, 1.6))),
    Stage(5, "Heian Nidan", "heian_nidan", "concave_polygon",
          "girar hacia el otro lado: una curva concava (rinon)",
          width=(3.2, 3.6), rmin=(2.8, 4.0), length=(80, 120), max_speed=7.0,
          params=dict(n_vertices=(4, 5), n_reflex=1, aspect=(1.0, 1.6), r_mult=(1.0, 1.6),
                      push=(0.12, 0.28))),
    Stage(6, "Heian Sandan", "heian_sandan", "concave_polygon",
          "secuencias en S: curvas alternas de distinto signo",
          width=(3.0, 3.4), rmin=(2.5, 3.5), length=(90, 130), max_speed=7.0,
          params=dict(n_vertices=(6, 7), n_reflex=(2, 3), aspect=(1.0, 1.7), r_mult=(1.0, 1.8),
                      push=(0.1, 0.22))),
    Stage(7, "Heian Yondan", "heian_yondan", "hairpin_polygon",
          "horquilla: frenada fuerte y giro de 180 grados",
          width=(3.0, 3.2), rmin=(2.2, 3.0), length=(90, 140), max_speed=8.0,
          params=dict(n_vertices=(4, 5), n_reflex=(0, 1), n_fingers=1, aspect=(1.3, 1.8),
                      r_mult=(1.0, 1.6), finger_depth=(6.0, 10.0), push=(0.1, 0.22))),
    Stage(8, "Heian Godan", "heian_godan", "hairpin_polygon",
          "combinacion: recta larga, horquillas y S a velocidad",
          width=(2.8, 3.2), rmin=(2.0, 2.8), length=(100, 150), max_speed=8.0,
          params=dict(n_vertices=(5, 6), n_reflex=1, n_fingers=(1, 2), aspect=(1.6, 2.2),
                      r_mult=(1.0, 1.8), finger_depth=(6.0, 11.0), push=(0.1, 0.22))),
    Stage(9, "Tekki Shodan", "tekki_shodan", "slalom",
          "precision lateral en pasillo estrecho: eslalon",
          width=(2.4, 2.8), rmin=(2.4, 3.2), length=(90, 140), max_speed=8.0,
          params=dict(n_shifts=(3, 4), shift=(1.6, 2.6), shift_len=(5.0, 7.0))),
    Stage(10, "Bassai Dai", "bassai_dai", "random_circuit",
          "asaltar la fortaleza: circuito aleatorio completo con horquillas y estrechamientos",
          width=(2.6, 3.2), rmin=(1.8, 2.6), length=(140, 220), max_speed=8.0,
          params=dict(n_vertices=(8, 11), n_reflex=(2, 3), n_fingers=(1, 2),
                      radial=(0.7, 1.0), r_mult=(1.0, 2.5), finger_depth=(5.0, 9.0),
                      push=(0.08, 0.18))),
]


def _u(rng: np.random.Generator, rng_or_val):
    """Uniform sample from a (lo, hi) tuple, or pass a scalar through."""
    if isinstance(rng_or_val, (tuple, list)):
        lo, hi = rng_or_val
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(lo, hi + 1))
        return float(rng.uniform(lo, hi))
    return rng_or_val


# --------------------------------------------------------------------------------------
# Fillet engine: polygon + per-corner radius -> centerline of straights and arcs
# --------------------------------------------------------------------------------------
class FilletError(ValueError):
    pass


def fillet_polygon(verts: np.ndarray, radii: np.ndarray, r_floor):
    """Turn a closed polygon into a C1 loop of straights and circular arcs.

    Radii that do not fit on an edge are shrunk proportionally (both corners of that
    edge) until every edge accommodates its two tangent lengths. Returns
    (points (N,2), straights [(start_idx, end_idx, length)], min_radius).
    Raises FilletError if any corner would drop below ``r_floor`` (scalar or per-corner
    array; hairpin tips carry their own, smaller floor) or is too sharp.
    """
    n = len(verts)
    floors = np.broadcast_to(np.asarray(r_floor, dtype=float), (n,))
    u1s, u2s, thetas, signs, edge_len = [], [], [], [], []
    for i in range(n):
        P, V, N = verts[i - 1], verts[i], verts[(i + 1) % n]
        u1, u2 = P - V, N - V
        l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if l1 < 1e-6 or l2 < 1e-6:
            raise FilletError("degenerate edge")
        u1, u2 = u1 / l1, u2 / l2
        cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = math.acos(cos_t)                      # interior angle at V
        if theta < math.radians(15) or theta > math.radians(174):
            raise FilletError("corner too sharp or too flat")
        u1s.append(u1); u2s.append(u2); thetas.append(theta)
        signs.append(1.0 if (u1[0] * u2[1] - u1[1] * u2[0]) < 0 else -1.0)  # CCW turn -> +1
        edge_len.append(l2)                           # edge i -> i+1
    r = np.asarray(radii, dtype=float).copy()
    tan_half = np.array([math.tan(t / 2) for t in thetas])
    for _ in range(12):
        d = r / tan_half
        changed = False
        for i in range(n):
            j = (i + 1) % n
            need = d[i] + d[j]
            if need > edge_len[i] * (1 + 1e-6):
                f = edge_len[i] / need
                r[i] *= f
                r[j] *= f
                changed = True
        if not changed:
            break
    else:
        raise FilletError("fillets do not converge")
    if np.any(r < floors - 1e-3):
        i = int(np.argmax(floors - r))
        raise FilletError(f"fillet {i} would need r={r[i]:.2f} < floor {floors[i]:.2f}")
    d = r / tan_half

    pts, straights = [], []
    T2_prev = None
    tangents = []
    for i in range(n):
        V = verts[i]
        T1, T2 = V + u1s[i] * d[i], V + u2s[i] * d[i]
        bis = u1s[i] + u2s[i]
        bis /= np.linalg.norm(bis)
        C = V + bis * (r[i] / math.sin(thetas[i] / 2))
        tangents.append((T1, T2, C))
    for i in range(n):
        T2_prev = tangents[i - 1][1]
        T1, T2, C = tangents[i]
        seg = T1 - T2_prev
        seg_len = float(np.linalg.norm(seg))
        start_idx = len(pts)
        n_seg = max(1, int(seg_len / ARC_DS))
        for k in range(n_seg):
            pts.append(T2_prev + seg * (k / n_seg))
        straights.append((start_idx, len(pts), seg_len))
        a1 = math.atan2(T1[1] - C[1], T1[0] - C[0])
        sweep = (math.pi - thetas[i]) * signs[i]
        n_arc = max(2, int(abs(sweep) * r[i] / ARC_DS))
        for k in range(n_arc):
            a = a1 + sweep * (k / n_arc)
            pts.append(C + r[i] * np.array([math.cos(a), math.sin(a)]))
    return np.asarray(pts), straights, float(r.min())


def _loop_length(verts, radii, r_floor):
    pts, _, _ = fillet_polygon(np.asarray(verts, float), np.asarray(radii, float), r_floor)
    return shp.LineString(np.vstack([pts, pts[:1]])).length


def _fit_length(verts, radii, r_floor, target, iters=3):
    """Scale the polygon (radii fixed) so the filleted loop length matches ``target``."""
    verts = np.asarray(verts, float)
    for _ in range(iters):
        L = _loop_length(verts, radii, r_floor)
        verts = verts * (target / L)
    return verts


# --------------------------------------------------------------------------------------
# Polygon families
# --------------------------------------------------------------------------------------
def poly_stadium(rng, st: Stage, width):
    """Rounded rectangle; r_frac = 1 gives exact semicircles (a stadium)."""
    R = _u(rng, st.rmin)
    L_target = _u(rng, st.length)
    r_frac = _u(rng, st.params["r_frac"])
    r = R * r_frac
    H = 2 * R
    Ls = (L_target - 2 * math.pi * r - 2 * (H - 2 * r)) / 2
    if Ls < 2.0:
        raise FilletError("stadium too short")
    W = Ls + 2 * r
    verts = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]])
    radii = np.full(4, r)
    return verts, radii, np.full(4, r)


def _rect_with_shifts(rng, st, n_shifts_bottom, n_shifts_top, shift_rng, shift_len_rng,
                      r_floor, gentle_mult=1.6):
    """Rounded rectangle whose long edges carry lateral S-shifts.

    One shift = lane change and back: pa (on edge) -> pb (offset) -> pc -> pd (back on edge),
    i.e. four gentle corners; consecutive shifts alternate sides (slalom).
    """
    L_target = _u(rng, st.length)
    r_main = r_floor * _u(rng, (1.0, 1.25))
    H = 2 * r_main * _u(rng, (1.0, 1.15))
    W = (L_target - 2 * math.pi * r_main - 2 * (H - 2 * r_main)) / 2 + 2 * r_main
    # make sure the busiest edge can host its shifts at full size
    n_max = max(n_shifts_bottom, n_shifts_top)
    room = 2 * (2 * r_main + 0.5) + n_max * (shift_len_rng[1] + 0.5 * shift_len_rng[1] + 1.5)
    W = max(W, room)

    def edge_with_shifts(p0, p1, n_shifts, outward):
        out_v, out_r = [], []
        e = p1 - p0
        L = np.linalg.norm(e)
        e /= L
        nrm = np.array([-e[1], e[0]]) * outward
        margin0 = 2 * r_main + 0.5
        usable = L - 2 * margin0
        if n_shifts == 0:
            return out_v, out_r
        slot = usable / n_shifts
        for k in range(n_shifts):
            h = _u(rng, shift_len_rng) / 2
            a = _u(rng, shift_rng)
            P = max(3.0, 0.5 * _u(rng, shift_len_rng))
            need = 2 * h + P
            if need > slot - 1.0:
                f = (slot - 1.0) / need
                h *= f; P *= f; a *= f
                if h < 1.5:
                    raise FilletError("no room for shifts")
                need = 2 * h + P
            sgn = 1.0 if k % 2 == 0 else -1.0
            start = margin0 + k * slot + rng.uniform(0.0, max(0.0, slot - need - 0.5))
            pa = p0 + e * start
            pb = pa + e * h + nrm * (a * sgn)
            pc = pb + e * P
            pd = pc + e * h - nrm * (a * sgn)
            for v in (pa, pb, pc, pd):
                out_v.append(v)
                out_r.append(r_floor * gentle_mult)
        return out_v, out_r

    corners = [np.array([-W / 2, -H / 2]), np.array([W / 2, -H / 2]),
               np.array([W / 2, H / 2]), np.array([-W / 2, H / 2])]
    verts, radii = [], []
    verts.append(corners[0]); radii.append(r_main)
    v, r = edge_with_shifts(corners[0], corners[1], n_shifts_bottom, outward=-1.0)
    verts += v; radii += r
    verts.append(corners[1]); radii.append(r_main)
    verts.append(corners[2]); radii.append(r_main)
    v, r = edge_with_shifts(corners[2], corners[3], n_shifts_top, outward=-1.0)
    verts += v; radii += r
    verts.append(corners[3]); radii.append(r_main)
    verts = np.asarray(verts)
    radii = np.asarray(radii)
    verts = _fit_length(verts, radii, r_floor, L_target)
    return verts, radii, np.full(len(radii), r_floor)


def poly_stadium_s(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    return _rect_with_shifts(rng, st, 1, 0, st.params["shift"], st.params["shift_len"], r_floor)


def poly_stadium_chicane(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    layout = rng.integers(0, 2)
    nb, nt = (2, 0) if layout == 0 else (1, 1)
    return _rect_with_shifts(rng, st, nb, nt, st.params["shift"], st.params["shift_len"],
                             r_floor, gentle_mult=1.3)


def _polar_polygon(rng, n, a, b, radial=(0.85, 1.0), jitter=0.35):
    """n vertices around an ellipse (a, b) with angular jitter and radial noise."""
    base = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ang = base + rng.uniform(-jitter, jitter, size=n) * (2 * math.pi / n)
    rad = rng.uniform(radial[0], radial[1], size=n)
    verts = np.stack([a * rad * np.cos(ang), b * rad * np.sin(ang)], axis=1)
    return verts


def _push_reflex(rng, verts, n_reflex, push_rng):
    """Push n non-adjacent vertices toward the centroid to create reflex (concave) corners."""
    n = len(verts)
    if n_reflex <= 0:
        return verts
    cands = list(range(n))
    rng.shuffle(cands)
    chosen = []
    for c in cands:
        if all(abs((c - o) % n) > 1 and abs((o - c) % n) > 1 for o in chosen):
            chosen.append(c)
        if len(chosen) == n_reflex:
            break
    centroid = verts.mean(axis=0)
    out = verts.copy()
    for c in chosen:
        mid = 0.5 * (verts[c - 1] + verts[(c + 1) % n])
        # move past the chord between neighbours so the corner becomes reflex
        depth = _u(rng, push_rng) * np.linalg.norm(verts[c] - centroid)
        direction = centroid - verts[c]
        direction /= np.linalg.norm(direction)
        new_pt = mid + direction * depth
        out[c] = new_pt
    return out


def _add_fingers(rng, verts, radii, n_fingers, width, depth_rng, r_floor):
    """Insert hairpin fingers on the longest edges: p1 -> p2 (out) -> p3 -> p4 (back).

    The two legs are ``gap`` apart (centerline to centerline) so a wall of
    ``gap - width`` remains between them; the tip is a semicircle of radius gap/2, which
    is deliberately below the stage radius floor (that is the hairpin) and gets its own
    floor entry.
    """
    gap = width + HAIRPIN_WALL
    r_tip = gap / 2
    r_base = r_floor * 1.1
    verts = [np.asarray(v, dtype=float) for v in verts]
    radii = list(radii)
    floors = [r_floor] * len(radii)
    for _ in range(n_fingers):
        n = len(verts)
        lengths = [np.linalg.norm(verts[(i + 1) % n] - verts[i]) for i in range(n)]
        order = np.argsort(lengths)[::-1]
        placed = False
        for i in order[:3]:
            A, B = verts[i], verts[(i + 1) % n]
            L = lengths[i]
            need = gap + 2 * r_base + 1.0           # finger footprint along the edge
            lo, hi = 1.5 * r_floor, L - need - 1.5 * r_floor
            if hi <= lo:
                continue
            e = (B - A) / L
            nrm = np.array([e[1], -e[0]])
            centroid = np.mean(np.asarray(verts), axis=0)
            if np.dot(nrm, (A + B) / 2 - centroid) < 0:
                nrm = -nrm
            D = max(_u(rng, depth_rng), r_base + r_tip + 1.0)
            t0 = rng.uniform(lo, hi)
            p1 = A + e * t0
            p2 = p1 + nrm * D
            p3 = p2 + e * gap
            p4 = p1 + e * gap
            verts = verts[: i + 1] + [p1, p2, p3, p4] + verts[i + 1:]
            radii = radii[: i + 1] + [r_base, r_tip, r_tip, r_base] + radii[i + 1:]
            floors = floors[: i + 1] + [r_floor, r_tip - 1e-3, r_tip - 1e-3, r_floor] + floors[i + 1:]
            placed = True
            break
        if not placed:
            raise FilletError("no edge long enough for a hairpin finger")
    return np.asarray(verts), np.asarray(radii), np.asarray(floors)


def _ccw(verts):
    x, y = verts[:, 0], verts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return verts if area > 0 else verts[::-1]


def poly_convex(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    n = _u(rng, st.params["n_vertices"])
    L_target = _u(rng, st.length)
    aspect = _u(rng, st.params["aspect"])
    # perimeter of a polygon inscribed in ellipse ~ 0.9 * pi * (a + b)
    a_plus_b = L_target / (0.9 * math.pi)
    b = a_plus_b / (1 + aspect)
    a = aspect * b
    verts = _ccw(_polar_polygon(rng, n, a, b, radial=(0.85, 1.0), jitter=0.3))
    radii = np.array([r_floor * _u(rng, st.params["r_mult"]) for _ in range(n)])
    verts = _fit_length(verts, radii, r_floor, L_target)
    return verts, radii, np.full(n, r_floor)


def poly_concave(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    n = _u(rng, st.params["n_vertices"])
    n_reflex = _u(rng, st.params["n_reflex"])
    L_target = _u(rng, st.length)
    aspect = _u(rng, st.params["aspect"])
    a_plus_b = L_target / (0.95 * math.pi)
    b = a_plus_b / (1 + aspect)
    a = aspect * b
    verts = _ccw(_polar_polygon(rng, n, a, b, radial=(0.9, 1.0), jitter=0.25))
    verts = _push_reflex(rng, verts, n_reflex, st.params["push"])
    radii = np.array([r_floor * _u(rng, st.params["r_mult"]) for _ in range(len(verts))])
    verts = _fit_length(verts, radii, r_floor, L_target)
    return verts, radii, np.full(len(radii), r_floor)


def poly_hairpin(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    n = _u(rng, st.params["n_vertices"])
    n_reflex = _u(rng, st.params["n_reflex"])
    n_fingers = _u(rng, st.params["n_fingers"])
    L_target = _u(rng, st.length)
    aspect = _u(rng, st.params["aspect"])
    depth_mean = float(np.mean(st.params["finger_depth"]))
    base_L = L_target - n_fingers * 2 * depth_mean
    a_plus_b = max(base_L, 50.0) / (0.95 * math.pi)
    b = a_plus_b / (1 + aspect)
    a = aspect * b
    verts = _ccw(_polar_polygon(rng, n, a, b, radial=(0.9, 1.0), jitter=0.2))
    verts = _push_reflex(rng, verts, n_reflex, st.params["push"])
    radii = np.array([r_floor * _u(rng, st.params["r_mult"]) for _ in range(len(verts))])
    verts = _fit_length(verts, radii, r_floor, max(base_L, 50.0))
    verts, radii, floors = _add_fingers(rng, verts, radii, n_fingers, width, st.params["finger_depth"], r_floor)
    return verts, radii, floors


def poly_slalom(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    n_shifts = _u(rng, st.params["n_shifts"])
    layout = rng.integers(0, 2)
    nb, nt = (n_shifts, 0) if layout == 0 else (n_shifts - 1, 1)
    return _rect_with_shifts(rng, st, nb, nt, st.params["shift"], st.params["shift_len"],
                             r_floor, gentle_mult=1.15)


def poly_random(rng, st: Stage, width):
    r_floor = _u(rng, st.rmin)
    n = _u(rng, st.params["n_vertices"])
    n_reflex = _u(rng, st.params["n_reflex"])
    n_fingers = _u(rng, st.params["n_fingers"])
    L_target = _u(rng, st.length)
    depth_mean = float(np.mean(st.params["finger_depth"]))
    base_L = L_target - n_fingers * 2 * depth_mean
    aspect = _u(rng, (1.0, 1.6))
    # random polar polygon perimeter is longer than the ellipse's: shrink accordingly
    a_plus_b = base_L / (1.15 * math.pi)
    b = a_plus_b / (1 + aspect)
    a = aspect * b
    verts = _ccw(_polar_polygon(rng, n, a, b, radial=st.params["radial"], jitter=0.3))
    verts = _push_reflex(rng, verts, n_reflex, st.params["push"])
    radii = np.array([r_floor * _u(rng, st.params["r_mult"]) for _ in range(len(verts))])
    verts = _fit_length(verts, radii, r_floor, max(base_L, 60.0))
    verts, radii, floors = _add_fingers(rng, verts, radii, n_fingers, width, st.params["finger_depth"], r_floor)
    return verts, radii, floors


FAMILIES = {
    "stadium": poly_stadium,
    "stadium_s": poly_stadium_s,
    "stadium_chicane": poly_stadium_chicane,
    "convex_polygon": poly_convex,
    "concave_polygon": poly_concave,
    "hairpin_polygon": poly_hairpin,
    "slalom": poly_slalom,
    "random_circuit": poly_random,
}


# --------------------------------------------------------------------------------------
# Validation and metrics
# --------------------------------------------------------------------------------------
def _resample_closed(pts: np.ndarray, ds: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(8, int(L / ds))
    si = np.linspace(0, L, n, endpoint=False)
    x = np.interp(si, s, closed[:, 0])
    y = np.interp(si, s, closed[:, 1])
    return np.stack([x, y], axis=1), L


def _curvature(pts: np.ndarray, ds: float) -> np.ndarray:
    """Signed discrete curvature of a uniformly sampled closed curve."""
    dx = (np.roll(pts[:, 0], -1) - np.roll(pts[:, 0], 1)) / (2 * ds)
    dy = (np.roll(pts[:, 1], -1) - np.roll(pts[:, 1], 1)) / (2 * ds)
    ddx = (np.roll(pts[:, 0], -1) - 2 * pts[:, 0] + np.roll(pts[:, 0], 1)) / ds ** 2
    ddy = (np.roll(pts[:, 1], -1) - 2 * pts[:, 1] + np.roll(pts[:, 1], 1)) / ds ** 2
    denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    return (dx * ddy - dy * ddx) / denom


def validate_and_measure(pts: np.ndarray, width: float, min_r_analytic: float, st: Stage) -> dict:
    """Raise FilletError if the corridor is not drivable; otherwise return metrics."""
    ring = shp.LinearRing(pts)
    if not ring.is_valid or not ring.is_simple:
        raise FilletError("centerline self-intersects")
    closed = shp.LineString(np.vstack([pts, pts[:1]]))
    L = closed.length
    if not (st.length[0] * 0.9 <= L <= st.length[1] * 1.1):
        raise FilletError(f"length {L:.1f} outside range")
    minx, miny, maxx, maxy = closed.bounds
    if max(maxx - minx, maxy - miny) > MAX_BBOX:
        raise FilletError("bounding box too large")
    half = width / 2 + WALL_MARGIN / 2
    if min_r_analytic < half + 0.04:
        raise FilletError("inner wall radius would be negative")
    corr = closed.buffer(half, join_style=1, cap_style=1, quad_segs=16)
    if corr.geom_type != "Polygon" or len(corr.interiors) != 1:
        raise FilletError("corridor pinches itself (hole count != 1)")
    expected_area = L * 2 * half
    if abs(corr.area - expected_area) / expected_area > 0.02:
        raise FilletError("corridor overlaps itself (area mismatch)")

    res_pts, _ = _resample_closed(pts, 0.25)
    k = _curvature(res_pts, 0.25)
    k_s = np.convolve(np.pad(k, 4, mode="wrap"), np.ones(9) / 9, mode="valid")
    abs_k = np.abs(k_s)
    straight_frac = float(np.mean(abs_k < 0.03))
    signs = np.sign(np.where(abs_k < 0.05, 0.0, k_s))
    nz = signs[signs != 0]
    dir_changes = int(np.sum(nz != np.roll(nz, 1))) if len(nz) else 0
    metrics = dict(
        length_m=round(float(L), 2),
        width_m=round(float(width), 2),
        min_radius_m=round(float(min_r_analytic), 2),
        mean_abs_curvature=round(float(np.mean(abs_k)), 4),
        turn_integral_rad=round(float(np.sum(abs_k) * 0.25), 2),   # total |heading change|
        straight_fraction=round(straight_frac, 3),
        direction_changes=dir_changes,
        bbox_m=[round(maxx - minx, 1), round(maxy - miny, 1)],
    )
    # composite difficulty index (0 easy .. ~1 hard), documented in the manifest
    metrics["difficulty_index"] = round(
        0.35 * min(1.0, 2.0 / metrics["min_radius_m"])
        + 0.25 * min(1.0, 2.2 / metrics["width_m"])
        + 0.25 * min(1.0, metrics["turn_integral_rad"] / (2 * math.pi) / 6.0)
        + 0.15 * min(1.0, metrics["direction_changes"] / 12.0), 3)
    return metrics


# --------------------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------------------
def _rotate_to_start(pts: np.ndarray, straights, rng) -> np.ndarray:
    """Start the loop at the midpoint of the longest straight; random driving direction."""
    start_idx, end_idx, _ = max(straights, key=lambda s: s[2])
    mid = (start_idx + end_idx) // 2
    pts = np.roll(pts, -mid, axis=0)
    if rng.uniform() < 0.5:
        pts = np.vstack([pts[:1], pts[1:][::-1]])   # keep the start point, reverse direction
    return pts - pts.mean(axis=0)


def write_track(name: str, pts: np.ndarray, width: float, maps_dir: pathlib.Path) -> dict:
    """Write <name>_map.png / .yaml / _centerline.csv in f1tenth_gym layout."""
    out = maps_dir / name
    out.mkdir(parents=True, exist_ok=True)
    closed = shp.LineString(np.vstack([pts, pts[:1]]))
    corr = closed.buffer(width / 2, join_style=1, cap_style=1, quad_segs=16)
    minx, miny, maxx, maxy = corr.bounds
    ox, oy = minx - IMG_MARGIN, miny - IMG_MARGIN
    W = int(math.ceil((maxx - minx + 2 * IMG_MARGIN) / RESOLUTION))
    H = int(math.ceil((maxy - miny + 2 * IMG_MARGIN) / RESOLUTION))

    def to_px(coords):
        return [((x - ox) / RESOLUTION, (y - oy) / RESOLUTION) for x, y in coords]

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(to_px(corr.exterior.coords), fill=255)
    for hole in corr.interiors:
        draw.polygon(to_px(hole.coords), fill=0)
    arr = np.array(img)
    # array row == y index (world orientation); PNG row 0 must be the TOP (max y),
    # F110Env flips the image top-bottom on load.
    Image.fromarray(np.flipud(arr)).save(out / f"{name}_map.png")

    with open(out / f"{name}_map.yaml", "w") as f:
        f.write(f"image: {name}_map.png\n")
        f.write(f"resolution: {RESOLUTION:.6f}\n")
        f.write(f"origin: [{ox:.6f}, {oy:.6f}, 0.000000]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.45\n")
        f.write("free_thresh: 0.196\n")

    wps, _ = _resample_closed(pts, CSV_DS)
    with open(out / f"{name}_centerline.csv", "w", newline="") as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for x, y in wps:
            f.write(f"{x:.6f}, {y:.6f}, {width / 2:.6f}, {width / 2:.6f}\n")
    return dict(image_px=[W, H], origin=[round(ox, 3), round(oy, 3)], n_waypoints=int(len(wps)))


def generate_track(st: Stage, seed: int, name: str, maps_dir: pathlib.Path, max_tries: int = 400):
    """Sample candidates for a stage until one passes validation. Returns manifest entry."""
    for attempt in range(max_tries):
        rng = np.random.default_rng(seed * 1000 + attempt)
        try:
            width = _u(rng, st.width)
            width = min(width, 2 * LAP_GATE_HALF_WIDTH + 0.4)
            verts, radii, floors = FAMILIES[st.family](rng, st, width)
            floors = np.maximum(np.asarray(floors, float), width / 2 + WALL_MARGIN / 2 + 0.05)
            pts, straights, min_r = fillet_polygon(np.asarray(verts, float), np.asarray(radii, float), floors)
            metrics = validate_and_measure(pts, width, min_r, st)
        except FilletError:
            continue
        pts = _rotate_to_start(pts, straights, rng)
        info = write_track(name, pts, width, maps_dir)
        return dict(name=name, seed=int(seed), attempts=attempt + 1, **metrics, **info)
    raise RuntimeError(f"could not generate a valid track for stage {st.stage} (seed {seed})")


def render_gallery(manifest: dict, maps_dir: pathlib.Path, out_png: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = manifest["stages"]
    n_cols = max(len(s["train_tracks"]) + len(s["eval_tracks"]) for s in stages)
    fig, axes = plt.subplots(len(stages), n_cols, figsize=(1.6 * n_cols, 1.75 * len(stages)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(stages):
        names = [t["name"] for t in s["train_tracks"]] + [t["name"] for t in s["eval_tracks"]]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c >= len(names):
                ax.axis("off")
                continue
            name = names[c]
            img = Image.open(maps_dir / name / f"{name}_map.png")
            ax.imshow(img, cmap="gray", interpolation="nearest")
            is_eval = c >= len(s["train_tracks"])
            ax.set_title(("EVAL " if is_eval else "") + name.split("_")[-1], fontsize=6,
                         color="tab:red" if is_eval else "black")
            if c == 0:
                ax.set_ylabel(f"{s['stage']:02d} {s['kata']}\nw={s['width_m'][0]}-{s['width_m'][1]}m "
                              f"r>={s['rmin_m'][0]}m", fontsize=6)
    fig.suptitle("Kata curriculum tracks (white = drivable)", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", type=pathlib.Path, default=DEFAULT_MAPS_DIR)
    ap.add_argument("--docs-dir", type=pathlib.Path, default=DEFAULT_DOCS_DIR)
    ap.add_argument("--per-stage", type=int, default=10, help="training tracks per stage")
    ap.add_argument("--eval-per-stage", type=int, default=1, help="held-out tracks per stage")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--stages", type=str, default="all", help="e.g. 1,2,3 or all")
    ap.add_argument("--only-gallery", action="store_true")
    args = ap.parse_args()

    manifest_path = args.maps_dir / "kata_curriculum.yaml"
    if args.only_gallery:
        manifest = yaml.safe_load(open(manifest_path))
        render_gallery(manifest, args.maps_dir, args.docs_dir / "kata_tracks_gallery.png")
        return

    wanted = STAGES if args.stages == "all" else [s for s in STAGES if s.stage in {int(x) for x in args.stages.split(",")}]
    manifest = dict(
        generator="training_tracks/kata_trackgen.py",
        seed=args.seed,
        resolution_m_per_px=RESOLUTION,
        wall_margin_m=WALL_MARGIN,
        difficulty_index="0.35*min(1,2/rmin) + 0.25*min(1,2.2/width) + 0.25*min(1,turn_integral/(2pi)/6) + 0.15*min(1,dir_changes/12)",
        stages=[],
    )
    rows = []
    for st in wanted:
        entry = dict(stage=st.stage, kata=st.kata, slug=st.slug, family=st.family, competence=st.competence,
                     width_m=list(st.width), rmin_m=list(st.rmin), length_m=list(st.length),
                     max_speed=st.max_speed, train_tracks=[], eval_tracks=[])
        for k in range(args.per_stage):
            name = f"kata_{st.stage:02d}_{st.slug}_{k + 1:02d}"
            seed = args.seed * 100 + st.stage * 100 + k
            t = generate_track(st, seed, name, args.maps_dir)
            entry["train_tracks"].append(t)
            rows.append(dict(stage=st.stage, kata=st.kata, split="train", **t))
            print(f"[ok] {name:38s} L={t['length_m']:6.1f} m  w={t['width_m']:.2f}  rmin={t['min_radius_m']:.2f}"
                  f"  turns={t['direction_changes']:2d}  D={t['difficulty_index']:.2f}  ({t['attempts']} tries)")
        for k in range(args.eval_per_stage):
            name = f"kata_{st.stage:02d}_{st.slug}_e{k + 1}"
            seed = args.seed * 100 + st.stage * 100 + 50 + k
            t = generate_track(st, seed, name, args.maps_dir)
            entry["eval_tracks"].append(t)
            rows.append(dict(stage=st.stage, kata=st.kata, split="eval", **t))
            print(f"[ok] {name:38s} L={t['length_m']:6.1f} m  w={t['width_m']:.2f}  rmin={t['min_radius_m']:.2f}"
                  f"  turns={t['direction_changes']:2d}  D={t['difficulty_index']:.2f}  (eval)")
        manifest["stages"].append(entry)

    args.maps_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    with open(args.docs_dir / "kata_tracks_metrics.csv", "w", newline="") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    render_gallery(manifest, args.maps_dir, args.docs_dir / "kata_tracks_gallery.png")

    print("\nStage summary (mean over training tracks):")
    print(f"{'st':>2} {'kata':18s} {'len':>6} {'width':>5} {'rmin':>5} {'turns':>5} {'straight%':>9} {'D':>5}")
    for s in manifest["stages"]:
        tt = s["train_tracks"]
        m = lambda key: float(np.mean([t[key] for t in tt]))
        print(f"{s['stage']:>2} {s['kata']:18s} {m('length_m'):6.1f} {m('width_m'):5.2f} {m('min_radius_m'):5.2f} "
              f"{m('direction_changes'):5.1f} {100 * m('straight_fraction'):9.1f} {m('difficulty_index'):5.2f}")
    print(f"\nmanifest: {manifest_path}\ngallery:  {args.docs_dir / 'kata_tracks_gallery.png'}")


if __name__ == "__main__":
    sys.exit(main())
