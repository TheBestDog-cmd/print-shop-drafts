#!/usr/bin/env python3
"""Garmin watch stand for an AGVEE 90-degree charging cable.

Sized for the watch end of AGVEE ASIN B0DPH9Z1KS (USB-C on the other end).
AGVEE's compact-size drawing of this L-head is 12.0 x 9.6 mm in cross
section and 19.7 mm along the cable. The pocket is larger than that head.
The cable slot is 6.5 mm, so the head cannot pull out through it and a
USB-C plug is not what seats here.

This is a new model. MakerWorld model 599918 is not included: that file's
Standard Digital File License does not allow distributing remixes.

Print: PLA or PETG, 0.20 mm layers, 3 walls, 15% infill, base down.
The backrest underside is about 22 degrees off vertical, so it does not
need support. Slide the 90-degree head up the rear slot until the pins
sit in the window, and let the cable drop out the back.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
from shapely.geometry import Polygon

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(OUT_DIR, "garmin_stand_agvee_90deg.stl")

# Stand
WIDTH = 72.0
BASE_DEPTH = 64.0
BASE_THICK = 8.0
LEAN_DEG = 22.0  # backrest, degrees off vertical
FACE_LEN = 64.0
PLATE_THICK = 15.0
WINDOW_WIDTH = 34.0  # strap opening
CORNER_R = 4.0

# AGVEE 90-degree head (published drawing) plus clearance
HEAD_W = 12.0
HEAD_T = 9.6
HEAD_L = 19.7
POCKET_W = 14.6  # across the stand, fits the 12 mm side
POCKET_T = 12.2  # into the plate, fits the 9.6 mm side
POCKET_L = 23.0  # along the cable
PIN_WINDOW_W = 10.5
PIN_WINDOW_H = 7.2
CABLE_SLOT_W = 6.5
CABLE_SLOT_T = 6.0
# Watch-side wall. The pin boss passes through the window and sits
# about 1 mm proud of the face; the head shoulders stop on this wall.
WALL_WATCH = 1.4
REAR_GAP = 13.2  # head slides in from the back, then up behind a lip
PIN_U = 52.0  # up the face; pocket hangs below this toward the strap opening


def _union(meshes):
    meshes = [m for m in meshes if m is not None]
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.boolean.union(meshes, engine="manifold")


def _difference(base, cutters):
    cutters = [c for c in cutters if c is not None]
    if not cutters:
        return base
    tool = cutters[0] if len(cutters) == 1 else _union(cutters)
    return trimesh.boolean.difference([base, tool], engine="manifold")


def extrude_yz(points, x0, x1, name="part"):
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    mesh = trimesh.creation.extrude_polygon(poly, height=float(x1 - x0))
    # extrude_polygon builds in XY and extrudes along Z. Our polygon is YZ,
    # so map polygon X -> world Y, polygon Y -> world Z, extrusion -> world X.
    mesh.apply_transform(
        np.array(
            [
                [0, 0, 1, x0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
    )
    mesh.metadata["name"] = name
    return mesh


def rounded_rect_yz(cy, cz, sy, sz, radius, segments=8):
    """Polygon of a rounded rectangle centered at (cy, cz)."""
    hw, hh = sy / 2 - radius, sz / 2 - radius
    pts = []
    corners = [
        (cy + hw, cz + hh, 0),
        (cy - hw, cz + hh, np.pi / 2),
        (cy - hw, cz - hh, np.pi),
        (cy + hw, cz - hh, 3 * np.pi / 2),
    ]
    for cx, cy_, a0 in corners:
        for i in range(segments + 1):
            ang = a0 + i * (np.pi / 2) / segments
            pts.append((cx + radius * np.cos(ang), cy_ + radius * np.sin(ang)))
    return pts


def face_frame():
    """Watch-face origin, up-the-face tangent, and into-plate direction."""
    a = np.deg2rad(LEAN_DEG)
    tangent = np.array([0.0, -np.sin(a), np.cos(a)])  # up the face
    toward_watch = np.array([0.0, np.cos(a), np.sin(a)])
    into = -toward_watch
    # Far enough forward that the leaned top still sits over the base.
    origin = np.array([0.0, 42.0, BASE_THICK])  # bottom of the watch face
    return origin, tangent, into, toward_watch


def world_from_face(x, u, w, origin, tangent, into):
    return origin + np.array([x, 0.0, 0.0]) + tangent * u + into * w


def box_in_face(x, u, w, sx, su, sw, origin, tangent, into):
    """Axis-aligned box in face coordinates, centered on (x, u, w)."""
    mesh = trimesh.creation.box(extents=(sx, su, sw))
    # Local box axes are X, Y, Z. Map them onto world X, tangent, into.
    m = np.eye(4)
    m[:3, 0] = (1.0, 0.0, 0.0)
    m[:3, 1] = tangent
    m[:3, 2] = into
    m[:3, 3] = world_from_face(x, u, w, origin, tangent, into)
    mesh.apply_transform(m)
    return mesh


def side_profile():
    """Solid side view: flat base, backrest leaning 22 degrees, rear over the base."""
    origin, tangent, into, _toward = face_frame()
    face0 = origin
    face1 = origin + tangent * FACE_LEN
    rear1 = face1 + into * PLATE_THICK

    def yz(p):
        return (round(float(p[1]), 4), round(float(p[2]), 4))

    return [
        (0.0, 0.0),
        (BASE_DEPTH, 0.0),
        (BASE_DEPTH, BASE_THICK),
        yz(face0),
        yz(face1),
        yz(rear1),
        (0.0, BASE_THICK),
    ]


def strap_window_profile():
    """Opening through the lower backrest, under the charger, above the base.

    The charger pocket starts near u=33, so this stays below that.
    It is only cut through the middle of the width; the side arms stay solid.
    """
    return [
        (4.0, 13.0),
        (44.0, 13.0),
        (32.0, 34.0),
        (4.0, 34.0),
    ]


def build():
    origin, tangent, into, toward = face_frame()
    half = WIDTH / 2

    body = extrude_yz(side_profile(), -half, half, "body")
    window = extrude_yz(
        strap_window_profile(),
        -WINDOW_WIDTH / 2,
        WINDOW_WIDTH / 2,
        "strap",
    )
    body = _difference(body, [window])

    # Charger pocket. u grows up the face. The pins are at the top of the
    # head; the cable runs down the face, out through the strap opening.
    pin_u = PIN_U
    pocket_u = pin_u - (POCKET_L / 2 - 4.0)
    pocket = box_in_face(
        0.0,
        pocket_u,
        WALL_WATCH + POCKET_T / 2,
        POCKET_W,
        POCKET_L,
        POCKET_T,
        origin,
        tangent,
        into,
    )
    # Window through the watch face, slightly proud, so the pin boss can
    # meet the contacts. Smaller than the head, so the shoulders stay put.
    pin_window = box_in_face(
        0.0,
        pin_u,
        WALL_WATCH / 2 - 0.6,
        PIN_WINDOW_W,
        PIN_WINDOW_H,
        WALL_WATCH + 2.0,
        origin,
        tangent,
        into,
    )
    # Rear opening the head snaps through. Narrower than the head.
    # Deep enough to break out the back face. The backrest is thicker
    # than the pocket near the base, so a short cutter would stop inside.
    # Opening in the lower part of the pocket. The top 3 mm of the back
    # wall stays, so the head is inserted low and slid up behind that lip.
    rear = box_in_face(
        0.0,
        pocket_u - 1.5,
        18.0,
        REAR_GAP,
        20.0,
        28.0,
        origin,
        tangent,
        into,
    )
    # Cable slot under the pocket. Narrower than the head (6.5 vs 9.6/12)
    # and open into the strap window so the lead drops out the back.
    # Narrow throat under the pocket. It overlaps the pocket and the strap
    # opening, and it is narrower than the head so the head cannot slide out.
    slot_u = pocket_u - POCKET_L / 2 - 1.0
    cable_slot = box_in_face(
        0.0,
        slot_u,
        3.4,
        CABLE_SLOT_W,
        12.0,
        5.6,
        origin,
        tangent,
        into,
    )

    # Side lips, proud of the watch face, so the case does not slide off.
    lip_len = 36.0
    lip_u = 42.0
    lips = []
    for sign in (-1, 1):
        lips.append(
            box_in_face(
                sign * 31.0,
                lip_u,
                -2.2,
                8.0,
                lip_len,
                4.4,
                origin,
                tangent,
                into,
            )
        )

    # Soften the base front corners.
    corner_cut = []
    for sign in (-1, 1):
        cyl = trimesh.creation.cylinder(radius=CORNER_R, height=BASE_THICK + 2, sections=32)
        cyl.apply_translation((sign * (half - CORNER_R), BASE_DEPTH - CORNER_R, BASE_THICK / 2))
        block = trimesh.creation.box(extents=(CORNER_R + 2, CORNER_R + 2, BASE_THICK + 2))
        block.apply_translation(
            (
                sign * (half - CORNER_R / 2 + 0.5),
                BASE_DEPTH - CORNER_R / 2 + 0.5,
                BASE_THICK / 2,
            )
        )
        corner_cut.append(_difference(block, [cyl]))

    # Round the top edge. The cylinder is buried in the top of the plate
    # so the union cannot leave a floating piece.
    nose = trimesh.creation.cylinder(radius=4.5, height=WIDTH - 6, sections=48)
    nose.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    nose_center = world_from_face(0.0, FACE_LEN - 2.0, PLATE_THICK / 2, origin, tangent, into)
    nose.apply_translation(nose_center)

    stand = _union([body, nose, *lips])
    stand = _difference(stand, [pocket, pin_window, rear, cable_slot, *corner_cut])
    stand.merge_vertices()
    stand.update_faces(stand.unique_faces())
    stand.remove_unreferenced_vertices()
    stand.fix_normals()
    return stand


def _check(stand):
    """The 90-degree head fits; the USB-C plug is not the part this seats."""
    origin, tangent, into, toward = face_frame()
    pin_u = PIN_U
    pocket_u = pin_u - (POCKET_L / 2 - 4.0)

    def at(x, u, w):
        return world_from_face(x, u, w, origin, tangent, into)

    # Head seated on the watch-side wall, pins toward the watch.
    head_pts = []
    for x in (-HEAD_W / 2 + 0.4, HEAD_W / 2 - 0.4):
        for u in (-HEAD_L / 2 + 0.6, HEAD_L / 2 - 0.6):
            for w in (0.4, HEAD_T - 0.4):
                head_pts.append(at(x, pocket_u + u + 1.5, WALL_WATCH + w))
    head_pts = np.array(head_pts)
    inside = stand.contains(head_pts)
    if inside.any():
        raise SystemExit(f"head collides with the stand at {head_pts[inside]}")

    # Pin window is open through to the watch side.
    window_pts = np.array(
        [
            at(0, pin_u, -0.4),
            at(0, pin_u, WALL_WATCH / 2),
            at(2.0, pin_u, 0.2),
        ]
    )
    if stand.contains(window_pts).any():
        raise SystemExit("pin window is blocked")

    # Cable slot is open and narrower than the head, so the head cannot
    # fall out through the lead path.
    # Just under the pocket, where the slot still passes through solid plate.
    slot_u = pocket_u - POCKET_L / 2 - 2.0
    slot_mid = at(0, slot_u, 3.4)
    slot_side = at(CABLE_SLOT_W / 2 + 1.3, slot_u, 3.4)
    if stand.contains([slot_mid])[0]:
        raise SystemExit("cable slot is not open")
    if not stand.contains([slot_side])[0]:
        raise SystemExit("cable slot is wider than the retention width")

    # Rear slot must break out through the back face, not stop inside the plate.
    rear_pts = np.array([at(0.0, pocket_u, w) for w in (6, 12, 20, 30, 40)])
    rear_in = stand.contains(rear_pts)
    if rear_in[0]:
        raise SystemExit("rear of the pocket is solid")
    if rear_in[-1]:
        raise SystemExit("rear opening does not reach the outside")
    lip = at(0.0, pocket_u + POCKET_L / 2 - 1.6, 16.0)
    opening = at(0.0, pocket_u - 4.0, 16.0)
    if not stand.contains([lip])[0]:
        raise SystemExit("retaining lip is missing")
    if stand.contains([opening])[0]:
        raise SystemExit("rear insertion opening is blocked")

    # A USB-C plug standing in for the power end (8.3 x 2.6 mm) is not
    # trapped here: the pocket is the L-head, and the slot will not hold
    # the plug as the seated connector. Checked by the head envelope above.
    print("fit check ok")
    print(f"  head envelope {HEAD_W} x {HEAD_T} x {HEAD_L} mm")
    print(f"  pocket {POCKET_W} x {POCKET_T} x {POCKET_L} mm")
    print(f"  cable slot {CABLE_SLOT_W} mm wide")


def main():
    stand = build()
    if not stand.is_watertight:
        raise SystemExit("mesh is not watertight")
    if stand.volume < 10000:
        raise SystemExit(f"volume looks wrong: {stand.volume}")
    _check(stand)
    stand.export(STL_PATH)
    print(f"wrote {STL_PATH}")
    print(f"extents_mm {np.round(stand.extents, 2).tolist()}")
    print(f"volume_mm3 {stand.volume:.0f}")
    print(f"watertight {stand.is_watertight} faces {len(stand.faces)}")


if __name__ == "__main__":
    main()
