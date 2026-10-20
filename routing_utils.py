"""
Shared base utilities for PCB routing.

This module contains the core position and geometry utilities used by all routing modules.
The bulk of routing functionality has been split into:
- connectivity.py: Endpoint finding, stub analysis, MST algorithms
- net_queries.py: Pad/net queries, MPS ordering, route length calculations
- pcb_modification.py: Add/remove routes, segment cleanup
"""

import math
from typing import Dict, List, Tuple

from kicad_parser import Segment, POSITION_DECIMALS


def build_layer_map(layers: List[str]) -> Dict[str, int]:
    """
    Build a mapping from layer names to indices.

    Args:
        layers: List of layer names (e.g., ['F.Cu', 'In1.Cu', 'B.Cu'])

    Returns:
        Dict mapping layer name to its index in the list
    """
    return {name: idx for idx, name in enumerate(layers)}


def pos_key(x: float, y: float) -> Tuple[float, float]:
    """
    Normalize coordinates for position-based lookups.

    Use this consistently when building position sets or checking position membership
    to avoid floating-point comparison issues.
    """
    return (round(x, POSITION_DECIMALS), round(y, POSITION_DECIMALS))


def segment_length(seg: Segment) -> float:
    """Calculate the length of a single segment."""
    return math.sqrt((seg.end_x - seg.start_x)**2 + (seg.end_y - seg.start_y)**2)


def dist_sq_to_rounded_rect(
    point_x: float, point_y: float,
    half_width: float, half_height: float,
    corner_radius: float = 0.0
) -> float:
    """
    Calculate squared distance from a point to a rounded rectangle centered at origin.

    The rectangle extends from (-half_width, -half_height) to (half_width, half_height).
    Corner radius creates rounded corners (0 for sharp corners).

    Args:
        point_x, point_y: Point coordinates (relative to rectangle center)
        half_width, half_height: Rectangle half-dimensions
        corner_radius: Radius of rounded corners (0 for sharp corners)

    Returns:
        Squared distance from point to rectangle edge (0 if inside)
    """
    abs_x, abs_y = abs(point_x), abs(point_y)

    if corner_radius > 0:
        # Inner rectangle bounds (where corners start)
        inner_half_w = half_width - corner_radius
        inner_half_h = half_height - corner_radius

        # Check if point is in a corner region
        if abs_x > inner_half_w and abs_y > inner_half_h:
            # Distance to corner arc center
            dx = abs_x - inner_half_w
            dy = abs_y - inner_half_h
            dist_to_corner_center = math.sqrt(dx * dx + dy * dy)
            # Distance to arc edge
            dist = dist_to_corner_center - corner_radius
            return dist * dist if dist > 0 else 0.0

    # Point is along a flat edge - rectangular distance
    closest_x = max(-half_width, min(point_x, half_width))
    closest_y = max(-half_height, min(point_y, half_height))
    dx = point_x - closest_x
    dy = point_y - closest_y
    return dx * dx + dy * dy


def pad_rotation_residual(pad) -> float:
    """
    Rotation (degrees, in [-45, 45]) of a pad's size_x/size_y frame on the board.

    The parser stores ``pad.rotation`` as the pad's absolute board orientation
    and already swaps size_x/size_y for pads near 90/270 degrees, so pads at
    0/90/180/270 return exactly 0.0 here. Anything else (e.g. a QFP placed at
    45 degrees) returns the angle still to be applied to the swapped sizes.
    KiCad's angles are counter-clockwise on screen (y-down board coords).
    """
    rot = getattr(pad, 'rotation', 0.0) % 360
    if 45 < rot % 180 < 135:  # Same rule the parser uses to swap size_x/size_y
        rot -= 90
    rot = (rot + 90) % 180 - 90
    return 0.0 if abs(rot) < 1e-6 else rot


def pad_half_extents(pad) -> Tuple[float, float]:
    """Axis-aligned half-extents (mm) of a pad's bounding box on the board."""
    half_w = pad.size_x / 2
    half_h = pad.size_y / 2
    rot = pad_rotation_residual(pad)
    if rot == 0.0:
        return half_w, half_h
    c = abs(math.cos(math.radians(rot)))
    s = abs(math.sin(math.radians(rot)))
    return half_w * c + half_h * s, half_w * s + half_h * c


def iter_pad_blocked_cells(
    pad_gx: int, pad_gy: int,
    half_width: float, half_height: float,
    margin: float,
    grid_step: float,
    corner_radius: float = 0.0,
    rotation: float = 0.0
):
    """
    Generate grid cells that should be blocked for a pad.

    Yields (gx, gy) tuples for all cells within margin distance of the pad edge.
    Uses rectangular-with-rounded-corners shape matching the actual pad geometry.

    Args:
        pad_gx, pad_gy: Pad center in grid coordinates
        half_width, half_height: Pad half-dimensions in mm
        margin: Blocking margin in mm (e.g., track_width/2 + clearance)
        grid_step: Grid step size in mm
        corner_radius: Corner radius for roundrect pads (0 for rectangular)
        rotation: Pad rotation in degrees from pad_rotation_residual()
            (0 for axis-aligned pads)

    Yields:
        (gx, gy) tuples for each blocked cell
    """
    if rotation:
        yield from _iter_rotated_pad_blocked_cells(
            pad_gx, pad_gy, half_width, half_height, margin, grid_step,
            corner_radius, rotation)
        return

    # Add half grid step buffer to account for grid discretization
    # (a track through a cell could be grid_step/2 closer to the pad than the cell center)
    # This applies to ALL pad shapes since diagonal approaches can occur with rectangular pads too
    corner_buffer = grid_step / 2
    expand_x = int(math.ceil((half_width + margin + corner_buffer) / grid_step))
    expand_y = int(math.ceil((half_height + margin + corner_buffer) / grid_step))
    margin_sq = margin * margin
    buffered_margin_sq = (margin + corner_buffer) * (margin + corner_buffer)

    # Inner rectangle bounds (where corners start)
    # For rectangular pads (corner_radius=0), define corner region with a threshold
    corner_threshold = grid_step / 2 if corner_radius == 0 else 0
    inner_half_w = half_width - corner_radius - corner_threshold
    inner_half_h = half_height - corner_radius - corner_threshold

    for ex in range(-expand_x, expand_x + 1):
        for ey in range(-expand_y, expand_y + 1):
            cell_x = ex * grid_step
            cell_y = ey * grid_step
            abs_x, abs_y = abs(cell_x), abs(cell_y)

            # Use buffered margin in corner/diagonal regions where grid discretization
            # can cause tracks to be closer than expected (applies to all pad shapes)
            in_corner = abs_x > inner_half_w and abs_y > inner_half_h
            effective_margin_sq = buffered_margin_sq if in_corner else margin_sq

            dist_sq = dist_sq_to_rounded_rect(cell_x, cell_y, half_width, half_height, corner_radius)
            if dist_sq < effective_margin_sq:
                yield (pad_gx + ex, pad_gy + ey)


def _iter_rotated_pad_blocked_cells(
    pad_gx: int, pad_gy: int,
    half_width: float, half_height: float,
    margin: float,
    grid_step: float,
    corner_radius: float,
    rotation: float
):
    """
    iter_pad_blocked_cells() for a pad that is not axis-aligned.

    Each cell offset is rotated into the pad's own frame and then tested with
    the same rounded-rectangle distance and corner buffer as the axis-aligned
    case. ``rotation`` follows KiCad's convention: a pad-frame point (lx, ly)
    sits at board offset (lx*cos + ly*sin, -lx*sin + ly*cos) in y-down coords.
    """
    corner_buffer = grid_step / 2
    rad = math.radians(rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    # Bounding box of the rotated pad plus margin
    reach_x = abs(half_width * cos_r) + abs(half_height * sin_r) + margin + corner_buffer
    reach_y = abs(half_width * sin_r) + abs(half_height * cos_r) + margin + corner_buffer
    expand_x = int(math.ceil(reach_x / grid_step))
    expand_y = int(math.ceil(reach_y / grid_step))
    margin_sq = margin * margin
    buffered_margin_sq = (margin + corner_buffer) * (margin + corner_buffer)

    corner_threshold = grid_step / 2 if corner_radius == 0 else 0
    inner_half_w = half_width - corner_radius - corner_threshold
    inner_half_h = half_height - corner_radius - corner_threshold

    for ex in range(-expand_x, expand_x + 1):
        for ey in range(-expand_y, expand_y + 1):
            dx = ex * grid_step
            dy = ey * grid_step
            # Board offset -> pad frame (inverse of the KiCad y-down rotation)
            local_x = dx * cos_r - dy * sin_r
            local_y = dx * sin_r + dy * cos_r
            in_corner = abs(local_x) > inner_half_w and abs(local_y) > inner_half_h
            effective_margin_sq = buffered_margin_sq if in_corner else margin_sq

            dist_sq = dist_sq_to_rounded_rect(local_x, local_y, half_width, half_height, corner_radius)
            if dist_sq < effective_margin_sq:
                yield (pad_gx + ex, pad_gy + ey)
