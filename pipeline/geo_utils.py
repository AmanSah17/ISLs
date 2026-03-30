"""
geo_utils.py — Python port of GLASSEAS Scala geo utilities.

Faithfully replicates:
  - SpatialToolkit (haversine, convex hull, Lagrange interpolation, projection)
  - Grid (spatial grid with cell lookup)
  - Polygon (point-in-polygon test, WKT format)
  - LocalDatabase (waypoint/convex hull spatial index)
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set


# ============================================================
# GeoPoint (port of geotools/GeoPoint.scala)
# ============================================================
class GeoPoint:
    __slots__ = ['longitude', 'latitude']

    def __init__(self, longitude: float, latitude: float):
        self.longitude = longitude
        self.latitude = latitude

    def __eq__(self, other):
        return isinstance(other, GeoPoint) and self.longitude == other.longitude and self.latitude == other.latitude

    def __hash__(self):
        return hash((self.longitude, self.latitude))

    def __repr__(self):
        return f"GeoPoint({self.longitude}, {self.latitude})"


# ============================================================
# Polygon (port of geotools/Polygon.scala)
# ============================================================
class Polygon:
    def __init__(self, points: List[GeoPoint] = None):
        self.points = points or []

    def is_inside(self, p: GeoPoint) -> bool:
        """Ray-casting point-in-polygon test — exact port of Polygon.isInside"""
        c = False
        n = len(self.points)
        if n < 3:
            return False
        all_pts = [self.points[-1]] + self.points
        for k in range(len(all_pts) - 1):
            i = all_pts[k]
            j = all_pts[k + 1]
            cond = (
                ((i.latitude <= p.latitude and p.latitude < j.latitude) or
                 (j.latitude <= p.latitude and p.latitude < i.latitude)) and
                (p.longitude < (j.longitude - i.longitude) * (p.latitude - i.latitude) /
                 (j.latitude - i.latitude) + i.longitude)
            )
            if cond:
                c = not c
        return c

    def to_wkt(self) -> str:
        coords = ", ".join(f"{gp.longitude} {gp.latitude}" for gp in self.points)
        return f'"POLYGON (({coords}))"'

    @staticmethod
    def from_wkt(line: str) -> 'Polygon':
        cleaned = line.replace('"POLYGON ((', '').replace('))"', '')
        parts = cleaned.split(", ")
        points = []
        for part in parts:
            xy = part.strip().split()
            points.append(GeoPoint(float(xy[0]), float(xy[1])))
        return Polygon(points)

    def __repr__(self):
        return self.to_wkt()


# ============================================================
# Cell (port of geotools/Cell.scala)
# ============================================================
class Cell:
    __slots__ = ['id', 'geo_point']

    def __init__(self, cell_id: int, geo_point: GeoPoint):
        self.id = cell_id
        self.geo_point = geo_point

    def __repr__(self):
        return f"Cell({self.id}, {self.geo_point})"


# ============================================================
# Grid (port of geotools/Grid.scala)
# ============================================================
class Grid:
    MU = 1000000

    def __init__(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 step_lon: float, step_lat: float, cells: Dict[GeoPoint, Cell] = None):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.step_lon = step_lon
        self.step_lat = step_lat
        if cells is None:
            self.cells = self._create_grid()
        else:
            self.cells = cells

    def _create_grid(self) -> Dict[GeoPoint, Cell]:
        """Exact port of Grid.createGrid"""
        cells = {}
        cell_id = 1
        y = self.min_lat
        while y < self.max_lat:
            x = self.min_lon
            while x < self.max_lon:
                lon = round(x, 2)
                lat = round(y, 2)
                gp = GeoPoint(lon, lat)
                cells[gp] = Cell(cell_id, gp)
                if lon != round(self.max_lon - self.step_lon, 2):
                    cell_id += 1
                x += self.step_lon
            cell_id += 1
            y += self.step_lat
        return cells

    def get_enclosing_cell(self, p: GeoPoint) -> Optional[Cell]:
        """Exact port of Grid.getEnclosingCell"""
        x = round(p.longitude * self.MU)
        y = round(p.latitude * self.MU)
        x_start = round(self.min_lon * self.MU)
        x_step = round(self.step_lon * self.MU)
        y_start = round(self.min_lat * self.MU)
        y_step = round(self.step_lat * self.MU)
        llx = x - ((x - x_start) % x_step)
        lly = y - ((y - y_start) % y_step)
        low_left_x = llx / self.MU
        low_left_y = lly / self.MU
        gp = GeoPoint(low_left_x, low_left_y)
        return self.cells.get(gp, None)


# ============================================================
# SpatialToolkit (port of geotools/SpatialToolkit.scala)
# ============================================================
def haversine_distance(gp1: GeoPoint, gp2: GeoPoint) -> float:
    """Distance in kilometers — exact port of getHaversineDistance"""
    lon1 = math.radians(gp1.longitude)
    lat1 = math.radians(gp1.latitude)
    lon2 = math.radians(gp2.longitude)
    lat2 = math.radians(gp2.latitude)
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371 * c


def haversine_distance_meters(gp1: GeoPoint, gp2: GeoPoint) -> float:
    """Distance in meters"""
    return haversine_distance(gp1, gp2) * 1000


def lagrange_interpolation(x: np.ndarray, y: np.ndarray, x_point: float) -> float:
    """Exact port of SpatialToolkit.lagrangeInterpolation"""
    total = 0.0
    n = len(x)
    for i in range(n):
        product = 1.0
        for j in range(n):
            if j != i:
                product *= (x_point - x[j]) / (x[i] - x[j])
        total += product * y[i]
    return total


def get_convex_hull(points: List[GeoPoint]) -> Polygon:
    """Exact port of SpatialToolkit.getConvexHull"""
    sorted_pts = sorted(points, key=lambda p: p.longitude)
    upper = _half_hull(sorted_pts)
    lower = _half_hull(list(reversed(sorted_pts)))
    if upper:
        upper.pop(0)
    if lower:
        lower.pop(0)
    polygon_points = upper + lower
    if polygon_points:
        polygon_points.append(polygon_points[0])
    return Polygon(polygon_points)


def _half_hull(points: List[GeoPoint]) -> List[GeoPoint]:
    upper = []
    for p in points:
        while len(upper) >= 2 and _left_turn(p, upper[0], upper[1]):
            upper.pop(0)
        upper.insert(0, p)
    return upper


def _left_turn(p1: GeoPoint, p2: GeoPoint, p3: GeoPoint) -> bool:
    slope = ((p2.longitude - p1.longitude) * (p3.latitude - p1.latitude) -
             (p2.latitude - p1.latitude) * (p3.longitude - p1.longitude))
    collinear = abs(slope) <= 1e-9
    left = slope < 0
    return collinear or left


def compute_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute bearing between two points in degrees"""
    prev_lon = lon1
    cur_lon = lon2
    # Same logic as in InterpolationSparkApp
    if prev_lon < 0 and cur_lon < 0:
        dl = abs(abs(prev_lon) - abs(cur_lon))
    elif prev_lon > 0 and cur_lon > 0:
        dl = abs(prev_lon - cur_lon)
    else:
        dl = abs(prev_lon) + abs(cur_lon)
    X = math.cos(math.radians(lat2)) * math.sin(math.radians(dl))
    Y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dl)))
    bearing = math.degrees(math.atan2(X, Y))
    return bearing


# ============================================================
# LocalDatabase (port of LocalDatabase.scala)
# ============================================================
class WaypointDatabase:
    """Spatial index for waypoints / convex hulls — port of LocalDatabase"""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.waypoints: Dict[int, int] = {}  # polygon_idx -> waypoint_id
        self.waypoint_polygons: List[Polygon] = []
        self.waypoints_per_index: Dict[int, Set[int]] = {}  # cell_id -> set of polygon indices

    def add_waypoints(self, polygons: List[Polygon], ids: List[int]):
        """Add waypoint polygons with their IDs"""
        for poly, wp_id in zip(polygons, ids):
            idx = len(self.waypoint_polygons)
            self.waypoint_polygons.append(poly)
            self.waypoints[idx] = wp_id
            # Index by grid cells
            for pt in poly.points:
                cell = self.grid.get_enclosing_cell(pt)
                if cell is not None:
                    if cell.id not in self.waypoints_per_index:
                        self.waypoints_per_index[cell.id] = set()
                    self.waypoints_per_index[cell.id].add(idx)

    def get_enclosing_waypoint(self, p: GeoPoint) -> Optional[Tuple[Polygon, int]]:
        """Port of LocalDatabase.getEnclosingWaypoint"""
        cell = self.grid.get_enclosing_cell(p)
        if cell is None:
            return None

        grid_idx_len = abs(self.grid.min_lon) + abs(self.grid.max_lon) / self.grid.step_lon
        neighbors = [
            cell.id, cell.id + 1, cell.id - 1,
            cell.id + grid_idx_len, cell.id + grid_idx_len + 1, cell.id + grid_idx_len - 1,
            cell.id - grid_idx_len, cell.id - grid_idx_len + 1, cell.id - grid_idx_len - 1,
        ]

        candidate_indices = set()
        for n in neighbors:
            if n in self.waypoints_per_index:
                candidate_indices |= self.waypoints_per_index[n]

        if not candidate_indices:
            return None

        for idx in candidate_indices:
            poly = self.waypoint_polygons[idx]
            if poly.is_inside(p):
                return (poly, self.waypoints[idx])
        return None


# ============================================================
# AIS Vessel Type Mapping
# ============================================================
VESSEL_TYPE_MAP = {
    range(20, 30): "WIG",
    range(30, 40): "Fishing",
    range(40, 50): "HSC",
    range(50, 60): "Tug/Special",
    range(60, 70): "Passenger",
    range(70, 80): "Cargo",
    range(80, 90): "Tanker",
    range(90, 100): "Other",
}


def map_vessel_type(code) -> str:
    """Map AIS numeric vessel type code to string name"""
    try:
        code_int = int(code)
    except (ValueError, TypeError):
        return "Unknown"
    for r, name in VESSEL_TYPE_MAP.items():
        if code_int in r:
            return name
    return "Unknown"
