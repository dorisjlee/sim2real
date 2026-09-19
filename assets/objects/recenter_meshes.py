"""Re-center STL meshes so origin = (x_center, y_center, z_min) of the bounding box.

The as-exported STLs carry their absolute CAD placement in their vertices,
which puts them far from the body origin once loaded into MuJoCo. This
rewrites each mesh in place (after backing up the original) so the origin
sits at the center of the object's base -- a natural pivot for placing it on
a table in a scene XML.
"""
import shutil
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
FILES = ["rectangle.stl", "rectangle_wide.stl", "tray.stl"]

for fname in FILES:
    path = HERE / fname
    backup = HERE / f"{path.stem}.orig.stl"
    if not backup.exists():
        shutil.copy(path, backup)

    mesh = trimesh.load(str(backup), force="mesh")
    bounds = mesh.bounds  # (2, 3): min, max
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    z_min = bounds[0][2]
    offset = np.array([center_xy[0], center_xy[1], z_min])

    mesh.apply_translation(-offset)
    mesh.export(str(path))

    size_mm = bounds[1] - bounds[0]
    print(f"{fname}: size(mm)={size_mm.round(2)} offset_removed(mm)={offset.round(2)}")
