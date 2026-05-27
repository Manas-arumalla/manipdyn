"""Motion planning: collision checking, sampling planners, and smoothing.

================  ============================================================
name              planner
================  ============================================================
``rrt``           Rapidly-exploring Random Tree
``rrt_connect``   bidirectional RRT-Connect (fast default)
``rrt_star``      RRT* (asymptotically optimal)
``informed_rrt_star``  Informed RRT* (ellipsoidal sampling, faster optimal)
``prm``           Probabilistic Roadmap (multi-query)
================  ============================================================
"""

from manipdyn.planning.base import Node, Planner, reconstruct
from manipdyn.planning.collision import CollisionChecker
from manipdyn.planning.prm import PRM
from manipdyn.planning.rrt import RRT, RRTConnect
from manipdyn.planning.rrt_star import InformedRRTStar, RRTStar
from manipdyn.planning.smoothing import shortcut_path, smooth_bspline

PLANNERS: dict[str, type[Planner]] = {
    RRT.name: RRT,
    RRTConnect.name: RRTConnect,
    RRTStar.name: RRTStar,
    InformedRRTStar.name: InformedRRTStar,
    PRM.name: PRM,
}

__all__ = [
    "Planner",
    "Node",
    "reconstruct",
    "CollisionChecker",
    "RRT",
    "RRTConnect",
    "RRTStar",
    "InformedRRTStar",
    "PRM",
    "shortcut_path",
    "smooth_bspline",
    "PLANNERS",
]
