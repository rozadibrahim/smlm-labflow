"""
schema.py

Canonical CSV contracts for each pipeline stage. The file contract between stages
is what makes methods pluggable: a stage consumes the previous stage's CSV and
produces its own, regardless of the tool or language behind it.

    localize/drift -> CANONICAL_COLUMNS   (one row per localization)
    track          -> TRACK_COLUMNS       (localizations linked into trajectories)
    analyze        -> ANALYSIS_COLUMNS     (one row per track)
"""

CANONICAL_COLUMNS = [
    "frame",
    "x",
    "y",
    "z",
    "photons",
    "background",
    "confidence",
    "backend",
    "source_file",
]


REQUIRED_COLUMNS = [
    "frame",
    "x",
    "y",
    "backend",
]


# Tracking stage: localizations linked into trajectories (track_id added).
TRACK_COLUMNS = [
    "track_id",
    "frame",
    "x",
    "y",
    "z",
    "photons",
]

REQUIRED_TRACK_COLUMNS = [
    "track_id",
    "frame",
    "x",
    "y",
]


# Cluster stage: localizations annotated with a cluster id (noise = -1),
# plus a one-row-per-cluster summary.
CLUSTER_COLUMNS = ["frame", "x", "y", "z", "cluster_id"]
CLUSTER_SUMMARY_COLUMNS = [
    "cluster_id",
    "n_localizations",
    "centroid_x",
    "centroid_y",
    "radius_gyration_nm",
]


# Analysis stage: per-track properties (one row per track).
ANALYSIS_COLUMNS = [
    "track_id",
    "n_localizations",
    "length_nm",
    "duration_frames",
    "diffusion_coefficient",
    "alpha",
    "label",
]

REQUIRED_ANALYSIS_COLUMNS = [
    "track_id",
]


def get_canonical_columns():
    return CANONICAL_COLUMNS.copy()


def get_required_columns():
    return REQUIRED_COLUMNS.copy()


def get_track_columns():
    return TRACK_COLUMNS.copy()


def get_analysis_columns():
    return ANALYSIS_COLUMNS.copy()
