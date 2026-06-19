#@ String input
#@ String output
"""
TrackMate tracking runner -- runs INSIDE the smlm-labflow/trackmate Fiji image as a
headless Jython script (Tinevez et al. 2017 / Ershov et al. 2022), invoked by
labflow's `runtime: docker` track method.

Contract: localizations.csv (frame, x, y) in -> tracks.csv (track_id, frame, x, y) out.

BINDING POINT: TrackMate's primary path is image -> detector -> tracks. Our pipeline
already localized, so to LINK pre-detected localizations you build a TrackMate Model
from the CSV spots and run the LAP tracker via the Java API. That scripting is the
binding point; the Fiji environment and the file contract are set up here. Sketch:

    from fiji.plugin.trackmate import Model, Settings, TrackMate, Spot, SpotCollection
    from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
    # 1. read `input` CSV -> Spot(x,y,z,radius,quality), add to SpotCollection by frame
    # 2. model.setSpots(collection, False)
    # 3. settings.trackerFactory = SparseLAPTrackerFactory(); settings.trackerSettings =
    #    {LINKING_MAX_DISTANCE, GAP_CLOSING_MAX_FRAME_GAP, ...}
    # 4. TrackMate(model, settings).process()
    # 5. export per-spot track ids -> `output` (track_id, frame, x, y)

Until wired it exits with a clear message (no fabricated API, no wrong tracks).
"""

raise Exception(
    "TrackMate scripting is the binding point: wire docker/trackmate/run_trackmate.py "
    "to the TrackMate Java API (build a Model from the CSV spots, run SparseLAPTracker, "
    "export tracks.csv). The Fiji image and the IO contract are already set up.")
