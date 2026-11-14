"""Overview integration hook; no camera capture/change detector implemented yet.

An adapter can publish EventKind.SCENE_CHANGED after confirming an unexplained
change. It must ignore/debounce expected robot activity and respect context.fake.
No motion commands and no independent writes to the main worker's object list.
"""
from core.runtime_events import EventKind, ProducerContext


def run(context: ProducerContext):
    context.stop.wait()  # Inert scaffold, no polling or camera connection.
