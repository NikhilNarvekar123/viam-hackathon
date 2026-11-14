"""Audio integration hook; no ASR model, microphone access or command mapping yet.

Replace the wait with bounded input reads respecting context.stop/context.fake.
After validating a command, context.publish(EventKind.RESET_BATCH) can request
fresh inspection at the next controller checkpoint. Unknown commands must do
nothing. No motion, automatic release, or immediate voice-stop API lives here.
"""
from core.runtime_events import EventKind, ProducerContext


def run(context: ProducerContext):
    context.stop.wait()  # Inert scaffold, not a functioning audio listener.
