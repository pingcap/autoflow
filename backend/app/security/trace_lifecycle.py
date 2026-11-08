"""Bound the pinned Langfuse v2 SDK's per-chat queues and final flush."""

import atexit
import logging
from queue import Empty
import time

FLUSH_SECONDS = 5
CLEANUP_SECONDS = 3
MAX_PENDING_EVENTS = 256
logger = logging.getLogger(__name__)


def preserve_text_trace(_event):
    # Autoflow traces text and JSON. The SDK otherwise treats a user-supplied
    # "data:" string as media and starts a separate requests.put upload outside
    # this client's transport. Preserve the original text without media upload.
    return None


class TraceLifecycle:
    def __init__(self, client, transport):
        self.manager = client.task_manager
        self.manager._media_manager.process_media_in_event = preserve_text_trace
        self.prompts = client.prompt_cache._task_manager
        self.transport = transport
        self.flushed = False
        self.closed = False
        self.queues = [
            self.manager._ingestion_queue,
            self.manager._media_upload_queue,
            self.prompts._queue,
        ]
        for queue in self.queues:
            with queue.mutex:
                queue.maxsize = MAX_PENDING_EVENTS

    def flush(self):
        if self.flushed or self.closed:
            return
        self.flushed = True
        deadline = time.monotonic() + FLUSH_SECONDS
        for queue in self.queues:
            with queue.all_tasks_done:
                while queue.unfinished_tasks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "Trace flush deadline reached; remaining events will be discarded"
                        )
                        return
                    queue.all_tasks_done.wait(timeout=remaining)

    def close(self):
        if self.closed:
            return
        self.manager._enabled = False
        self.flush()
        self.closed = True
        # The SDK's native shutdown uses unbounded queue/thread joins. Remove
        # only these owned exit hooks and stop only these owned consumers.
        atexit.unregister(self.manager.shutdown)
        atexit.unregister(self.prompts.shutdown)
        consumers = [
            *self.manager._ingestion_consumers,
            *self.manager._media_upload_consumers,
            *self.prompts._consumers,
        ]
        for consumer in consumers:
            consumer.pause()
        self.transport.abort()
        for queue in self.queues:
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break
                else:
                    queue.task_done()
        deadline = time.monotonic() + CLEANUP_SECONDS
        for consumer in consumers:
            consumer.join(timeout=max(0, deadline - time.monotonic()))
        if any(consumer.is_alive() for consumer in consumers):
            # OS DNS or SDK CPU work cannot be killed by Python threads. The
            # aborted transport refuses any subsequent network connection.
            logger.warning("Trace cleanup deadline reached; transport remains disabled")
