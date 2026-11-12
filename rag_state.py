from contextvars import ContextVar

CURRENT_THREAD_ID: ContextVar = ContextVar("current_thread_id", default=None)
