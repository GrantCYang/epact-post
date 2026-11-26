"""Local episode-server transport; no model or simulator dependency."""

import json
import socket


def advance_state(previous, reply):
    """The action endpoint omits query; retain immutable task context."""
    return {
        **reply,
        **{
            k: previous[k]
            for k in ("query", "budget", "protocol", "reference_help")
            if k in previous
        },
    }


class ServerLink:
    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.settimeout(300.0)
        self.file = self.sock.makefile("rwb")

    def close(self):
        try:
            self.file.close()
        finally:
            self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, **msg) -> dict:
        self.file.write((json.dumps(msg) + "\n").encode())
        self.file.flush()
        reply = json.loads(self.file.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"server error: {reply.get('error')}")
        return reply
