"""Start a command in its own session, so it keeps running after the terminal or agent session that
launched it has gone.

    python3 adapters/detach.py LOGFILE COMMAND [ARGS...]
"""
import subprocess
import sys

log = open(sys.argv[1], "ab")
child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
print(child.pid)
