"""HiveMind launcher - the double-click entry (issue #41, wave #41).

Frozen by PyInstaller this module is the body of HiveMind.exe:
it starts the Streamlit server on a free local port, opens the
default browser once the server answers, and keeps the process
alive until the server dies. The build is windowed - no console
- so stdout and stderr are redirected to a log file first: a
silent failure is the one failure we never allow.
"""
import os
import socket
import sys
import threading
import time
import webbrowser

LOG_DIR = os.path.join(os.path.expanduser("~"), ".hivemind")
LOG_PATH = os.path.join(LOG_DIR, "launcher.log")
PREFERRED_PORT = 8501
START_TIMEOUT_SECONDS = 120


def _redirect_output():
    """Windowed exes have no console: give every print a home."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    sys.stdout = log
    sys.stderr = log
    sys.__stdout__ = log
    sys.__stderr__ = log
    print("launcher: starting at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))


def _free_port(preferred):
    """The preferred port, or the next free one nearby."""
    for candidate in (preferred, preferred + 1, preferred + 2, 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate))
            return sock.getsockname()[1]
        except OSError:
            continue
        finally:
            sock.close()
    return preferred


def _port_answers(port):
    """True when something is listening on 127.0.0.1:port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _app_path():
    """app.py: bundled next to this module in a frozen build."""
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "app.py")


def _configure(port):
    """Tell Streamlit, via its environment variables, how to run."""
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"


def _server_thread(app_path, failed):
    """Run Streamlit's own CLI in this thread; record any death."""
    try:
        from streamlit.web import cli as st_cli
        sys.argv = ["streamlit", "run", app_path]
        st_cli.main()
    except BaseException as exc:
        failed.append("%s: %s" % (type(exc).__name__, exc))


def main():
    """Start the server, open the browser, keep the process alive."""
    _redirect_output()
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and meipass not in sys.path:
        sys.path.insert(0, meipass)
    app_path = _app_path()
    if not os.path.exists(app_path):
        print("launcher: app.py not found at %s" % app_path)
        return 1
    port = _free_port(PREFERRED_PORT)
    _configure(port)
    failed = []
    threading.Thread(target=_server_thread, args=(app_path, failed),
                     daemon=True).start()
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if failed:
            print("launcher: server failed to start: %s" % failed[0])
            return 1
        if _port_answers(port):
            break
        time.sleep(0.5)
    else:
        print("launcher: server did not answer within %d s"
              % START_TIMEOUT_SECONDS)
        return 1
    url = "http://localhost:%d" % port
    print("launcher: server up, opening %s" % url)
    webbrowser.open(url)
    while not failed and _port_answers(port):
        time.sleep(2.0)
    if failed:
        print("launcher: server died: %s" % failed[0])
        return 1
    print("launcher: server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
