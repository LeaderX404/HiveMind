"""HiveMind launcher - the double-click entry (issues #41, #42).

Frozen by PyInstaller this module is the body of HiveMind.exe:
it starts the Streamlit server on a free local port, opens the
default browser once the server answers, and keeps the process
alive until the server dies. The build is windowed - no console
- so stdout and stderr are redirected to a log file first: a
silent failure is the one failure we never allow.

Wave #42: a system tray icon owns the taskbar after the browser
closes - "Open HiveMind" reopens the app, "Quit" ends the
process on purpose, so nothing ghosts in Task Manager. Without
pystray (plain dev runs) the launcher falls back to the old
heartbeat loop.
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


def _tray_image():
    """Draw the tray icon at runtime - a green tile with an H -
    no binary asset to forget in the bundle."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (64, 64), (31, 127, 55))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 55), outline=(255, 255, 255), width=3)
    draw.text((24, 20), "H", fill=(255, 255, 255))
    return image


def _watch_server(port, icon):
    """End the tray when the server dies: no dead-app ghost."""
    while _port_answers(port):
        time.sleep(2.0)
    print("launcher: server stopped while the tray was up")
    try:
        icon.stop()
    except Exception:
        pass


def _run_tray(url, port):
    """Own the main thread with the tray icon. False when pystray
    is unavailable (dev runs) - the caller falls back."""
    try:
        import pystray
    except ImportError:
        print("launcher: pystray missing - no tray icon this run")
        return False

    def _reopen(icon, item):
        print("launcher: reopen requested from the tray")
        webbrowser.open(url)

    def _quit(icon, item):
        print("launcher: quit requested from the tray")
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open HiveMind", _reopen),
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("HiveMind", _tray_image(), "HiveMind", menu)
    threading.Thread(target=_watch_server, args=(port, icon),
                     daemon=True).start()
    print("launcher: tray icon ready")
    icon.run()
    print("launcher: tray stopped, exiting")
    return True


def main():
    """Start the server, open the browser, hold the tray."""
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
    try:
        if _run_tray(url, port):
            return 0
    except Exception as exc:
        print("launcher: tray failed (%s) - heartbeat fallback" % exc)
    while not failed and _port_answers(port):
        time.sleep(2.0)
    if failed:
        print("launcher: server died: %s" % failed[0])
        return 1
    print("launcher: server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
