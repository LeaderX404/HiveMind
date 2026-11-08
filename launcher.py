"""HiveMind launcher - the double-click entry (issues #41, #42,
#44, #46).

The MAIN THREAD runs the Streamlit server and nothing else.
Streamlit registers signal handlers while booting, and Python
allows that only in the main thread (issue #46: the old flow
ran the server in a background thread, and Streamlit died at
once with "signal only works in main thread of the main
interpreter"). The browser and the tray icon live in daemon
threads and never touch the main thread. The build is windowed
- no console - so stdout and stderr are redirected to a log
file first: a silent failure is the one failure we never
allow. Issue #44: frozen builds write launcher.log next to the
exe (not in ~/.hivemind), and the Debug target
(HiveMind-Debug.exe) keeps a console so the first crash is
printed, not buried.

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
import traceback
import webbrowser

LOG_DIR = os.path.join(os.path.expanduser("~"), ".hivemind")
PREFERRED_PORT = 8501
START_TIMEOUT_SECONDS = 120


def _debug_build():
    """True for the HiveMind-Debug.exe target (issue #44): the
    console build prints straight to the black window."""
    return "debug" in os.path.basename(sys.executable).lower()


def _log_path():
    """launcher.log lives next to the exe in frozen builds so
    the user always finds it (issue #44); dev runs keep
    ~/.hivemind/launcher.log."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable),
                            "launcher.log")
    return os.path.join(LOG_DIR, "launcher.log")


def _redirect_output():
    """Windowed exes have no console: give every print a home."""
    log_path = _log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a", buffering=1, encoding="utf-8")
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


def _background(port):
    """Everything except the server: wait for the port to answer,
    open the browser, then hold the tray (or the heartbeat
    fallback). Daemon thread - the server owns the main thread
    (issue #46)."""
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline and not _port_answers(port):
        time.sleep(0.5)
    if not _port_answers(port):
        print("launcher: server did not answer within %d s"
              % START_TIMEOUT_SECONDS)
        return
    url = "http://localhost:%d" % port
    print("launcher: server up, opening %s" % url)
    webbrowser.open(url)
    try:
        if _run_tray(url, port):
            return
    except Exception as exc:
        print("launcher: tray failed (%s) - heartbeat fallback" % exc)
    while _port_answers(port):
        time.sleep(2.0)
    print("launcher: server stopped")


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
    """Hold the taskbar with the tray icon from a background
    thread. False when pystray is unavailable (dev runs) - the
    caller falls back."""
    try:
        import pystray
    except ImportError:
        print("launcher: pystray missing - no tray icon this run")
        return False

    def _reopen(icon, item):
        print("launcher: reopen requested from the tray")
        webbrowser.open(url)

    def _quit(icon, item):
        """The server owns the main thread, so Quit ends the whole
        process on purpose - nothing ghosts in Task Manager."""
        print("launcher: quit requested from the tray")
        icon.stop()
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Open HiveMind", _reopen),
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("HiveMind", _tray_image(), "HiveMind", menu)
    threading.Thread(target=_watch_server, args=(port, icon),
                     daemon=True).start()
    print("launcher: tray icon ready")
    icon.run()
    print("launcher: tray stopped")
    return True


def main():
    """The main thread IS the Streamlit server (issue #46); the
    browser and the tray run in background threads."""
    if _debug_build():
        print("launcher: debug build - output stays on this console")
    else:
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
    threading.Thread(target=_background, args=(port,),
                     daemon=True).start()
    try:
        from streamlit.web import cli as st_cli
        sys.argv = ["streamlit", "run", app_path]
        st_cli.main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        return 1
    print("launcher: server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
