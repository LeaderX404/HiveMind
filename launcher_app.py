"""HiveMind launcher logic (issues #41, #42, #44, #46, wave #47).

Shipped as a DATA file next to the exe: the onedir delta-update
architecture (wave #47) patches this file in kilobytes instead
of rebuilding the executable.

The MAIN THREAD runs the Streamlit server and nothing else
(issue #46: Streamlit registers signal handlers, and Python
allows that only in the main thread). The browser, the tray
icon, the one-time core bootstrap and the delta-update cycle
all live in daemon threads or run before the server starts.
The build is windowed - no console - so stdout and stderr are
redirected to a log file first (issue #44: launcher.log sits
next to the exe; the Debug build prints to its console).

Wave #47: the heavy libraries (torch, transformers) live in
core\ beside the app dir, downloaded once and verified; the
app checks the public manifest and applies verified delta
patches before starting the server.
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


def _app_root():
    """The install root: .../HiveMind/app (frozen) or the repo."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _core_dir():
    """core\ lives beside the app dir: Programs/HiveMind/core."""
    root = _app_root()
    if getattr(sys, "frozen", False):
        return os.path.normpath(os.path.join(root, "..", "core"))
    return os.path.join(root, "core")


def _cleanup_old_build():
    """Delete leftover .old executables from a self-update."""
    root = _app_root()
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        if name.endswith(".old"):
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass


def _ensure_core():
    """Heavy libraries live in core\ (wave #47): put that dir
    on sys.path, downloading the core once if it is missing."""
    core = _core_dir()
    if os.path.isdir(os.path.join(core, "torch")):
        sys.path.insert(0, core)
        return True
    print("launcher: core missing at %s" % core)
    try:
        from hivemind.ui import updater
        if updater.ensure_core(core):
            sys.path.insert(0, core)
            print("launcher: core ready")
            return True
    except Exception as exc:
        print("launcher: core bootstrap failed: %s" % exc)
    return False


def _apply_updates():
    """Delta updates (wave #47): verify, patch, relaunch. A
    failure never blocks the current build from running."""
    try:
        from hivemind.ui import updater
        if updater.check_and_apply(_app_root()):
            print("launcher: update applied - relaunching")
            os.execv(sys.executable, [sys.executable])
    except Exception as exc:
        print("launcher: update cycle failed: %s" % exc)


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
    browser and the tray run in background threads, and the
    core bootstrap + delta update run before the server (wave
    #47)."""
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
    _cleanup_old_build()
    if not _ensure_core():
        print("launcher: continuing without the core - model "
              "features will fail until it is installed")
    _apply_updates()
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
