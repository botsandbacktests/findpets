"""
Passenger entry point for Namecheap cPanel "Setup Python App".

cPanel/Passenger speaks WSGI, but FastAPI is an ASGI app. a2wsgi wraps the
ASGI app so Passenger can serve it. Passenger looks for a module-level
`application` object in this file.

This version is self-healing: if a2wsgi isn't importable (e.g. the cPanel
"Run Pip Install" button installed into a different environment than the one
Passenger runs), it installs a2wsgi into THIS interpreter's environment on
startup, then imports it. It also makes sure this folder is on sys.path so
`import app.main` works regardless of the working directory Passenger uses.
"""
import os
import sys
import subprocess

# --- Make sure this folder (which contains the `app/` package) is importable.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# --- Ensure a2wsgi is available; install into THIS environment if missing.
try:
    from a2wsgi import ASGIMiddleware
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "a2wsgi==1.10.10"],
        check=False,
    )
    # Also try a normal (non --user) install in case --user isn't honored.
    try:
        from a2wsgi import ASGIMiddleware
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "a2wsgi==1.10.10"],
            check=False,
        )
        from a2wsgi import ASGIMiddleware

# --- Import the FastAPI ASGI app from app/main.py
from app.main import app as asgi_app

# --- Passenger will call this WSGI callable
application = ASGIMiddleware(asgi_app)
