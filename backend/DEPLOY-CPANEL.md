# Deploy the FindMyPet backend on Namecheap cPanel (Setup Python App)

This runs your FastAPI backend on the hosting you already pay for. It uses
the fallback embedder (no torch), which fits shared-hosting limits.

> **Why the shim?** cPanel/Passenger speaks WSGI; FastAPI is ASGI. The included
> `passenger_wsgi.py` uses **a2wsgi** to bridge them. That file is already written.

---

## What you upload

From your `backend/` folder, you need:

```
backend/
  app/                  (the whole package: main.py, db.py, embedder.py, ...)
  requirements.txt      (now includes a2wsgi)
  passenger_wsgi.py     (the Passenger entry point — already created)
```

You do NOT need: the `__pycache__` folders, the local `data/` DB, or the venv.

---

> **This site has multiple sites in subfolders.** Everything for FindMyPet stays
> under `/findpets`. The API answers at **`tech956.com/findpets/api`** (a PATH,
> not a subdomain). The Python *code* still lives OUTSIDE `public_html` in its own
> app folder — that's normal and keeps your source private; it does not touch your
> other subfolder sites.

## Step 1 — Create the Python app in cPanel

1. cPanel → **Setup Python App** (the "Python" screen you saw) → **Create Application**.
2. Set:
   - **Python version:** 3.10 or newer (3.11/3.12 fine if offered).
   - **Application root:** `findpets-api`  (cPanel makes `/home/USER/findpets-api` —
     this is OUTSIDE public_html, so it doesn't affect your other sites)
   - **Application URL:** choose your main domain `tech956.com`, and in the path box
     next to it type **`findpets/api`**. Final URL = `tech956.com/findpets/api`.
   - **Application startup file:** `passenger_wsgi.py`
   - **Application Entry point:** `application`
3. Click **Create**. cPanel shows a command to activate the virtualenv — copy it
   (looks like `source /home/USER/virtualenv/findpets-api/3.x/bin/activate`).

> **Heads-up on the path box:** if cPanel won't let you type `findpets/api` because
> `findpets` already exists (it's your static site folder), that's fine — pick the
> path `findpets-api` at the top level instead, OR create the app at a plain path and
> we adjust `config.js` to match. Tell me exactly what cPanel accepts and I'll confirm
> the `config.js` value. The static site and the API are separate things sharing the
> `/findpets` name space, so cPanel may or may not allow a nested path — we'll adapt.

## Step 2 — Upload the backend files

Using cPanel **File Manager** (or FTP), upload into the Application root
(`/home/USER/findpets-api/`):

- the `app/` folder
- `requirements.txt`
- `passenger_wsgi.py`

(File Manager → open the folder → Upload. You can drag the whole `app` folder.)

## Step 3 — Install the packages

Back on the **Setup Python App** page, either:

- **Easy way:** in the app's panel there's a **"Run pip install"** box — type
  `requirements.txt` and run it; **or**
- **Terminal way:** cPanel → **Terminal**, paste the activate command from Step 1,
  then:
  ```
  pip install -r requirements.txt
  ```

This installs FastAPI, numpy, Pillow, a2wsgi, etc. (a minute or two).

## Step 4 — Set environment variables

In the Setup Python App panel, add these under **Environment variables**:

| Name | Value | Why |
|---|---|---|
| `SECRET_KEY` | a long random string | signs login tokens (don't leave the default) |
| `DATABASE_URL` | `sqlite:////home/USER/findpets-api/findmypet.db` | **absolute** local path — avoids the SQLite "disk I/O error" on synced dirs |
| `SQUARE_PAYMENT_LINK` | your Square link | already defaulted, override if it changes |

> Note the **four** slashes in the sqlite URL (`sqlite:////home/...`) — three for
> the scheme + one for the absolute path.

## Step 5 — Restart & test

1. Click **Restart** on the app.
2. Visit `https://tech956.com/findpets/api/api/health`.
   You should see JSON with `embedder` = the fallback name and `unlock_price_usd`.
   (Yes, `api/api` — cPanel serves the app under `/findpets/api`, and the health
   route inside the app is `/api/health`. If you'd rather not have the doubled
   `api`, tell me and I'll strip the `/api` prefix out of the routes in `main.py`.)
3. If you see a 500 / Passenger error, check **Step 6** below.

## Step 6 — Point the website at the backend

1. Open `petwebsite/config.js`.
2. Set it to your API URL (no trailing slash), e.g.:
   ```js
   window.FINDMYPET_BACKEND = "https://tech956.com/findpets/api";
   ```
3. Re-upload `config.js` to `public_html/findpets/`.
4. Reload `https://tech956.com/findpets/app.html` — the "offline" banner should
   disappear and the engine name should show.

---

## Troubleshooting

- **Passenger/500 error:** cPanel logs are in the app root as `stderr.log` /
  `passenger.log`. Most common cause = a package failed to install → re-run Step 3.
- **`ModuleNotFoundError: app`:** make sure `passenger_wsgi.py` sits in the SAME
  folder that contains the `app/` package, and Entry point is `application`.
- **`a2wsgi` not found:** it didn't install — activate the venv and
  `pip install a2wsgi==1.10.10`.
- **SQLite "disk I/O error":** your `DATABASE_URL` isn't a plain local path —
  use the absolute `sqlite:////home/USER/.../findmypet.db` form from Step 4.
- **Photos don't load in results:** confirm `config.js` points at the backend;
  the app now prepends that origin to `/photos/...` automatically.
- **CORS error in browser console:** the backend currently allows all origins
  (`allow_origins=["*"]`), so this shouldn't happen. To lock it down later, set
  it to `["https://tech956.com"]` in `app/main.py` and restart.

## If cPanel fights you (ASGI quirks)

Shared hosting + ASGI can be finicky. If after Step 5 you still get Passenger
errors you can't clear, the fallback is **Render free tier** — native FastAPI,
no shim. It runs on Render's servers (unrelated to Namecheap, so no ToS issue)
and your static site stays on Namecheap. Ask me and I'll write those steps.

## Keeping it within shared-hosting limits

- Stick with the **fallback embedder** (torch stays commented out in
  requirements.txt). The heavy DINOv2 engine will blow the memory cap.
- Low traffic (Nextdoor ads) is well within limits. Passenger idles the app
  when unused and wakes it on request — that's the intended pattern, not a
  persistent daemon, so it won't trip the "no long-running processes" rule.
