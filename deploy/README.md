# Deploying the image AI service (Kardlyz Background Remover)

Target: the Kardlyz VPS (Ubuntu 22.04, Python 3.10, Apache 2.4, 4 vCPU / 16 GB, no GPU). The box also runs the live Kardlyz API (`:4000`), kardlyz-worker, ces-backend-v2 and keryar-api, so this service is fenced in.

| | |
| --- | --- |
| Code | `/home/ubuntu/kardlyz-image-ai`, a clone of `https://github.com/ptlshubham/format-converter` (`main`) |
| Process | PM2 app `kardlyz-image-ai` ([`ecosystem.config.js`](ecosystem.config.js)), one uvicorn worker on `127.0.0.1:8110` |
| CPU / priority | all 4 cores at the lowest priority (`nice -n 19`): the API and the other apps always come first |
| Memory | above 6.7 GB peak per image on this box (Linux, onnxruntime 1.23); PM2 restarts it above 10 GB; models unload after 90 s idle |
| Public URL | `https://api.kardlyz.com/image-ai/api/background-remover` (Apache strips `/image-ai/`, [`apache-image-ai.conf`](apache-image-ai.conf)) |
| Callers | `https://kardlyz.com`, `https://www.kardlyz.com`, and `http://localhost:4200` for the owner's local testing |

Rule: **local first, then server.** Change and test on a laptop, push to GitHub, then pull here. Never edit files on the VPS.

## Settings (environment variables)

All are set in `ecosystem.config.js`. With none set, the service runs in development mode, as on a laptop.

| Variable | Production value | Default | What it does |
| --- | --- | --- | --- |
| `IMAGE_AI_ENV` | `production` | `development` | `production` turns off `/docs`, `/redoc` and `/openapi.json`, and does not mount `/status`, `/debug/latest`, `/diagnostics` or `/composite` (all 404) |
| `IMAGE_AI_ALLOWED_ORIGINS` | `https://kardlyz.com,https://www.kardlyz.com,http://localhost:4200` | `http://localhost:4200,http://127.0.0.1:4200` | Browser origins CORS allows. `*` is refused at startup |
| `IMAGE_AI_RATE_LIMIT_PER_MINUTE` | `6` | `6` | `POST /remove` plus `POST /export-hd` per client IP per minute, shared. Over it: 429 with `Retry-After`. `0` turns it off |
| `IMAGE_AI_MAX_UPLOAD_MB` | `26` | `26` | Largest body for `/remove` (multipart). Over it: 413 |
| `IMAGE_AI_MAX_JSON_MB` | `20` | `20` | Largest body for `/export-hd` (JSON). Over it: 413 |
| `IMAGE_AI_PROFILER_LOG` | `off` | `on` in development, `off` in production | Per-request profiler JSON files in `backend/tests/diagnostic_results/performance/` (sizes and timings, never pixels) |
| `IMAGE_AI_TRUSTED_PROXIES` | `127.0.0.1,::1` | `127.0.0.1,::1` | Peers whose `X-Forwarded-For` is believed when working out the client IP |
| `BG_MODEL_IDLE_TIMEOUT`, `BG_SR_IDLE_TIMEOUT` | unset (90 s) | `90` | Seconds before an idle model is unloaded. Keep the default on this shared box |
| `DEBUG_BACKGROUND_REMOVAL` | unset | off | Never set it in production |

No thread-count variables are set: ONNX Runtime sets its own 4 threads in code, and `nice -n 19` keeps them behind every other app.

## First install

Run as `ubuntu`. Steps 7–8 need `sudo`.

1. **Check the box.**
   ```bash
   python3 --version                         # 3.10.x
   python3 -m venv --help >/dev/null || sudo apt install -y python3.10-venv
   command -v nice                           # /usr/bin/nice
   ss -ltnp | grep ':8110 ' || echo "8110 is free"
   free -h                                   # needs ~4 GB available
   df -h /home                               # ~2 GB for the venv and models
   pm2 ls                                    # note what is running; leave it alone
   ```

2. **Clone over HTTPS** (the repo is public, so no key is needed; this avoids the box's SSH deploy-key setup). Git LFS is not installed here, so the `.onnx` files arrive as small pointer files; step 4 replaces them.
   ```bash
   cd /home/ubuntu
   GIT_LFS_SKIP_SMUDGE=1 git clone --branch main https://github.com/ptlshubham/format-converter.git kardlyz-image-ai
   cd kardlyz-image-ai
   ```

3. **Create the venv and install packages** (CPU `onnxruntime` only; never add `onnxruntime-gpu`).
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -r backend/requirements.txt
   ```

4. **Fetch the models** (about 557 MB from GitHub's LFS media URLs, each checked against its SHA-256). Then confirm.
   ```bash
   .venv/bin/python backend/models/download_models.py
   .venv/bin/python backend/models/download_models.py --check   # three [OK] lines, exit 0
   ```
   Afterwards `git status` lists the three `.onnx` files as modified. That is expected: they are now the real files, not the pointers. Do not `git checkout .` or `git reset --hard` here, because that puts the pointers back. If it happens, re-run the fetch. Downloads count against the GitHub account's LFS bandwidth, so fetch once and keep the files.

5. **Check that the app imports in production mode.**
   ```bash
   IMAGE_AI_ENV=production .venv/bin/python -c "import backend.background_remover.main" && echo OK
   ```

6. **Start it under PM2.**
   ```bash
   mkdir -p logs
   pm2 start deploy/ecosystem.config.js
   pm2 save                                  # saves the whole current list; check `pm2 ls` first
   pm2 logs kardlyz-image-ai --lines 30      # look for "Image AI settings: env=production, ..."
   curl -s http://127.0.0.1:8110/api/background-remover/health    # {"status":"online",...}
   ps -o pid,ni,rss,cmd -p $(pm2 pid kardlyz-image-ai)   # NI 19
   ```
   Never run a bare `pm2 start`, and never change the other apps' PM2 entries.

7. **Add the Apache lines.** Back up the vhost, paste the contents of `deploy/apache-image-ai.conf` inside the `:443` vhost for `api.kardlyz.com`, **above** `ProxyPass / http://127.0.0.1:4000/`, then test and reload.
   ```bash
   VHOST=$(grep -l "ProxyPass / http://127.0.0.1:4000/" /etc/apache2/sites-enabled/*); echo "$VHOST"
   sudo cp "$(readlink -f "$VHOST")" ~/api.kardlyz.com.conf.bak-$(date +%F)
   sudo nano "$(readlink -f "$VHOST")"
   sudo apache2ctl configtest                # must say "Syntax OK"; if not, fix or restore the backup
   sudo systemctl reload apache2
   ```

8. **Smoke test from outside** (your laptop or the VPS).
   ```bash
   B=https://api.kardlyz.com/image-ai
   A=$B/api/background-remover

   curl -s $A/health                                          # {"status":"online",...}
   curl -s -o /dev/null -w "%{http_code}\n" https://api.kardlyz.com/health   # the Kardlyz API still answers

   # CORS preflight from the site: 200 + access-control-allow-origin: https://kardlyz.com
   curl -si -X OPTIONS $A/remove -H "Origin: https://kardlyz.com" \
     -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: content-type" | grep -iE "^HTTP|^access-control-allow-origin"
   # ...and from a stranger: 400, and NO access-control-allow-origin line
   curl -si -X OPTIONS $A/remove -H "Origin: https://evil.example" \
     -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: content-type" | grep -iE "^HTTP|^access-control-allow-origin"

   # Hidden in production: each prints 404
   for p in /docs /redoc /openapi.json /api/background-remover/debug/latest /api/background-remover/status; do
     curl -s -o /dev/null -w "$p %{http_code}\n" $B$p; done
   for p in diagnostics composite; do curl -s -o /dev/null -w "/$p %{http_code}\n" -X POST $A/$p; done

   # Too large: 413 (Apache answers first above 26 MB)
   head -c 30000000 /dev/zero | curl -s -o /dev/null -w "%{http_code}\n" -X POST \
     -H "Content-Type: application/octet-stream" --data-binary @- $A/remove

   # Rate limit: no file, so 422 without loading a model, then 429 on the 7th.
   # This uses up your IP's budget for a minute.
   for i in 1 2 3 4 5 6 7; do curl -s -o /dev/null -w "%{http_code} " -X POST -F sensitivity=10 $A/remove; done; echo
   ```
   Then on the VPS run `pm2 logs kardlyz-image-ai --lines 20`. The access lines must show **your** public IP, not `127.0.0.1`. If they show `127.0.0.1`, the proxy headers are not getting through, and every visitor would share one rate-limit budget.

   Last, one real photo, from the Kardlyz page (KZ-IMG-10) or with `curl -F file=@photo.jpg $A/remove -o /dev/null -w "%{http_code} %{time_total}s\n"`. Expect a 200. The benchmarks say 60–75 s plus about 15 s for a cold model load, but those ran on 4+ cores and this service gets 2, so it may take longer. Note the time: the page gives up on `/remove` after 180 s (`REMOVE_TIMEOUT_MS`), so a time near that means the page timeout must go up or the service needs more cores. Watch `pm2 monit` or `free -h` while it runs.

9. **Point the website at it.** In Kardlyz-Frontend, set `imageAiUrl: 'https://api.kardlyz.com/image-ai/api/background-remover'` in `src/environments/environment.ts`, then build and deploy the static site.

## Updating

```bash
cd /home/ubuntu/kardlyz-image-ai
git fetch origin
git log --oneline HEAD..origin/main                       # what is coming
git diff --stat HEAD origin/main -- backend/requirements.txt deploy/
git pull --ff-only
.venv/bin/pip install -r backend/requirements.txt         # only if requirements.txt changed
.venv/bin/python backend/models/download_models.py        # no-op unless a model changed
pm2 restart kardlyz-image-ai                              # code-only change
# If deploy/ecosystem.config.js changed, restart takes no notice of it. Instead:
#   pm2 delete kardlyz-image-ai && pm2 start deploy/ecosystem.config.js && pm2 save
curl -s http://127.0.0.1:8110/api/background-remover/health
```
A restart cuts off any image being processed, so pick a quiet moment. If `git pull` stops with *"local changes … would be overwritten"* on a `.onnx` file, a model has changed upstream. Run `git checkout -- <that file>`, pull again, then run the fetch script, whose hash list will have been updated in the same commit.

If `deploy/apache-image-ai.conf` changed, re-apply it by hand (step 7). It is never read from the repo.

## Rollback

- **Switch the feature off, fast:** `pm2 stop kardlyz-image-ai`. Apache then returns 503 for `/image-ai/` and the page says the service can't be reached. The Kardlyz API is not affected. Bring it back with `pm2 start kardlyz-image-ai`.
- **Hide it on the website:** set `imageAiUrl: ''` in Kardlyz-Frontend `environment.ts` and redeploy. The page shows "Coming soon" and sends nothing.
- **Previous code:** `git log --oneline -5`, then `git checkout <good-sha>` and `pm2 restart kardlyz-image-ai`. Run `git checkout main` when a fix has landed.
- **Apache:** restore `~/api.kardlyz.com.conf.bak-<date>` over the vhost file, run `sudo apache2ctl configtest`, then `sudo systemctl reload apache2`.
- **Remove completely:** `pm2 delete kardlyz-image-ai && pm2 save`, remove the Apache lines (configtest, reload), then `rm -rf /home/ubuntu/kardlyz-image-ai`.

## Notes

- **One worker, always.** Each worker would load its own copy of the model and its own inference lock, and the rate limiter is in-process.
- **The CPU limit is a priority, not a pin.** ONNX Runtime starts 4 inference threads (`min(4, os.cpu_count())` in `segmentation/model_manager.py`, and 4 in `processing/super_resolution.py`; both files are protected, see `docs/ENGINE_FREEZE_RULES.md`). A first setup pinned them to 2 cores with `taskset`, and one image then ran past 150 s. At `nice -n 19` on all 4 cores, a 518x512 portrait took 99 s end to end on 13 Oct 2026, and the Kardlyz API's slowest of 49 probes during it was 73 ms.
- **PM2's `max_memory_restart` is checked periodically.** It is a backstop, not a hard limit. If the box gets tight, add swap or move the service to its own VPS.
- **Limits come in three layers.** The page refuses files over 25 MB. Apache refuses bodies over 26 MB on `/image-ai/`, with an HTML 413 that the page reports as unreachable. The service refuses over 26 MB on `/remove` and over 20 MB on `/export-hd` with a JSON 413. A second image while one is processing gets 429 (*"currently processing another image"*), as does a client over the rate limit. The Kardlyz page shows both as "busy, try again in a minute".
- **Logs:** `logs/image-ai-out.log` and `logs/image-ai-error.log` in the repo folder (ignored by git). The engine prints a timing report for every request. If `pm2-logrotate` is not installed on the box, watch the log size.
