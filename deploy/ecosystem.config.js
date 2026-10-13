/**
 * PM2 definition for the Kardlyz image AI service (Background Remover) on the
 * Kardlyz VPS, which it shares with the live Kardlyz API, kardlyz-worker,
 * ces-backend-v2 and keryar-api. See deploy/README.md.
 *
 * Run from /home/ubuntu/kardlyz-image-ai:
 *   pm2 start deploy/ecosystem.config.js && pm2 save
 * Never a bare `pm2 start`, and never touch the other apps' PM2 entries.
 *
 * Sharing the box:
 * - CPU: all four vCPUs at the lowest priority (`nice -n 19`), so the API, the
 *   worker and CES always win the CPU when they want it and the inference only
 *   uses what they leave idle. Measured 13 Oct 2026: a 518x512 portrait took
 *   99 s end to end while the Kardlyz API answered 49 probes in at most 73 ms.
 *   The first setup, `taskset -c 2,3` + `nice -n 10`, was dropped: ONNX
 *   Runtime still starts 4 threads (frozen model_manager.py), so they shared 2
 *   cores and one image ran past 150 s, beyond the page's timeout. `nice`
 *   execs uvicorn, so PM2 still tracks the right PID.
 * - Memory: on this box (Linux, onnxruntime 1.23 on Python 3.10) one inference
 *   went above 6.7 GB RSS, not the ~3.5 GB measured on Windows. PM2 restarts
 *   the service above 10 GB (checked every 30 s or so: a backstop, not a hard
 *   cap; the box has 16 GB and its other apps use under 2 GB). Idle
 *   models unload after 90 s (BG_MODEL_IDLE_TIMEOUT / BG_SR_IDLE_TIMEOUT), which
 *   gives the memory back between visitors; keep that on a shared box.
 * - One worker only: each worker would load its own model copy and its own
 *   inference lock, and the rate limiter lives in the process.
 */
module.exports = {
  apps: [
    {
      name: "kardlyz-image-ai",
      cwd: "/home/ubuntu/kardlyz-image-ai",
      script: "/usr/bin/nice",
      args: [
        "-n", "19",
        ".venv/bin/uvicorn", "backend.background_remover.main:app",
        "--host", "127.0.0.1",
        "--port", "8110",
        "--workers", "1",
        // Apache on this box is the only trusted proxy: believe its X-Forwarded-For.
        "--proxy-headers",
        "--forwarded-allow-ips", "127.0.0.1",
      ],
      interpreter: "none",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      max_memory_restart: "10G",
      // Give uvicorn time to shut down and unload the models before SIGKILL.
      kill_timeout: 10000,
      env: {
        IMAGE_AI_ENV: "production",
        IMAGE_AI_ALLOWED_ORIGINS: "https://kardlyz.com,https://www.kardlyz.com,http://localhost:4200",
        IMAGE_AI_PROFILER_LOG: "off",
        // Defaults, spelled out so they are easy to find and change:
        IMAGE_AI_RATE_LIMIT_PER_MINUTE: "6",
        IMAGE_AI_MAX_UPLOAD_MB: "26",
        IMAGE_AI_MAX_JSON_MB: "20",
        IMAGE_AI_TRUSTED_PROXIES: "127.0.0.1,::1",
        PYTHONUNBUFFERED: "1",
      },
      error_file: "logs/image-ai-error.log",
      out_file: "logs/image-ai-out.log",
      merge_logs: true,
      time: true,
    },
  ],
};
