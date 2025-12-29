import os
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Deque, Dict, Any

import board
import busio
import adafruit_scd4x

from tapo import ApiClient  # pip/uv install tapo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
import uvicorn

import threading
import time

import cv2


# ------------------------- env helpers -------------------------

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)

def _env_int0(name: str, default: int) -> int:
    """Parse int from env, accepting base prefixes like 0x.. (base=0)."""
    v = os.getenv(name)
    return default if v is None else int(v, 0)

def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


# ------------------------- logging (also to web) -------------------------

class DequeLogHandler(logging.Handler):
    def __init__(self, buf: Deque[str]):
        super().__init__()
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.buf.append(msg)
        except Exception:
            pass


# ------------------------- shared runtime state -------------------------

@dataclass
class RuntimeState:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: Optional[datetime] = None

    co2: Optional[int] = None
    temp_c: Optional[float] = None
    rh: Optional[float] = None

    # thresholds
    co2_on: int = 1000
    co2_off: int = 800

    rh_on: float = 85.0
    rh_off: float = 95.0

    # trigger states
    co2_active: bool = False
    rh_active: bool = False

    # tapo config summary
    shared_device: bool = False
    model_co2: str = ""
    ip_co2: str = ""
    model_rh: str = ""
    ip_rh: str = ""

    # last commanded states
    last_shared_state: Optional[bool] = None
    last_co2_state: Optional[bool] = None
    last_rh_state: Optional[bool] = None

    # last error (shown in UI)
    last_error: Optional[str] = None

    # ---- tapo retry/backoff ----
    tapo_next_attempt_at: float = 0.0  # time.monotonic() seconds
    tapo_backoff_s: float = 0.0
    tapo_min_interval_s: float = 10.0
    tapo_backoff_base_s: float = 10.0
    tapo_backoff_max_s: float = 300.0

    # ---- camera (RTSP -> MJPEG) ----
    cam_rtsp_url: str = ""
    cam_latest_jpeg: Optional[bytes] = None
    cam_last_frame: Optional[datetime] = None
    cam_running: bool = False



# ------------------------- tapo helper -------------------------

async def _connect_device(client: ApiClient, model: str, ip: str):
    """
    model is the method name on ApiClient, e.g. 'p110', 'p115', 'l530', etc.
    """
    factory = getattr(client, model, None)
    if factory is None:
        raise ValueError(f"Unsupported TAPO_MODEL '{model}' (no ApiClient.{model}(...))")
    return await factory(ip)


def start_camera_thread(state: RuntimeState, log: logging.Logger) -> None:
    rtsp = _env_str("CAM_RTSP_URL", "")
    if not rtsp:
        log.info("Camera: CAM_RTSP_URL not set; camera disabled")
        return

    fps = _env_int("CAM_FPS", 5)
    quality = _env_int("CAM_JPEG_QUALITY", 80)
    max_w = _env_int("CAM_MAX_WIDTH", 960)

    state.cam_rtsp_url = rtsp
    state.cam_running = True

    stop_evt = threading.Event()

    def worker():
        # Use TCP for RTSP if your network is flaky (optional):
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        cap = None
        last_ok = time.time()

        while not stop_evt.is_set():
            try:
                if cap is None or not cap.isOpened():
                    log.info("Camera: connecting to RTSP...")
                    cap = cv2.VideoCapture(rtsp)
                    # give it a moment
                    time.sleep(0.3)

                ok, frame = cap.read()
                if not ok or frame is None:
                    # reconnect if stalled
                    if time.time() - last_ok > 5:
                        log.warning("Camera: no frames, reconnecting...")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    time.sleep(0.2)
                    continue

                last_ok = time.time()

                # downscale (optional)
                if max_w and frame.shape[1] > max_w:
                    h, w = frame.shape[:2]
                    new_h = int(h * (max_w / w))
                    frame = cv2.resize(frame, (max_w, new_h))

                # encode jpeg
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    state.cam_latest_jpeg = buf.tobytes()
                    state.cam_last_frame = datetime.now(timezone.utc)

                # throttle capture rate a bit
                time.sleep(max(0.0, 1.0 / max(1, fps)))

            except Exception as e:
                log.warning("Camera worker error: %s", e)
                time.sleep(1.0)

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        log.info("Camera: stopped")

    t = threading.Thread(target=worker, daemon=True, name="camera_rtsp_worker")
    t.start()

    # attach stop handle to state (simple attribute)
    state._cam_stop_evt = stop_evt  # type: ignore[attr-defined]

# ------------------------- web app -------------------------

def build_app(state: RuntimeState, logbuf: Deque[str]) -> FastAPI:
    app = FastAPI(title="Mushroom Monitor")
    @app.get("/api/cam.jpg")
    async def cam_jpg():
        if not state.cam_latest_jpeg:
            return PlainTextResponse("No frame yet", status_code=503)
        return Response(content=state.cam_latest_jpeg, media_type="image/jpeg")


    @app.get("/api/cam.mjpg")
    async def cam_mjpg():
        if not state.cam_rtsp_url:
            return PlainTextResponse("Camera disabled (CAM_RTSP_URL not set)", status_code=404)

        async def gen():
            boundary = b"frame"
            while True:
                jpg = state.cam_latest_jpeg
                if not jpg:
                    await asyncio.sleep(0.2)
                    continue

                yield b"--" + boundary + b"\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                yield jpg
                yield b"\r\n"

                await asyncio.sleep(0.05)  # tiny pause for browser friendliness

        return StreamingResponse(
            gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache"},
        )


    @app.get("/", response_class=HTMLResponse)
    async def home():
        # Minimal dashboard that polls /api/status and /api/logs
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mushroom Monitor</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 12px; }}
    .k {{ color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    .v {{ font-size: 28px; margin-top: 6px; }}
    pre {{ background: #0b0b0b; color: #d6d6d6; padding: 12px; border-radius: 12px; overflow: auto; }}
    .row {{ display:flex; gap: 12px; flex-wrap: wrap; }}
    .pill {{ display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid #ddd; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>Mushroom Monitor</h2>
  <div class="row">
    <span class="pill" id="uptime">Uptime: …</span>
    <span class="pill" id="last_update">Last update: …</span>
    <span class="pill" id="tapo_state">Tapo: …</span>
    <span class="pill" id="error">Error: …</span>
  </div>

  <div class="grid" style="margin-top: 12px;">
    <div class="card">
      <div class="k">CO₂</div>
      <div class="v" id="co2">—</div>
      <div class="k">Trigger</div>
      <div id="co2_trigger">—</div>
      <div class="k">Thresholds</div>
      <div id="co2_thresh">—</div>
    </div>

    <div class="card">
      <div class="k">Humidity</div>
      <div class="v" id="rh">—</div>
      <div class="k">Trigger</div>
      <div id="rh_trigger">—</div>
      <div class="k">Mode / Thresholds</div>
      <div id="rh_thresh">—</div>
    </div>

    <div class="card">
      <div class="k">Temperature</div>
      <div class="v" id="temp">—</div>
    </div>
  </div>

    <div class="card" style="grid-column: 1 / -1;">
        <div class="k">Camera</div>
        <img src="/api/cam.mjpg" style="width:100%; border-radius: 12px; margin-top: 8px;" />
    </div>


  <h3 style="margin-top: 18px;">Logs (tail)</h3>
  <pre id="logs">Loading…</pre>

<script>
function fmt(dt) {{
  if (!dt) return "—";
  try {{ return new Date(dt).toLocaleString(); }} catch {{ return dt; }}
}}

async function refresh() {{
  const s = await fetch("/api/status").then(r => r.json());
  document.getElementById("co2").textContent = (s.co2 ?? "—") + (s.co2 != null ? " ppm" : "");
  document.getElementById("rh").textContent  = (s.rh ?? "—") + (s.rh != null ? " %" : "");
  document.getElementById("temp").textContent= (s.temp_c ?? "—") + (s.temp_c != null ? " °C" : "");

  document.getElementById("co2_trigger").textContent = s.co2_active ? "ACTIVE" : "inactive";
  document.getElementById("rh_trigger").textContent  = s.rh_active ? "ACTIVE" : "inactive";

  document.getElementById("co2_thresh").textContent = `Fan: ON ≥ ${{s.co2_on}} ppm, OFF ≤ ${{s.co2_off}} ppm`;
  document.getElementById("rh_thresh").textContent  = `Humidifier: ON < ${{s.rh_on}}%, OFF > ${{s.rh_off}}%`;

  document.getElementById("uptime").textContent = "Uptime: " + (s.uptime ?? "—");
  document.getElementById("last_update").textContent = "Last update: " + fmt(s.last_update);
  document.getElementById("tapo_state").textContent = "Tapo: " + (s.tapo_summary ?? "—");
  document.getElementById("error").textContent = "Error: " + (s.last_error ?? "none");

  const logs = await fetch("/api/logs").then(r => r.text());
  document.getElementById("logs").textContent = logs || "(no logs yet)";
}}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""

    @app.get("/api/status")
    async def status() -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        uptime_s = int((now - state.started_at).total_seconds())
        if uptime_s < 60:
            uptime = f"{uptime_s}s"
        elif uptime_s < 3600:
            uptime = f"{uptime_s//60}m {uptime_s%60}s"
        else:
            uptime = f"{uptime_s//3600}h {(uptime_s%3600)//60}m"

        # best-effort tapo summary
        if state.shared_device:
            desired = (state.co2_active or state.rh_active)
            tapo_summary = f"shared desired={'ON' if desired else 'OFF'} last={state.last_shared_state}"
        else:
            tapo_summary = f"co2={'ON' if state.co2_active else 'OFF'} last={state.last_co2_state}; rh={'ON' if state.rh_active else 'OFF'} last={state.last_rh_state}"

        return {
            "co2": state.co2,
            "temp_c": None if state.temp_c is None else round(state.temp_c, 1),
            "rh": None if state.rh is None else round(state.rh, 1),
            "last_update": None if state.last_update is None else state.last_update.isoformat(),
            "uptime": uptime,

            "co2_on": state.co2_on,
            "co2_off": state.co2_off,
            "rh_on": state.rh_on,
            "rh_off": state.rh_off,

            "co2_active": state.co2_active,
            "rh_active": state.rh_active,

            "tapo_summary": tapo_summary,
            "last_error": state.last_error,
        }

    @app.get("/api/logs", response_class=PlainTextResponse)
    async def logs():
        return "\n".join(logbuf)

    return app



# ------------------------- sensor + control loop -------------------------

async def run_sensor_loop(state: RuntimeState, log: logging.Logger):
    # -------- thresholds with hysteresis --------
    state.co2_on = _env_int("CO2_ON_PPM", 1000)
    state.co2_off = _env_int("CO2_OFF_PPM", state.co2_on - 200)

    # Humidifier hysteresis:
    # - ON when RH < RH_ON
    # - OFF when RH > RH_OFF
    state.rh_on = _env_float("RH_ON", 85.0)
    state.rh_off = _env_float("RH_OFF", 95.0)

    # Safety: ensure correct ordering (humidifier requires rh_on < rh_off)
    if state.rh_on >= state.rh_off:
        log.warning("RH_ON must be < RH_OFF for humidifier. Forcing RH_ON=85, RH_OFF=95.")
        state.rh_on = 85.0
        state.rh_off = 95.0

    # Basic validation
    if state.co2_off >= state.co2_on:
        log.warning("CO2_OFF_PPM should be < CO2_ON_PPM (got off=%s on=%s).", state.co2_off, state.co2_on)

    # -------- tapo retry/backoff config --------
    state.tapo_min_interval_s = _env_float("TAPO_MIN_COMMAND_INTERVAL_S", 10.0)
    state.tapo_backoff_base_s = _env_float("TAPO_BACKOFF_BASE_S", 10.0)
    state.tapo_backoff_max_s = _env_float("TAPO_BACKOFF_MAX_S", 300.0)

    # -------- tapo config --------
    username = _env_str("TAPO_USERNAME")
    password = _env_str("TAPO_PASSWORD")

    default_model = _env_str("TAPO_MODEL")
    default_ip = _env_str("TAPO_IP")

    state.model_co2 = _env_str("TAPO_MODEL_CO2", default_model)
    state.ip_co2 = _env_str("TAPO_IP_CO2", default_ip)

    state.model_rh = _env_str("TAPO_MODEL_RH", default_model)
    state.ip_rh = _env_str("TAPO_IP_RH", default_ip)

    if not (username and password):
        state.last_error = "Missing TAPO_USERNAME/TAPO_PASSWORD"
        log.error(state.last_error)

    # -------- init I2C + sensor (with retry + diagnostics) --------
    scd4x_i2c_addr = _env_int0("SCD4X_I2C_ADDRESS", 0x62)
    sensor_retry_s = _env_float("SCD4X_RETRY_S", 30.0)

    i2c = None
    scd4x = None
    next_sensor_attempt_at = 0.0
    sensor_init_logged_stack = False

    def _i2c_scan(addrs_i2c) -> str:
        try:
            if hasattr(addrs_i2c, "try_lock") and hasattr(addrs_i2c, "unlock") and hasattr(addrs_i2c, "scan"):
                got = []
                if addrs_i2c.try_lock():
                    try:
                        got = list(addrs_i2c.scan())
                    finally:
                        addrs_i2c.unlock()
                if got:
                    return ", ".join(hex(a) for a in got)
                return "(no devices found)"
        except Exception:
            pass
        return "(scan unavailable)"

    def _sensor_hint_text(i2c_scan: str) -> str:
        return (
            "Troubleshooting: ensure the sensor is powered (3.3V), wired to SDA/SCL correctly, "
            "I2C is enabled (raspi-config), and try i2cdetect. "
            f"Expected SCD4X at 0x{scd4x_i2c_addr:02x}; I2C scan saw: {i2c_scan}. "
            "You can override the address with SCD4X_I2C_ADDRESS=0x62."
        )

    async def ensure_sensor_connected(now_mono: float) -> None:
        nonlocal i2c, scd4x, next_sensor_attempt_at, sensor_init_logged_stack

        if scd4x is not None:
            return
        if now_mono < next_sensor_attempt_at:
            return

        next_sensor_attempt_at = now_mono + max(5.0, sensor_retry_s)

        if i2c is None:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
            except AttributeError:
                state.last_error = "board.SCL/board.SDA not available; cannot init I2C"
                log.error(state.last_error)
                return
            except Exception as e:
                state.last_error = f"Failed to init I2C: {e}"
                log.warning(state.last_error)
                return

        scan = _i2c_scan(i2c)
        try:
            # Newer adafruit_scd4x supports address=; fall back for older versions.
            try:
                scd4x = adafruit_scd4x.SCD4X(i2c, address=scd4x_i2c_addr)
            except TypeError:
                scd4x = adafruit_scd4x.SCD4X(i2c)
            try:
                log.info("Found SCD4x, serial: %s", [hex(i) for i in scd4x.serial_number])
            except Exception:
                log.info("Found SCD4x (could not read serial number)")
            scd4x.start_periodic_measurement()
            log.info("Measuring... (first reading may take ~5-10s)")
            state.last_error = None
        except Exception as e:
            state.last_error = f"Failed to initialize SCD4X: {e}"
            msg = state.last_error + ". " + _sensor_hint_text(scan)
            if not sensor_init_logged_stack:
                sensor_init_logged_stack = True
                log.exception(msg)
            else:
                log.warning(msg)
            scd4x = None

    # -------- connect tapo device(s) --------
    client = None
    device_shared = None
    device_co2 = None
    device_rh = None

    state.shared_device = (
        state.model_co2 == state.model_rh
        and state.ip_co2 == state.ip_rh
        and state.model_co2
        and state.ip_co2
    )

    async def ensure_tapo_connected():
        nonlocal client, device_shared, device_co2, device_rh
        if not (username and password):
            return
        if not ((state.model_co2 and state.ip_co2) or (state.model_rh and state.ip_rh)):
            return

        if client is None:
            client = ApiClient(username, password)

        try:
            if state.shared_device:
                if device_shared is None:
                    device_shared = await _connect_device(client, state.model_co2, state.ip_co2)
                    log.info("Connected to shared Tapo device %s@%s", state.model_co2, state.ip_co2)
            else:
                if device_co2 is None and state.model_co2 and state.ip_co2:
                    device_co2 = await _connect_device(client, state.model_co2, state.ip_co2)
                    log.info("Connected to CO2 Tapo device %s@%s", state.model_co2, state.ip_co2)
                if device_rh is None and state.model_rh and state.ip_rh:
                    device_rh = await _connect_device(client, state.model_rh, state.ip_rh)
                    log.info("Connected to RH Tapo device %s@%s", state.model_rh, state.ip_rh)
        except Exception as e:
            state.last_error = f"Failed to connect to Tapo device(s): {e}"
            log.warning(state.last_error)
            # allow retry later

    # initial connect attempt
    await ensure_tapo_connected()

    def _is_session_timeout(err: Exception) -> bool:
        name = err.__class__.__name__
        msg = str(err)
        return ("SessionTimeout" in name) or ("SessionTimeout" in msg)

    def _note_tapo_failure_backoff(now_mono: float) -> None:
        # Exponential backoff, capped.
        if state.tapo_backoff_s <= 0:
            state.tapo_backoff_s = max(1.0, state.tapo_backoff_base_s)
        else:
            state.tapo_backoff_s = min(state.tapo_backoff_max_s, state.tapo_backoff_s * 2)
        state.tapo_next_attempt_at = now_mono + state.tapo_backoff_s

    def _note_tapo_success(now_mono: float) -> None:
        state.tapo_backoff_s = 0.0
        state.tapo_next_attempt_at = now_mono + max(0.0, state.tapo_min_interval_s)

    try:
        while True:
            now_mono = time.monotonic()

            # periodically retry sensor init so wiring/config fixes can recover without restart
            await ensure_sensor_connected(now_mono)

            # retry tapo occasionally if not connected (also rate-limited by backoff)
            if now_mono >= state.tapo_next_attempt_at:
                if (device_shared is None and state.shared_device) or (
                    not state.shared_device and (device_co2 is None or device_rh is None)
                ):
                    await ensure_tapo_connected()

            if scd4x is not None and scd4x.data_ready:
                state.co2 = int(scd4x.CO2)
                state.temp_c = float(scd4x.temperature)
                state.rh = float(scd4x.relative_humidity)
                state.last_update = datetime.now(timezone.utc)

                log.info("CO2: %s ppm | Temp: %.1f C | RH: %.1f %%", state.co2, state.temp_c, state.rh)

                # ----- CO2 hysteresis (standard: ON above, OFF below) -----
                if not state.co2_active and state.co2 >= state.co2_on:
                    state.co2_active = True
                elif state.co2_active and state.co2 <= state.co2_off:
                    state.co2_active = False

                # ----- RH hysteresis (humidifier-only) -----
                # ON when too dry, OFF when humid enough
                if state.rh is not None:
                    if (not state.rh_active) and (state.rh < state.rh_on):
                        state.rh_active = True
                    elif state.rh_active and (state.rh > state.rh_off):
                        state.rh_active = False

                # ----- send commands (avoid spamming) -----
                now_mono = time.monotonic()
                if now_mono >= state.tapo_next_attempt_at:
                    attempted = False
                    any_succeeded = False
                    any_failed = False

                    if state.shared_device:
                        desired = state.co2_active or state.rh_active
                        if device_shared is not None and desired != state.last_shared_state:
                            attempted = True
                            try:
                                await (device_shared.on() if desired else device_shared.off())
                                log.info(
                                    "Tapo: %s (CO2/RH trigger %s)",
                                    "ON" if desired else "OFF",
                                    "active" if desired else "cleared",
                                )
                                state.last_shared_state = desired
                                any_succeeded = True
                            except Exception as e:
                                state.last_error = f"Tapo command failed: {e}"
                                log.warning(state.last_error)
                                any_failed = True
                                _note_tapo_failure_backoff(now_mono)
                                if _is_session_timeout(e):
                                    log.info("Tapo: session timeout; forcing reconnect")
                                    client = None
                                    device_shared = None

                    else:
                        if device_co2 is not None:
                            desired = state.co2_active
                            if desired != state.last_co2_state:
                                attempted = True
                                try:
                                    await (device_co2.on() if desired else device_co2.off())
                                    log.info("Tapo CO2 device: %s", "ON" if desired else "OFF")
                                    state.last_co2_state = desired
                                    any_succeeded = True
                                except Exception as e:
                                    state.last_error = f"Tapo command failed: {e}"
                                    log.warning(state.last_error)
                                    any_failed = True
                                    _note_tapo_failure_backoff(now_mono)
                                    if _is_session_timeout(e):
                                        log.info("Tapo CO2: session timeout; forcing reconnect")
                                        client = None
                                        device_co2 = None

                        if device_rh is not None:
                            desired = state.rh_active
                            if desired != state.last_rh_state:
                                attempted = True
                                try:
                                    await (device_rh.on() if desired else device_rh.off())
                                    log.info("Tapo RH device: %s", "ON" if desired else "OFF")
                                    state.last_rh_state = desired
                                    any_succeeded = True
                                except Exception as e:
                                    state.last_error = f"Tapo command failed: {e}"
                                    log.warning(state.last_error)
                                    any_failed = True
                                    _note_tapo_failure_backoff(now_mono)
                                    if _is_session_timeout(e):
                                        log.info("Tapo RH: session timeout; forcing reconnect")
                                        client = None
                                        device_rh = None

                    if any_failed and attempted and state.tapo_backoff_s <= 0:
                        _note_tapo_failure_backoff(now_mono)
                    elif any_succeeded and (not any_failed):
                        state.last_error = None
                        _note_tapo_success(now_mono)

            await asyncio.sleep(2)

    except asyncio.CancelledError:
        log.info("Sensor loop cancelled")
        raise
    except KeyboardInterrupt:
        log.info("Sensor loop interrupted by user")
    finally:
        if scd4x is not None:
            try:
                scd4x.stop_periodic_measurement()
            except Exception:
                pass


# ------------------------- main -------------------------

async def main_async():
    # in-memory log tail (shown in /api/logs)
    logbuf = deque(maxlen=_env_int("LOG_TAIL_LINES", 400))

    logger = logging.getLogger("mushroom")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    dh = DequeLogHandler(logbuf)
    dh.setFormatter(fmt)
    logger.addHandler(dh)

    state = RuntimeState()
    start_camera_thread(state, logger)

    # web server config
    web_host = _env_str("WEB_HOST", "0.0.0.0")
    web_port = _env_int("WEB_PORT", 8080)

    app = build_app(state, logbuf)

    # run both tasks
    sensor_task = asyncio.create_task(run_sensor_loop(state, logger), name="sensor_loop")

    uv_config = uvicorn.Config(app=app, host=web_host, port=web_port, log_level="warning")
    server = uvicorn.Server(uv_config)
    web_task = asyncio.create_task(server.serve(), name="web_server")

    logger.info("Web UI: http://%s:%s", web_host, web_port)

    try:
        done, pending = await asyncio.wait(
            {sensor_task, web_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        # if one task errors, stop the other
        for t in done:
            exc = t.exception()
            if exc:
                raise exc
    finally:
        # stop camera thread
        try:
            if hasattr(state, "_cam_stop_evt"):
                state._cam_stop_evt.set()  # type: ignore[attr-defined]
        except Exception:
            pass

        # try to stop uvicorn cleanly
        server.should_exit = True
        for t in (sensor_task, web_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(sensor_task, web_task, return_exceptions=True)


def main():
    print("Hello from mushroom!")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
