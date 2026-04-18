"""EMSN Sonar Hardware Monitor.

Verzamelt systeemstatistieken (CPU, geheugen, disk, temperatuur) en schrijft
ze naar de centrale PostgreSQL ``system_health`` tabel. Zo verschijnt Sonar
in de ochtendmail en het Grafana dashboard naast Zolder/Berging/Meteo.

Draait elke minuut via ``sonar-hardware-monitor.timer``.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.core.secrets import get_pg_config

logger = logging.getLogger("sonar_hardware_monitor")

STATION = "sonar"


def _read_cpu_temp() -> float | None:
    """Lees CPU temperatuur via /sys of vcgencmd fallback."""
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal.exists():
        try:
            return round(int(thermal.read_text().strip()) / 1000.0, 1)
        except (ValueError, OSError):
            pass

    try:
        result = subprocess.run(
            ["/usr/bin/vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and "=" in result.stdout:
            return float(result.stdout.split("=")[1].split("'")[0])
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    return None


def _ping_latency_ms(host: str = "192.168.1.25") -> int | None:
    """Meet ping latency naar NAS (is PostgreSQL host)."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "time=" in line:
                ms = line.split("time=")[1].split()[0]
                return int(float(ms))
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return None


def _service_status(service: str) -> str:
    """Return 'running' / 'stopped' / 'unknown' voor een systemd service."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            timeout=3,
        )
        out = result.stdout.strip()
        if out == "active":
            return "running"
        if out in ("inactive", "failed"):
            return "stopped"
        return "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _calc_health_score(cpu_temp: float | None, memory_pct: float, disk_pct: float, latency: int | None) -> int:
    """Simpele health score 0-100: aftrek per overschrijding."""
    score = 100
    if cpu_temp and cpu_temp > 80:
        score -= 30
    elif cpu_temp and cpu_temp > 70:
        score -= 15

    if memory_pct > 95:
        score -= 25
    elif memory_pct > 85:
        score -= 10

    if disk_pct > 90:
        score -= 25
    elif disk_pct > 80:
        score -= 10

    if latency is None:
        score -= 20
    elif latency > 200:
        score -= 10

    return max(0, score)


def collect_metrics() -> dict:
    """Verzamel alle hardware metrics."""
    cpu_pct = psutil.cpu_percent(interval=1)
    cpu_temp = _read_cpu_temp()

    mem = psutil.virtual_memory()
    memory_pct = mem.percent

    disk = psutil.disk_usage("/")
    disk_pct = round(disk.used / disk.total * 100, 2)

    latency = _ping_latency_ms()
    network_status = "good" if latency and latency < 100 else "degraded" if latency else "poor"

    # NAS bereikbaar?
    try:
        socket.create_connection(("192.168.1.25", 5433), timeout=3).close()
        db_status = "running"
    except OSError:
        db_status = "stopped"

    score = _calc_health_score(cpu_temp, memory_pct, disk_pct, latency)

    return {
        "timestamp": datetime.now(),
        "cpu_usage": round(cpu_pct, 2),
        "cpu_temp": cpu_temp,
        "memory_usage": round(memory_pct, 2),
        "memory_total": mem.total // (1024 * 1024),
        "memory_available": mem.available // (1024 * 1024),
        "disk_usage": disk_pct,
        "disk_total": disk.total,
        "disk_available": disk.free,
        "network_latency_ms": latency,
        "network_status": network_status,
        "birdnet_status": _service_status("sonar-monitor.service"),
        "mqtt_status": _service_status("sonar-stats-publisher.service"),
        "database_status": db_status,
        "overall_health_score": score,
    }


def save_to_postgres(metrics: dict) -> bool:
    """INSERT metrics in ``system_health``."""
    config = get_pg_config()
    try:
        with psycopg2.connect(
            host=config["host"],
            port=config["port"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            connect_timeout=10,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO system_health (
                        station, measurement_timestamp,
                        cpu_usage, cpu_temp,
                        memory_usage, memory_total, memory_available,
                        disk_usage, disk_total, disk_available,
                        network_latency_ms, network_status,
                        birdnet_status, mqtt_status, database_status,
                        overall_health_score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        STATION,
                        metrics["timestamp"],
                        metrics["cpu_usage"],
                        metrics["cpu_temp"],
                        metrics["memory_usage"],
                        metrics["memory_total"],
                        metrics["memory_available"],
                        metrics["disk_usage"],
                        metrics["disk_total"],
                        metrics["disk_available"],
                        metrics["network_latency_ms"],
                        metrics["network_status"],
                        metrics["birdnet_status"],
                        metrics["mqtt_status"],
                        metrics["database_status"],
                        metrics["overall_health_score"],
                    ),
                )
        return True
    except psycopg2.Error as exc:
        logger.error("PostgreSQL INSERT mislukt: %s", exc)
        return False


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    metrics = collect_metrics()
    logger.info(
        "cpu=%s%% temp=%s°C mem=%s%% disk=%s%% score=%s",
        metrics["cpu_usage"],
        metrics["cpu_temp"],
        metrics["memory_usage"],
        metrics["disk_usage"],
        metrics["overall_health_score"],
    )
    return 0 if save_to_postgres(metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
