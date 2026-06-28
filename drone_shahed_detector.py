# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║          DRONE DETECTION SYSTEM — Windows + macOS                ║
║  Auteur : alexandre196 ( Alexandre Martin 2026 )                 ║
║                                                                  ║
║  INSTALLATION (première fois) :                                  ║
║    pip install ultralytics opencv-python numpy matplotlib        ║
║               reportlab Pillow                                   ║
║                                                                  ║
║  OPTIONNEL pour audio dans la vidéo :                            ║
║    Windows : winget install ffmpeg                               ║
║    macOS   : brew install ffmpeg                                 ║
║                                                                  ║
║  MODELE SHAHED :                                                 ║
║    runs/detect/shahed_detector/weights/best.pt                   ║
║    Classes : bird / not / shahed  (mAP@50 = 91.1%)               ║
╚══════════════════════════════════════════════════════════════════╝

FEATURES :
  ✔ YOLOv8 detection temps réel — modèle Shahed entraîné
  ✔ Kalman filter + multi-drone tracking
  ✔ Live preview (Tkinter canvas — macOS & Windows)
  ✔ Alarme sonore temps réel (Windows: winsound / macOS: silencieux)
  ✔ Audio alarme intégré dans la vidéo (ffmpeg)
  ✔ Géolocalisation estimée (sans GPS)
  ✔ Mini-carte radar live dans le GUI
  ✔ Export KML → Google Earth
  ✔ Export CSV → Excel / QGIS
  ✔ Rapport PDF avec stats, graphiques, carte, photo du drone
  ✔ Photo unique du drone dangereux le plus proche
  ✔ Calibration caméra par objet de référence
  ✔ Analyse comportementale (hovering, circling, approche, erratique)
  ✔ Alertes email + push Ntfy

CORRECTIONS v2 :
  ✔ pixel_to_azimuth : signe corrigé (droite image = est, pas ouest)
  ✔ export_kml : altitude estimée ajoutée + altitudeMode clampToGround
  ✔ export_kml : Placemark observateur avec altitudeMode

CORRECTIONS v3 — UNITÉS COHÉRENTES (MÈTRES) :
  ✔ TOUTES les distances sont désormais en MÈTRES (estimation, HUD,
    seuil danger, CSV, KML, PDF, emails, stats)
  ✔ Largeur réelle PAR CLASSE : Shahed-136 = 2,5 m d'envergure,
    fpv-drone = 0,40 m, bird = 0,50 m → distances enfin réalistes
  ✔ Seuil de danger en mètres (défaut 300 m, réglable 50–2000 m)
  ✔ Suppression du hack "cm traités comme mètres" dans la géoloc :
    les positions KML/CSV/carte utilisent la vraie distance estimée
  ✔ Calibration caméra en mètres (la focale en pixels ne change pas)
  ✔ Seuils comportementaux (APPROCHE) recalibrés en m/frame
  ⚠ Si un ancien drone_detector_config.json existe, supprimez-le ou
    vérifiez danger_dist / calib_real_width (anciennes valeurs en cm)
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from collections import deque
from datetime import datetime
import logging
import wave
import subprocess

from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from PIL import Image as PILImage, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("drone_detector.log", encoding="utf-8")]
)
logger = logging.getLogger(__name__)

BG         = "#080c10"
BG2        = "#0d1117"
BG3        = "#161b22"
CARD       = "#0d1117"
ACCENT     = "#00ff41"
ACCENT2    = "#ff3c3c"
WARNING    = "#ffb700"
TEXT       = "#ffffff"
TEXT_MUTED = "#aaaaaa"
BORDER     = "#21262d"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_MONO  = ("Segoe UI", 11)
FONT_LABEL = ("Segoe UI", 11)
FONT_BTN   = ("Segoe UI", 11, "bold")
FONT_STAT  = ("Segoe UI", 20, "bold")
FONT_CARD  = ("Segoe UI", 10, "bold")
FONT_HINT  = ("Segoe UI", 9)

# ══════════════════════════════════════════════════════════════════
#  UNITÉS : tout le système travaille en MÈTRES.
#  La distance est estimée via :  dist_m = (largeur_réelle_m × F) / largeur_px
#  où F est la focale en PIXELS (sans unité de longueur).
# ══════════════════════════════════════════════════════════════════

# Largeur réelle par classe, en mètres (envergure / dimension max visible)
CLASS_REAL_WIDTH_M = {
    "shahed":     2.5,    # envergure Shahed-136 ≈ 2,5 m
    "shahed-136": 2.5,
    "fpv-drone":  0.40,
    "drone":      0.40,
    "bird":       0.50,
    "not":        0.50,
}

KNOWN_WIDTH_M      = 0.5     # largeur par défaut (m) si classe inconnue
FOCAL_LENGTH       = 2000    # focale en pixels (calibrable)
CALIBRATION_ACTIVE = False

DANGER_DIST     = 300        # seuil de danger en MÈTRES
MIN_DIST_FLOOR  = 0          # distance plancher en MÈTRES (0 = désactivé)
                             # utile pour les vidéos de synthèse/simulation
CAM_HEADING     = 0          # cap caméra en degrés (0=nord, 90=est, 180=sud…)
CONFIRM_FRAMES  = 1          # nb de frames consécutives pour déclencher l'alarme
                             # (1 = immédiat, 3-5 = anti-faux-positifs)
TRAIL_LENGTH = 40
FRAME_SKIP   = 1

CLASS_COLORS = [
    (0, 255, 65), (0, 200, 255), (255, 200, 0),
    (255, 100, 200), (100, 255, 200),
]

# ── Couleurs spécifiques par classe ──────────────────────────────
CLASS_NAME_COLORS = {
    "shahed":     (0,   0,   220),   # rouge — menace principale
    "shahed-136": (0,   0,   220),
    "bird":       (0,   200, 50),    # vert — pas de danger
    "not":        (180, 180, 180),   # gris  — neutre
    "fpv-drone":  (0,   140, 255),   # orange
    "drone":      (0,   140, 255),
}

EARTH_R = 6_371_000.0


def class_width_m(name):
    """Largeur réelle (m) connue pour cette classe, sinon valeur par défaut."""
    return CLASS_REAL_WIDTH_M.get(name.lower(), KNOWN_WIDTH_M)


def pixel_to_azimuth(cx, cy, frame_w, frame_h, cam_hfov_deg=60.0,
                     cam_heading_deg=0.0):
    """
    Convertit la position horizontale d'un pixel en azimut géographique.
    cam_heading_deg : cap de la caméra (0=nord, 90=est, 180=sud, 270=ouest).
    Si la caméra pointe au nord (0°), un drone au centre → azimut 0°.
    Si la caméra pointe à l'est (90°), un drone au centre → azimut 90°.
    """
    norm_x  = (cx - frame_w / 2) / (frame_w / 2)
    h_angle = norm_x * (cam_hfov_deg / 2)
    return (cam_heading_deg + h_angle) % 360


def offset_latlon(lat, lon, distance_m, azimuth_deg):
    az_rad  = np.radians(azimuth_deg)
    delta_n = distance_m * np.cos(az_rad)
    delta_e = distance_m * np.sin(az_rad)
    lat2 = lat + np.degrees(delta_n / EARTH_R)
    lon2 = lon + np.degrees(delta_e / (EARTH_R * np.cos(np.radians(lat))))
    return lat2, lon2


class BehaviourAnalyser:
    HOVER_SPEED_PX   = 3.0
    HOVER_MIN_FRAMES = 20
    CIRCLE_ARC_DEG   = 270
    APPROACH_RATE    = 2.0    # m/frame (≈ 180 km/h à 25 fps) — v3 en mètres
    ERRATIC_ANGLE    = 90

    def __init__(self):
        self._hover_count   = {}
        self._angle_history = {}
        self._dist_history  = {}
        self._alerts        = {}

    def update(self, tid, vx, vy, dist):
        speed      = (vx**2 + vy**2) ** 0.5
        new_alerts = []

        if tid not in self._hover_count:
            self._hover_count[tid]   = 0
            self._angle_history[tid] = deque(maxlen=60)
            self._dist_history[tid]  = deque(maxlen=30)
            self._alerts[tid]        = set()

        if speed < self.HOVER_SPEED_PX:
            self._hover_count[tid] += 1
        else:
            self._hover_count[tid] = max(0, self._hover_count[tid] - 2)

        if self._hover_count[tid] >= self.HOVER_MIN_FRAMES:
            if "HOVERING" not in self._alerts[tid]:
                self._alerts[tid].add("HOVERING")
                new_alerts.append("HOVERING")
        else:
            self._alerts[tid].discard("HOVERING")

        if speed > 1.0:
            angle = np.degrees(np.arctan2(vy, vx)) % 360
            self._angle_history[tid].append(angle)
            if len(self._angle_history[tid]) >= 30:
                angles = list(self._angle_history[tid])
                diffs  = [abs((angles[i+1]-angles[i]+180) % 360 - 180)
                          for i in range(len(angles)-1)]
                if sum(diffs) >= self.CIRCLE_ARC_DEG:
                    if "CIRCLING" not in self._alerts[tid]:
                        self._alerts[tid].add("CIRCLING")
                        new_alerts.append("CIRCLING")
                else:
                    self._alerts[tid].discard("CIRCLING")

        if dist > 0:
            self._dist_history[tid].append(dist)
            if len(self._dist_history[tid]) >= 5:
                d_list = list(self._dist_history[tid])
                rate   = (d_list[0] - d_list[-1]) / len(d_list)
                if rate > self.APPROACH_RATE:
                    if "FAST APPROACH" not in self._alerts[tid]:
                        self._alerts[tid].add("FAST APPROACH")
                        new_alerts.append("FAST APPROACH")
                else:
                    self._alerts[tid].discard("FAST APPROACH")

        if speed > 2.0 and len(self._angle_history[tid]) >= 2:
            angles = list(self._angle_history[tid])
            change = abs((angles[-1] - angles[-2] + 180) % 360 - 180)
            if change >= self.ERRATIC_ANGLE:
                if "ERRATIC" not in self._alerts[tid]:
                    self._alerts[tid].add("ERRATIC")
                    new_alerts.append("ERRATIC")
            else:
                self._alerts[tid].discard("ERRATIC")

        return new_alerts

    def get_active(self, tid):
        return list(self._alerts.get(tid, set()))

    def reset(self, tid):
        for d in (self._hover_count, self._angle_history,
                  self._dist_history, self._alerts):
            d.pop(tid, None)


def send_email_alert(smtp_host, smtp_port, sender, password,
                     recipient, subject, body_html, image_paths=None):
    try:
        msg = MIMEMultipart("related")
        msg["From"]    = sender
        msg["To"]      = recipient
        msg["Subject"] = subject
        alt = MIMEMultipart("alternative")
        msg.attach(alt)
        alt.attach(MIMEText(body_html, "html", "utf-8"))
        if image_paths:
            for i, path in enumerate(image_paths):
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        img = MIMEImage(f.read())
                    img.add_header("Content-ID", f"<img{i}>")
                    img.add_header("Content-Disposition", "inline",
                                   filename=os.path.basename(path))
                    msg.attach(img)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, recipient, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)


try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

_alarm_active = False
_alarm_thread = None
_alarm_stop   = threading.Event()

def _alarm_loop(frequency=1200, interval=0.6):
    global _alarm_active
    while not _alarm_stop.is_set() and _alarm_active:
        if SOUND_AVAILABLE:
            winsound.Beep(frequency, 300)
        _alarm_stop.wait(interval)

def start_alarm():
    global _alarm_active, _alarm_thread, _alarm_stop
    if _alarm_active:
        return
    _alarm_active = True
    _alarm_stop.clear()
    _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
    _alarm_thread.start()

def stop_alarm():
    global _alarm_active
    _alarm_active = False
    _alarm_stop.set()


def generate_alarm_audio(danger_frames, total_frames, fps, output_wav,
                          freq=1200, sample_rate=44100):
    duration_s = total_frames / max(fps, 1)
    n_samples  = int(duration_s * sample_rate)
    audio      = np.zeros(n_samples, dtype=np.float32)
    for fid in set(danger_frames):
        t_start = fid / fps
        t_end   = (fid + 1) / fps
        s_start = int(t_start * sample_rate)
        s_end   = min(int(t_end * sample_rate), n_samples)
        t       = np.linspace(0, t_end - t_start, s_end - s_start, endpoint=False)
        tone    = 0.6 * np.sin(2 * np.pi * freq * t)
        fade    = min(64, len(tone))
        if fade > 0:
            tone[:fade]  *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        audio[s_start:s_end] += tone
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8
    audio_int = (audio * 32767).astype(np.int16)
    with wave.open(output_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int.tobytes())

def mux_video_audio(video_path, audio_path, output_path):
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-c:v", "copy", "-c:a", "aac", "-shortest", output_path]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


class KalmanDrone:
    def __init__(self, cx, cy):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4, dtype=float) * 0.1
        self.R = np.eye(2, dtype=float) * 5.0
        self.x = np.array([[cx],[cy],[0.],[0.]], dtype=float)
        self.P = np.eye(4, dtype=float) * 500.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return int(self.x[0,0]), int(self.x[1,0])

    def update(self, cx, cy):
        z = np.array([[cx],[cy]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_velocity(self):
        return float(self.x[2,0]), float(self.x[3,0])

    def get_position(self):
        return int(self.x[0,0]), int(self.x[1,0])


class DroneTracker:
    def __init__(self, max_lost=25, match_threshold=180):
        self.tracks          = {}
        self.next_id         = 0
        self.max_lost        = max_lost
        self.match_threshold = match_threshold

    def _new_track(self, name, cx, cy):
        tid = self.next_id
        self.tracks[tid] = {
            "kalman":    KalmanDrone(cx, cy),
            "trail":     deque(maxlen=TRAIL_LENGTH),
            "lost":      0,
            "name":      name,
            "predicted": (cx, cy),
        }
        self.tracks[tid]["trail"].append((cx, cy))
        self.next_id += 1
        return tid

    def track(self, detections):
        for tid, t in self.tracks.items():
            px, py = t["kalman"].predict()
            t["predicted"] = (px, py)
            t["lost"] += 1

        centers      = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for _,_,b in detections]
        matched_tids = set()
        matched_dets = set()
        det_to_tid   = {}
        tid_list     = sorted(self.tracks.keys())

        for det_i, (cx, cy) in enumerate(centers):
            best_tid, best_d = None, self.match_threshold
            for tid in tid_list:
                if tid in matched_tids:
                    continue
                px, py    = self.tracks[tid]["predicted"]
                d         = ((cx-px)**2 + (cy-py)**2) ** 0.5
                threshold = self.match_threshold * (1 + self.tracks[tid]["lost"] * 0.3)
                if d < threshold and d < best_d:
                    best_d, best_tid = d, tid
            if best_tid is not None:
                matched_tids.add(best_tid)
                matched_dets.add(det_i)
                det_to_tid[det_i] = best_tid
                self.tracks[best_tid]["kalman"].update(cx, cy)
                self.tracks[best_tid]["lost"] = 0
                self.tracks[best_tid]["name"] = detections[det_i][0]
                sx, sy = self.tracks[best_tid]["kalman"].get_position()
                self.tracks[best_tid]["trail"].append((sx, sy))

        for det_i in range(len(detections)):
            if det_i not in matched_dets:
                name, _, _ = detections[det_i]
                cx, cy = centers[det_i]
                already_exists = False
                for tid, t in self.tracks.items():
                    if tid in matched_tids:
                        continue
                    px, py = t["predicted"]
                    if ((cx-px)**2 + (cy-py)**2)**0.5 < self.match_threshold * 2:
                        self.tracks[tid]["kalman"].update(cx, cy)
                        self.tracks[tid]["lost"] = 0
                        self.tracks[tid]["name"] = name
                        sx, sy = self.tracks[tid]["kalman"].get_position()
                        self.tracks[tid]["trail"].append((sx, sy))
                        matched_tids.add(tid)
                        det_to_tid[det_i] = tid
                        already_exists = True
                        break
                if not already_exists:
                    new_tid = self._new_track(name, cx, cy)
                    det_to_tid[det_i] = new_tid

        results = []
        for det_i, (name, dist, bbox) in enumerate(detections):
            tid = det_to_tid.get(det_i, self.next_id - 1)
            if tid in self.tracks:
                vx, vy = self.tracks[tid]["kalman"].get_velocity()
                results.append((tid, name, dist, bbox,
                                list(self.tracks[tid]["trail"]), (vx, vy)))

        lost_ids = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in lost_ids:
            del self.tracks[tid]

        return results


# ── Classes dangereuses (Shahed = toujours danger) ────────────────
DANGER_CLASSES  = {"shahed", "shahed-136", "fpv-drone"}
NEUTRAL_CLASSES = {"bird", "not"}

DRONE_CLASS_ALIASES = {"airplane","aeroplane","bird","kite","helicopter",
                       "drone","uav","shahed","shahed-136","fpv-drone","not"}

def normalize_name(name):
    n = name.lower()
    if n in ("shahed", "shahed-136", "shahed136"):
        return "shahed"
    if n in ("fpv", "fpv-drone"):
        return "fpv-drone"
    if n == "bird":
        return "bird"
    if n == "not":
        return "not"
    return name

def is_threat(name):
    """Retourne True si la classe est une menace réelle."""
    return name.lower() in DANGER_CLASSES


def get_class_color(name, is_danger_dist=False):
    """Retourne la couleur BGR selon la classe."""
    n = name.lower()
    if n in CLASS_NAME_COLORS:
        return CLASS_NAME_COLORS[n]
    if is_danger_dist:
        return (0, 0, 220)
    return (0, 200, 50)


def get_direction(cx, cy, frame_w, frame_h, prev_dist, curr_dist,
                  prev_cx=-1, prev_cy=-1):
    if prev_cx >= 0 and prev_cy >= 0:
        dx = cx - prev_cx
        dy = cy - prev_cy
        if abs(dx) < 3 and abs(dy) < 3:
            dx = cx - frame_w // 2
            dy = cy - frame_h // 2
    else:
        dx = cx - frame_w // 2
        dy = cy - frame_h // 2

    angle_deg = np.degrees(np.arctan2(-dy, dx))
    if   -22.5  <= angle_deg <  22.5:  label = "EST"
    elif  22.5  <= angle_deg <  67.5:  label = "NORD-EST"
    elif  67.5  <= angle_deg < 112.5:  label = "NORD"
    elif 112.5  <= angle_deg < 157.5:  label = "NORD-OUEST"
    elif angle_deg >= 157.5 or angle_deg < -157.5: label = "OUEST"
    elif -157.5 <= angle_deg < -112.5: label = "SUD-OUEST"
    elif -112.5 <= angle_deg <  -67.5: label = "SUD"
    else:                              label = "SUD-EST"

    # v3 : seuils en MÈTRES par frame (~±2 m/frame ≈ ±180 km/h à 25 fps)
    if prev_dist > 0 and curr_dist > 0:
        delta = curr_dist - prev_dist
        if delta < -2.0:   approach = " + APPROCHE"
        elif delta > 2.0:  approach = " + S ELOIGNE"
        else:              approach = " STABLE"
    else:
        approach = ""

    return label + approach, angle_deg


def draw_direction_arrow(frame, cx, cy, angle_deg, color, frame_count):
    pulse     = 1.0 + 0.15 * np.sin(frame_count * 0.3)
    length    = int(55 * pulse)
    angle_rad = np.radians(angle_deg)
    ex = int(cx + length * np.cos(angle_rad))
    ey = int(cy - length * np.sin(angle_rad))
    cv2.arrowedLine(frame, (cx,cy), (ex,ey), (0,0,0), 5, tipLength=0.35)
    cv2.arrowedLine(frame, (cx,cy), (ex,ey), color, 2, tipLength=0.35)
    cv2.circle(frame, (cx,cy), 5, color, -1)
    cv2.circle(frame, (cx,cy), 7, (0,0,0), 1)


def draw_direction_label(frame, x1, y2, direction_label, color):
    fs = 0.45
    (tw, th), _ = cv2.getTextSize(direction_label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.rectangle(frame, (x1, y2+2), (x1+tw+6, y2+th+10), (0,0,0), -1)
    cv2.putText(frame, direction_label, (x1+3, y2+th+4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)


def estimate_distance(pixel_width, name=""):
    """
    Distance estimée en MÈTRES.
    dist_m = (largeur_réelle_m de la classe × focale_px) / largeur_px
    Si MIN_DIST_FLOOR > 0, la distance est au moins ce plancher
    (utile pour vidéos de synthèse où l'objet est virtuellement très proche).
    """
    if pixel_width > 0:
        d = round((class_width_m(name) * FOCAL_LENGTH) / pixel_width, 1)
        if MIN_DIST_FLOOR > 0:
            d = max(d, float(MIN_DIST_FLOOR))
        return d
    return -1


def draw_trail(frame, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha     = i / len(pts)
        c         = tuple(int(x * alpha) for x in color)
        thickness = max(1, int(2 * alpha))
        cv2.line(frame, pts[i-1], pts[i], c, thickness)


def draw_hud_overlay(frame, tracked, drones_in_danger, frame_count, h, w,
                     max_speed_kmh=0.0):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8,8), (580,160), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (8,8), (580,160), (0,200,50), 1)
    cv2.line(frame, (8,30), (580,30), (0,200,50), 1)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"[ SHAHED DETECTION SYSTEM ]  {ts}",
                (16,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,60), 1)
    calib_txt = "CAL:OK" if CALIBRATION_ACTIVE else f"CAL:EST  F={FOCAL_LENGTH:.0f}"
    calib_col = (0, 200, 50) if CALIBRATION_ACTIVE else (255, 183, 0)
    cv2.putText(frame, calib_txt,
                (400, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, calib_col, 1)
    cv2.putText(frame, f"OBJETS DETECTES: {len(tracked)}",
                (16,60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,65), 2)
    if drones_in_danger:
        danger_str = "  ".join([f"ID#{tid} {name} {dist:.0f}m"
                                 for tid,name,dist in drones_in_danger])
        cv2.putText(frame, f"[!] SHAHED: {danger_str}",
                    (16,95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    else:
        cv2.putText(frame, "THREAT: NONE",
                    (16,95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,50), 1)
    all_dists = [d for _,_,d,_,_,_ in tracked if d > 0]
    min_d = min(all_dists) if all_dists else 0
    cv2.putText(frame, f"MIN DISTANCE: {min_d:.0f} m",
                (16,128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,65), 2)
    spd_col = (0, 80, 255) if max_speed_kmh > 30 else (200, 200, 200)
    cv2.putText(frame, f"MAX SPEED: {max_speed_kmh:.1f} km/h",
                (300, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, spd_col, 2)
    if drones_in_danger:
        al = frame.copy()
        cv2.rectangle(al, (0,h-65), (w,h), (0,0,200), -1)
        cv2.addWeighted(al, 0.55, frame, 0.45, 0, frame)
        if (frame_count // 12) % 2 == 0:
            cv2.putText(frame, "////  ALERT: SHAHED DETECTED  ////",
                        (20,h-22), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)


def analyze_video(video_input, video_output, model_path,
                  frame_skip, log_cb, progress_cb, stop_event,
                  sound_enabled=True, preview_cb=None,
                  observer_lat=48.8566, observer_lon=2.3522,
                  cam_hfov_deg=60.0, cam_heading_deg=0.0,
                  confirm_frames=1):

    global FOCAL_LENGTH

    target_img  = cv2.imread("target.png", cv2.IMREAD_UNCHANGED)
    target_size = 90
    if target_img is not None:
        target_img = cv2.resize(target_img, (target_size, target_size))

    try:
        model = YOLO(model_path)
        log_cb(f"[OK] Modèle chargé : {model_path}", "success")
        log_cb(f"[OK] Classes : {list(model.names.values())}", "info")
    except Exception as e:
        log_cb(f"[ERROR] Chargement modèle : {e}", "error")
        return [], [], [], {}, [], []

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        log_cb(f"[ERROR] Vidéo introuvable : {video_input}", "error")
        return [], [], [], {}, [], []

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live_src = (total <= 0)  # webcam/RTSP n'ont pas de frame count
    log_cb(f"[INFO] {w}x{h} @ {fps}fps"
           f"{f' - {total} frames' if not is_live_src else ' - LIVE'}", "info")
    # v3.1 : focale auto déduite du FOV et de la résolution RÉELLE de la vidéo
    # (le F=2000 fixe n'était valable que pour du ~1080p)
    if not CALIBRATION_ACTIVE:
        FOCAL_LENGTH = (w / 2) / np.tan(np.radians(cam_hfov_deg / 2))
        log_cb(f"[CALIB] Focale auto : FOV {cam_hfov_deg:.0f}° @ {w}px "
               f"→ F={FOCAL_LENGTH:.0f}px", "info")
    log_cb(f"[CAP] Caméra orientée à {cam_heading_deg:.0f}° "
           f"(0=N 90=E 180=S 270=W) — confirmation {confirm_frames} frame(s)", "info")
    if CALIBRATION_ACTIVE:
        log_cb(f"[CALIB] ✅ Calibration active — F={FOCAL_LENGTH:.0f}px  "
               f"largeur Shahed={class_width_m('shahed'):.2f}m", "success")
    else:
        log_cb(f"[CALIB] ⚠  Focale estimée — F={FOCAL_LENGTH:.0f}px  "
               f"largeur Shahed={class_width_m('shahed'):.2f}m", "warning")

    out = cv2.VideoWriter(video_output,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker            = DroneTracker()
    confirm_counter    = {}   # tid → nb frames consécutives de danger
    all_distances      = []
    distance_timeline  = []
    all_detections     = []
    drones_in_danger_total = {}
    danger_frames      = []
    frame_count        = 0
    last_results       = []
    prev_distances     = {}
    prev_centers       = {}
    danger_crops       = {}
    geo_positions      = []
    behaviour_events   = []
    behaviour_analyser = BehaviourAnalyser()

    while True:
        if stop_event.is_set():
            log_cb("[STOP] Interrompu.", "warning")
            break
        ret, frame = cap.read()
        if not ret:
            break

        is_yolo_frame = (frame_count % frame_skip == 0)
        if is_yolo_frame:
            try:
                yolo_results = model(frame, conf=0.25, imgsz=640,
                                     agnostic_nms=True, verbose=False)
                detections = []
                for result in yolo_results:
                    for box in result.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        bw = x2-x1
                        if bw * (y2-y1) < 100:
                            continue
                        cid  = int(box.cls[0])
                        name = normalize_name(result.names[cid])
                        dist = estimate_distance(bw, name)
                        detections.append((name, dist, (x1,y1,x2,y2)))
                last_results = detections
            except Exception as exc:
                logger.debug(f"Frame {frame_count} YOLO error: {exc}")
                last_results = []

        # FIX frame_skip : sur les frames intermédiaires on ne repasse
        # PAS les mêmes détections au tracker (ça tirait les vitesses
        # Kalman vers 0). On fait juste predict() pour interpoler.
        if is_yolo_frame:
            tracked = tracker.track(last_results)
        else:
            # Prédiction pure Kalman — avance les positions sans update
            tracked = []
            for tid, t in tracker.tracks.items():
                px, py = t["kalman"].predict()
                t["predicted"] = (px, py)
                t["trail"].append((px, py))
                t["lost"] += 1
                vx, vy = t["kalman"].get_velocity()
                # Reconstruire un bbox centré sur la prédiction
                last_name = t.get("name", "not")
                last_dist = prev_distances.get(tid, -1)
                hw = 30; hh = 20
                fake_bbox = (px-hw, py-hh, px+hw, py+hh)
                tracked.append((tid, last_name, last_dist,
                                fake_bbox, list(t["trail"]), (vx, vy)))
            # Nettoyer les tracks trop perdus
            lost_ids = [tid for tid, t in tracker.tracks.items()
                        if t["lost"] > tracker.max_lost]
            for tid in lost_ids:
                del tracker.tracks[tid]
        drones_in_danger = []
        closest_crop     = None
        min_d_frame      = 99999

        for tid, name, dist, (x1,y1,x2,y2), trail, velocity in tracked:
            all_distances.append(dist)
            all_detections.append({"frame": frame_count, "id": tid,
                                    "name": name, "distance": dist})

            # ── Logique de danger : Shahed = toujours danger ──────
            threat     = is_threat(name)
            is_danger  = threat or (0 < dist < DANGER_DIST and name not in NEUTRAL_CLASSES)
            color      = get_class_color(name, is_danger)

            draw_trail(frame, trail, color)
            thickness = 2 if is_danger else 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

            bw = x2-x1; bh = y2-y1; area = bw*bh
            fs = 0.45 if area < 3000 else 0.75

            # ── Label avec classe et distance (mètres) ────────────
            conf_str = ""
            label    = f"#{tid} {name.upper()} {dist:.0f}m{conf_str}"
            if threat:
                label = f"[!] #{tid} {name.upper()} {dist:.0f}m"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            txt_col = (255,255,255) if threat else (0,0,0)
            cv2.putText(frame, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, txt_col, 1)

            cx_drone = x1 + bw // 2
            cy_drone = y1 + bh // 2
            vx, vy   = velocity
            speed    = (vx**2 + vy**2) ** 0.5

            if speed > 1.5:
                angle_deg = np.degrees(np.arctan2(-vy, vx))
                if   -22.5  <= angle_deg <  22.5:  dir_txt = "EST"
                elif  22.5  <= angle_deg <  67.5:  dir_txt = "NORD-EST"
                elif  67.5  <= angle_deg < 112.5:  dir_txt = "NORD"
                elif 112.5  <= angle_deg < 157.5:  dir_txt = "NORD-OUEST"
                elif angle_deg >= 157.5 or angle_deg < -157.5: dir_txt = "OUEST"
                elif -157.5 <= angle_deg < -112.5: dir_txt = "SUD-OUEST"
                elif -112.5 <= angle_deg <  -67.5: dir_txt = "SUD"
                else:                              dir_txt = "SUD-EST"
            else:
                prev_d = prev_distances.get(tid, dist)
                dir_txt, angle_deg = get_direction(
                    cx_drone, cy_drone, w, h, prev_d, dist,
                    prev_centers.get(tid, (-1,-1))[0],
                    prev_centers.get(tid, (-1,-1))[1])

            prev_d = prev_distances.get(tid, dist)
            # v3 : seuils en MÈTRES par frame
            if prev_d > 0 and dist > 0:
                delta = dist - prev_d
                if delta < -2.0:   approach = " APPROCHE"
                elif delta > 2.0:  approach = " S ELOIGNE"
                else:              approach = " STABLE"
            else:
                approach = " STABLE"

            dir_label = dir_txt + approach
            draw_direction_arrow(frame, cx_drone, cy_drone, angle_deg, color, frame_count)
            draw_direction_label(frame, x1, y2, dir_label, color)
            prev_distances[tid] = dist
            prev_centers[tid]   = (cx_drone, cy_drone)

            # v3 : vitesse via largeur réelle (m) de la classe
            if bw > 0 and fps > 0:
                m_per_px  = class_width_m(name) / bw
                speed_kmh = speed * m_per_px * fps * 3.6
            else:
                speed_kmh = 0.0

            spd_txt = f"{speed_kmh:.1f} km/h"
            spd_col = (0, 80, 255) if speed_kmh > 30 else (200, 200, 200)
            (sw, sh), _ = cv2.getTextSize(spd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            sx = x1 + bw - sw - 2
            sy = y1 - th - 20
            if sy > 10:
                cv2.rectangle(frame, (sx-2, sy-sh-2), (sx+sw+2, sy+4), (0,0,0), -1)
                cv2.putText(frame, spd_txt, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, spd_col, 1)

            if speed > 0.5 and fps > 0:
                pred_frames = int(fps * 2)
                pred_pts = []
                for step in range(1, 6):
                    t_pred  = pred_frames * step / 5
                    px_pred = int(cx_drone + vx * t_pred)
                    py_pred = int(cy_drone + vy * t_pred)
                    px_pred = max(0, min(w-1, px_pred))
                    py_pred = max(0, min(h-1, py_pred))
                    pred_pts.append((px_pred, py_pred))
                prev_pt    = (cx_drone, cy_drone)
                pred_color = (0, 60, 200) if is_danger else (180, 180, 180)
                for i, pt in enumerate(pred_pts):
                    cv2.line(frame, prev_pt, pt, (0,0,0), 2)
                    cv2.line(frame, prev_pt, pt, pred_color, 1, cv2.LINE_AA)
                    radius = max(2, 5-i)
                    cv2.circle(frame, pt, radius, (0,0,0), -1)
                    cv2.circle(frame, pt, max(1,radius-1), pred_color, -1)
                    prev_pt = pt
                if pred_pts:
                    final = pred_pts[-1]
                    cv2.circle(frame, final, 7, (0,0,0), -1)
                    cv2.circle(frame, final, 5, pred_color, -1)
                    cv2.putText(frame, "+2s", (final[0]+6, final[1]+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, pred_color, 1)

            if dist > 0:
                az = pixel_to_azimuth(cx_drone, cy_drone, w, h,
                                      cam_hfov_deg, cam_heading_deg)
                # v3 : dist est déjà en mètres — plus aucun hack d'unité
                dist_m = float(dist)
                dlat, dlon = offset_latlon(observer_lat, observer_lon, dist_m, az)
                geo_positions.append({
                    "frame": frame_count, "tid": tid,
                    "lat": dlat, "lon": dlon,
                    "dist": dist, "azimuth": az,
                    "speed_kmh": round(speed_kmh, 1),
                    "name": name,
                })

            new_beh = behaviour_analyser.update(tid, vx, vy, dist)
            for b in new_beh:
                behaviour_events.append({
                    "frame": frame_count, "tid": tid,
                    "name": name, "behaviour": b, "dist": dist})
                log_cb(f"[BEHAVIOUR] ID#{tid} {name} → {b} ({dist:.0f}m)", "warning")
            active_beh = behaviour_analyser.get_active(tid)
            if active_beh:
                badge_col = (0, 165, 255)
                badge_txt = " | ".join(active_beh)
                (bw2, bh2), _ = cv2.getTextSize(
                    badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame,
                              (x1, y2+14), (x1+bw2+6, y2+bh2+22),
                              (0,0,0), -1)
                cv2.putText(frame, badge_txt, (x1+3, y2+bh2+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, badge_col, 1)

            if is_danger:
                drones_in_danger.append((tid, name, dist))
                drones_in_danger_total[tid] = (name, dist)
                if dist < min_d_frame:
                    min_d_frame  = dist
                    closest_crop = frame[y1:y2, x1:x2]
                prev_best = danger_crops.get(tid, {}).get("dist", 99999)
                if dist < prev_best and (y2-y1) > 10 and (x2-x1) > 10:
                    pad  = 20
                    py1  = max(0, y1-pad); py2 = min(frame.shape[0], y2+pad)
                    px1  = max(0, x1-pad); px2 = min(frame.shape[1], x2+pad)
                    crop = frame[py1:py2, px1:px2].copy()
                    if crop.size > 0:
                        danger_crops[tid] = {"name": name, "dist": dist,
                                             "frame": frame_count, "img": crop}
                if target_img is not None and target_img.shape[2] == 4:
                    cx2 = x1 + bw//2 - target_size//2
                    cy2 = y1 + bh//2 - target_size//2
                    ov  = target_img[:,:,:3]; mk = target_img[:,:,3]
                    ht, wt = ov.shape[:2]
                    roi = frame[cy2:cy2+ht, cx2:cx2+wt]
                    if roi.shape[:2] == (ht, wt):
                        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mk))
                        fg = cv2.bitwise_and(ov,  ov,  mask=mk)
                        frame[cy2:cy2+ht, cx2:cx2+wt] = cv2.add(bg, fg)

        frame_dists = [d for _,_,d,_,_,_ in tracked if d > 0]
        fd_min = min(frame_dists) if frame_dists else 0
        distance_timeline.append((frame_count, fd_min))

        frame_max_speed = 0.0
        for tid, name, dist, (x1,y1,x2,y2), trail, velocity in tracked:
            vx, vy = velocity
            bw_t   = x2-x1
            if bw_t > 0 and fps > 0:
                spd = ((vx**2+vy**2)**0.5) * (class_width_m(name)/bw_t) * fps * 3.6
                frame_max_speed = max(frame_max_speed, spd)

        # Confirmation temporelle : n'alarmer qu'après confirm_frames consécutives
        confirmed_danger = []
        current_tids = {tid for tid,_,_,_,_,_ in tracked}
        for tid, name, dist in drones_in_danger:
            confirm_counter[tid] = confirm_counter.get(tid, 0) + 1
            if confirm_counter[tid] >= max(1, confirm_frames):
                confirmed_danger.append((tid, name, dist))
        # Réinitialiser les compteurs des tids absents de danger cette frame
        for tid in list(confirm_counter.keys()):
            if tid not in {t for t,_,_ in drones_in_danger}:
                confirm_counter[tid] = 0
        # Nettoyer les tids perdus
        for tid in list(confirm_counter.keys()):
            if tid not in current_tids:
                del confirm_counter[tid]

        if confirmed_danger and sound_enabled:
            start_alarm()
            danger_frames.append(frame_count)
        else:
            stop_alarm()
        drones_in_danger = confirmed_danger  # HUD n'affiche que les confirmés

        draw_hud_overlay(frame, tracked, drones_in_danger, frame_count, h, w,
                         max_speed_kmh=frame_max_speed)

        if closest_crop is not None and closest_crop.size > 0:
            hud = cv2.resize(closest_crop, (140, 140))
            xs, ys = w-150, h-150
            frame[ys:ys+140, xs:xs+140] = hud
            cv2.rectangle(frame, (xs,ys), (xs+140,ys+140), (0,255,65), 1)
            cv2.putText(frame, "TARGET", (xs+2, ys+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,65), 1)

        out.write(frame)

        if preview_cb is not None:
            preview_cb(frame)

        if frame_count % 20 == 0:
            if is_live_src:
                progress_cb(-1, frame_count, frame_count,
                            len(tracked), len(drones_in_danger))
            else:
                pct = int((frame_count / max(total,1)) * 100)
                progress_cb(pct, frame_count, total,
                            len(tracked), len(drones_in_danger))
            log_cb(f"[INFO] Frame {frame_count}/{total} - "
                   f"{len(tracked)} objet(s) - {len(drones_in_danger)} menace(s)", "info")

        frame_count += 1

    cap.release()
    out.release()
    stop_alarm()

    if danger_frames:
        log_cb("[INFO] Ajout audio alarme dans la vidéo...", "info")
        tmp_wav   = video_output.replace(".mp4", "_alarm.wav")
        final_out = video_output.replace(".mp4", "_audio.mp4")
        try:
            generate_alarm_audio(danger_frames, frame_count, fps, tmp_wav)
            if mux_video_audio(video_output, tmp_wav, final_out):
                try:
                    os.remove(video_output)
                    os.rename(final_out, video_output)
                    log_cb("[OK] Audio intégré dans la vidéo.", "success")
                except Exception:
                    log_cb(f"[OK] Vidéo avec audio : {os.path.basename(final_out)}", "success")
            else:
                log_cb("[WARN] ffmpeg non trouvé - vidéo sans audio.", "warning")
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception as e:
            log_cb(f"[WARN] Audio non intégré : {e}", "warning")

    log_cb(f"[OK] Terminé : {frame_count} frames", "success")
    return all_distances, distance_timeline, all_detections, danger_crops, geo_positions, behaviour_events


def generate_charts(all_distances, distance_timeline, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#080c10")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")

    if all_distances:
        valid = [d for d in all_distances if d > 0]
        axes[0].hist(valid, bins=30, color="#00ff41", edgecolor="#080c10", alpha=0.85)
        axes[0].axvline(DANGER_DIST, color="#ff3c3c", linestyle="--",
                         linewidth=1.5, label=f"Seuil danger ({DANGER_DIST}m)")
        axes[0].set_title("Distribution des distances", color="white", fontsize=12)
        axes[0].set_xlabel("Distance (m)", color="#c9d1d9")
        axes[0].set_ylabel("Fréquence", color="#c9d1d9")
        axes[0].legend(facecolor="#0d1117", edgecolor="#21262d",
                        labelcolor="white", fontsize=9)

    if distance_timeline:
        frames = [x[0] for x in distance_timeline if x[1] > 0]
        dists  = [x[1] for x in distance_timeline if x[1] > 0]
        axes[1].plot(frames, dists, color="#00ff41", linewidth=1.2, alpha=0.85)
        axes[1].fill_between(frames, dists, alpha=0.15, color="#00ff41")
        axes[1].axhline(DANGER_DIST, color="#ff3c3c", linestyle="--",
                         linewidth=1.5, label="Seuil danger")
        axes[1].fill_between(frames, 0, DANGER_DIST,
                              where=[d < DANGER_DIST for d in dists],
                              color="#ff3c3c", alpha=0.2, label="Zone danger")
        axes[1].set_title("Distance minimale par frame", color="white", fontsize=12)
        axes[1].set_xlabel("Frame", color="#c9d1d9")
        axes[1].set_ylabel("Distance (m)", color="#c9d1d9")
        axes[1].legend(facecolor="#0d1117", edgecolor="#21262d",
                        labelcolor="white", fontsize=9)

    plt.suptitle("Rapport — Shahed Detection System",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def generate_geomap(geo_positions, observer_lat, observer_lon,
                    danger_dist_m, output_path):
    if not geo_positions:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#080c10")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#484f58", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262d")

    def to_xy(lat, lon):
        dy = (lat - observer_lat) * EARTH_R * np.pi / 180
        dx = (lon - observer_lon) * EARTH_R * np.cos(np.radians(observer_lat)) * np.pi / 180
        return dx, dy

    # v3 : le cercle de danger correspond au VRAI seuil en mètres
    all_dists_m = [p["dist"] for p in geo_positions if p["dist"] > 0]
    max_dist_m  = max(all_dists_m) if all_dists_m else float(danger_dist_m)
    danger_m    = float(danger_dist_m)
    view_m      = max(max_dist_m * 1.2, danger_m * 1.5)

    for r_m in [view_m * 0.25, view_m * 0.5, view_m]:
        circle = plt.Circle((0,0), r_m, color="#21262d",
                             fill=False, linewidth=0.8, linestyle="--")
        ax.add_patch(circle)
        ax.text(0, r_m, f"{r_m:.0f}m", color="#484f58",
                fontsize=6, ha="center", va="bottom")

    ax.add_patch(plt.Circle((0,0), danger_m, color="#ff3c3c",
                             fill=True, alpha=0.08, linewidth=1.2,
                             linestyle="-", edgecolor="#ff3c3c"))
    ax.text(0, -danger_m, f"danger {danger_m:.0f}m", color="#ff3c3c",
            fontsize=6, ha="center", va="top")

    by_tid = {}
    for p in geo_positions:
        by_tid.setdefault(p["tid"], []).append(p)

    tid_colors = ["#00ff41","#00c8ff","#ffc800","#ff64c8","#64ffc8"]
    for i, (tid, pts) in enumerate(by_tid.items()):
        col = tid_colors[i % len(tid_colors)]
        xs  = [to_xy(p["lat"], p["lon"])[0] for p in pts]
        ys  = [to_xy(p["lat"], p["lon"])[1] for p in pts]
        ax.plot(xs, ys, color=col, linewidth=1.0, alpha=0.6)
        for p in pts:
            x, y = to_xy(p["lat"], p["lon"])
            c = "#ff0000" if p.get("name","") == "shahed" else (
                "#ff3c3c" if p["dist"] < danger_dist_m else col)
            ax.scatter(x, y, color=c, s=8, zorder=3, alpha=0.7)
        if xs:
            ax.scatter(xs[-1], ys[-1], color=col, s=60,
                       zorder=5, marker="^",
                       edgecolors="white", linewidths=0.5)
            ax.text(xs[-1]+view_m*0.02, ys[-1],
                    f"#{tid}", color=col, fontsize=7, va="center")

    ax.scatter(0, 0, color="#ffffff", s=120, zorder=6,
               marker="o", edgecolors=ACCENT, linewidths=2)
    ax.text(view_m*0.03, 0, "YOU", color="white",
            fontsize=7, va="center", fontweight="bold")

    for label, x, y in [("N",0,view_m*0.95),("S",0,-view_m*0.95),
                          ("E",view_m*0.95,0),("W",-view_m*0.95,0)]:
        ax.text(x, y, label, color="#484f58", fontsize=8,
                ha="center", va="center", fontweight="bold")

    lim = view_m
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("East ←→ West (m)", color="#484f58", fontsize=7)
    ax.set_ylabel("South ↕ North (m)", color="#484f58", fontsize=7)
    ax.set_title("Positions estimées — Shahed Detector", color="white", fontsize=10)
    ax.grid(color="#161b22", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return True


def export_csv(geo_positions, fps, output_path):
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame","time_s","drone_id","name",
                         "latitude","longitude",
                         "distance_m","azimuth_deg","speed_kmh","danger","shahed"])
        for p in geo_positions:
            writer.writerow([
                p["frame"],
                round(p["frame"] / max(fps,1), 3),
                p["tid"],
                p.get("name",""),
                round(p["lat"], 8),
                round(p["lon"], 8),
                p["dist"],
                round(p["azimuth"], 2),
                p.get("speed_kmh", 0.0),
                1 if p["dist"] < DANGER_DIST else 0,
                1 if p.get("name","") == "shahed" else 0,
            ])
    return True


def export_kml(geo_positions, observer_lat, observer_lon, fps, output_path):
    """
    Export KML pour Google Earth.
    v3 : distances en mètres, positions directement issues de l'estimation.
    """
    by_tid = {}
    for p in geo_positions:
        by_tid.setdefault(p["tid"], []).append(p)

    GREEN  = "ff41ff00"
    RED    = "ff0000ff"
    WHITE  = "ffffffff"
    ORANGE = "ff0080ff"

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '  <Document>',
             f'    <name>Shahed Detection — {datetime.now().strftime("%Y-%m-%d %H:%M")}</name>',
             f'    <Style id="safe"><LineStyle><color>{GREEN}</color><width>2</width></LineStyle></Style>',
             f'    <Style id="danger"><LineStyle><color>{RED}</color><width>3</width></LineStyle></Style>',
             f'    <Style id="shahed"><LineStyle><color>{RED}</color><width>4</width></LineStyle></Style>',
             f'    <Style id="observer"><IconStyle><color>{WHITE}</color><scale>1.2</scale></IconStyle></Style>',
             '    <Placemark>',
             '      <name>Observer</name>',
             '      <styleUrl>#observer</styleUrl>',
             '      <Point>',
             '        <altitudeMode>clampToGround</altitudeMode>',
             f'        <coordinates>{observer_lon},{observer_lat},0</coordinates>',
             '      </Point>',
             '    </Placemark>',
    ]

    # Icône Google Earth intégrée (avion = plus proche d'un drone)
    SHAHED_ICON = "http://maps.google.com/mapfiles/kml/shapes/airports.png"
    BIRD_ICON   = "http://maps.google.com/mapfiles/kml/shapes/donut.png"

    for tid, pts in sorted(by_tid.items()):
        # clampToGround plaque les points au sol — z=0 suffit
        coords = "\n          ".join(
            f"{p['lon']},{p['lat']},0" for p in pts
        )
        name_class = pts[0].get("name","")
        is_shahed  = (name_class == "shahed")
        style      = "shahed" if is_shahed else (
                     "danger" if any(p["dist"] < DANGER_DIST for p in pts) else "safe")

        # Style icône unique pour ce tid
        icon_id    = f"icon_{tid}"
        icon_href  = SHAHED_ICON if is_shahed else BIRD_ICON
        icon_color = "ff0000ff" if is_shahed else "ff41ff00"
        icon_scale = "1.5"      if is_shahed else "0.9"
        lines += [
            f'    <Style id="{icon_id}">',
            '      <IconStyle>',
            f'        <color>{icon_color}</color>',
            f'        <scale>{icon_scale}</scale>',
            '        <Icon>',
            f'          <href>{icon_href}</href>',
            '        </Icon>',
            '      </IconStyle>',
            '      <LabelStyle>',
            f'        <color>{icon_color}</color>',
            '        <scale>0.85</scale>',
            '      </LabelStyle>',
            '    </Style>',
        ]

        # Tracé de la trajectoire
        lines += [
            '    <Placemark>',
            f'      <name>{name_class.upper()} #{tid}</name>',
            f'      <styleUrl>#{style}</styleUrl>',
            '      <LineString>',
            '        <tessellate>1</tessellate>',
            '        <altitudeMode>clampToGround</altitudeMode>',
            f'        <coordinates>{coords}</coordinates>',
            '      </LineString>',
            '    </Placemark>',
        ]

        # Icône sur la dernière position détectée
        last = pts[-1]
        # v3.1 : appliquer le plancher dans le nom du pin KML aussi
        pin_dist = max(last["dist"], float(MIN_DIST_FLOOR)) \
                   if MIN_DIST_FLOOR > 0 and last["dist"] > 0 \
                   else last["dist"]
        pin_spd  = last.get("speed_kmh", 0.0)
        if is_shahed:
            pin_name = (f"[SHAHED] #{tid}  {pin_dist:.0f}m  "
                        f"az:{last['azimuth']:.0f}deg  {pin_spd:.0f}km/h")
        else:
            pin_name = (f"[{name_class.upper()}] #{tid}  {pin_dist:.0f}m  "
                        f"{pin_spd:.0f}km/h")
        lines += [
            '    <Placemark>',
            f'      <name>{pin_name}</name>',
            f'      <styleUrl>#{icon_id}</styleUrl>',
            '      <Point>',
            '        <altitudeMode>clampToGround</altitudeMode>',
            f'        <coordinates>{last["lon"]},{last["lat"]},0</coordinates>',
            '      </Point>',
            '    </Placemark>',
        ]

    lines += ['  </Document>', '</kml>']
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def export_pdf(all_distances, all_detections, chart_path, output_path,
               danger_crops=None, crops_dir=None, geomap_path=None):
    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               topMargin=2*cm, bottomMargin=2*cm,
                               leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    t_s    = ParagraphStyle("T", parent=styles["Title"], fontSize=18,
                            textColor=rl_colors.HexColor("#0d1117"), spaceAfter=6)
    sub_s  = ParagraphStyle("S", parent=styles["Normal"], fontSize=10,
                            textColor=rl_colors.grey, spaceAfter=4)
    b_s    = ParagraphStyle("B", parent=styles["Normal"], fontSize=11, spaceAfter=4)

    valid = [d for d in all_distances if d > 0]
    from collections import Counter
    frames_count   = Counter(d["frame"] for d in all_detections)
    drones_uniques = Counter(frames_count.values()).most_common(1)[0][0] if frames_count else 0
    danger_per_frame = Counter(d["frame"] for d in all_detections
                               if 0 < d["distance"] < DANGER_DIST)
    drones_danger  = Counter(danger_per_frame.values()).most_common(1)[0][0] \
                     if danger_per_frame else 0
    shahed_count   = sum(1 for d in all_detections if d.get("name","") == "shahed")
    statut = "⚠ ALERTE — Shahed détecté !" if shahed_count > 0 else (
             "ALERTE — Menaces détectées" if drones_danger > 0 else "OK — Aucune menace")

    calib_info = (f"Calibré — F={FOCAL_LENGTH:.0f}px / largeur Shahed="
                  f"{class_width_m('shahed'):.2f}m"
                  if CALIBRATION_ACTIVE
                  else f"Estimé — F={FOCAL_LENGTH:.0f}px / largeur Shahed="
                       f"{class_width_m('shahed'):.2f}m")

    elements = []
    elements.append(Paragraph("Rapport — Shahed Detection System", t_s))
    elements.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", sub_s))
    elements.append(Paragraph(f"Modèle : YOLOv8s — mAP@50 = 91.1% (bird/not/shahed)", sub_s))
    elements.append(Paragraph(f"Calibration caméra : {calib_info}", sub_s))
    elements.append(Spacer(1, 0.4*cm))

    summ = Table([
        ["Objets détectés",        str(drones_uniques)],
        ["Shahed détectés",        str(shahed_count)],
        ["Menaces en zone danger", str(drones_danger)],
        ["Statut",                 statut],
    ], colWidths=[7*cm, 8*cm])
    summ.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), rl_colors.HexColor("#0d1117")),
        ("TEXTCOLOR",  (0,0), (0,-1), rl_colors.white),
        ("BACKGROUND", (1,0), (1,-1), rl_colors.HexColor("#f0f4f8")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 11),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("GRID",       (0,0), (-1,-1), 0.5, rl_colors.grey),
        ("ROWHEIGHT",  (0,0), (-1,-1), 26),
    ]))
    elements.append(summ)
    elements.append(Spacer(1, 0.5*cm))

    if valid:
        elements.append(Paragraph("<b>Statistiques distances</b>", b_s))
        elements.append(Spacer(1, 0.2*cm))
        st = Table([
            ["Statistique",        "Valeur (m)"],
            ["Distance minimum",   f"{min(valid):.0f}"],
            ["Distance maximum",   f"{max(valid):.0f}"],
            ["Distance moyenne",   f"{sum(valid)/len(valid):.0f}"],
            ["Seuil danger",       f"< {DANGER_DIST}"],
        ], colWidths=[8*cm, 7*cm])
        st.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), rl_colors.HexColor("#1a2332")),
            ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [rl_colors.HexColor("#f5f5f5"), rl_colors.white]),
            ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.grey),
            ("ROWHEIGHT",     (0,0), (-1,-1), 22),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
        ]))
        elements.append(st)
        elements.append(Spacer(1, 0.5*cm))

    if os.path.exists(chart_path):
        elements.append(Paragraph("<b>Analyse graphique</b>", b_s))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(RLImage(chart_path, width=17*cm, height=5.5*cm))

    if geomap_path and os.path.exists(geomap_path):
        elements.append(Spacer(1, 0.6*cm))
        elements.append(Paragraph("<b>Carte trajectoire estimée</b>", b_s))
        elements.append(Spacer(1, 0.2*cm))
        try:
            from PIL import Image as _PIL
            with _PIL.open(geomap_path) as _im:
                _iw, _ih = _im.size
            _ratio  = _iw / _ih
            _disp_w = 14 * cm
            _disp_h = _disp_w / _ratio
        except Exception:
            _disp_w = 14 * cm
            _disp_h = 14 * cm
        elements.append(RLImage(geomap_path, width=_disp_w, height=_disp_h))

    if danger_crops:
        best = min(danger_crops.values(), key=lambda x: x["dist"])
        if crops_dir and best.get("img") is not None:
            crop_path = os.path.join(crops_dir, "danger_drone_best.png")
            try:
                cv2.imwrite(crop_path, best["img"])
            except Exception:
                crop_path = None
        else:
            crop_path = None

        if crop_path and os.path.exists(crop_path):
            elements.append(Spacer(1, 0.6*cm))
            elements.append(Paragraph("<b>Zone danger — capture menace la plus proche</b>", b_s))
            elements.append(Spacer(1, 0.3*cm))
            ih, iw  = best["img"].shape[:2]
            max_w   = 10 * cm; max_h = 8 * cm
            scale   = min(max_w/iw, max_h/ih)
            disp_w  = iw * scale; disp_h = ih * scale
            cap_style = ParagraphStyle("cap", parent=b_s, fontSize=9,
                                        alignment=1,
                                        textColor=rl_colors.HexColor("#333333"))
            frame_table = Table(
                [[RLImage(crop_path, width=disp_w, height=disp_h)],
                 [Paragraph(
                     f"<b>{best['name'].upper()}</b> · distance: <b>{best['dist']:.0f} m</b>"
                     f" · frame {best['frame']}",
                     cap_style)]],
                colWidths=[disp_w + 1*cm])
            frame_table.setStyle(TableStyle([
                ("ALIGN",      (0,0), (-1,-1), "CENTER"),
                ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                ("BOX",        (0,0), (-1,-1), 1, rl_colors.HexColor("#cc0000")),
                ("BACKGROUND", (0,0), (0,0),   rl_colors.HexColor("#1a2332")),
                ("BACKGROUND", (0,1), (0,1),   rl_colors.HexColor("#fff0f0")),
                ("ROWHEIGHT",  (0,0), (0,0),   disp_h+6),
                ("ROWHEIGHT",  (0,1), (0,1),   28),
            ]))
            elements.append(frame_table)

    doc.build(elements)


class DroneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SHAHED DETECTION SYSTEM")
        self.state("zoomed")
        self.configure(bg=BG)

        self.video_path  = tk.StringVar()
        self.source_type = tk.StringVar(value="file")  # file / webcam / rtsp
        self.rtsp_url    = tk.StringVar(value="rtsp://")

        # ── Modèle Shahed entraîné ────────────────────────────────
        _shahed_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs","detect","shahed_detector","weights","best.pt")
        _fallback = "yolov8n.pt"
        if os.path.exists(_shahed_model):
            self.model_path = tk.StringVar(value=_shahed_model)
        else:
            self.model_path = tk.StringVar(value=_fallback)

        self.output_dir  = tk.StringVar()
        self.frame_skip  = tk.IntVar(value=1)
        self.danger_dist = tk.IntVar(value=300)     # v3 : MÈTRES
        self.dist_floor  = tk.IntVar(value=0)       # v3.1 : plancher (0=off)
        self.is_running  = False
        self.stop_event  = threading.Event()
        self._last_video_out    = ""
        self._last_pdf_out      = ""
        self._last_run_dir      = ""
        self._last_kml_out      = ""
        self._last_csv_out      = ""
        self.sound_on           = True
        self._preview_counter   = 0
        self._preview_photo     = None
        self.observer_lat       = tk.StringVar(value="48.8566")
        self.observer_lon       = tk.StringVar(value="2.3522")
        self.cam_hfov           = tk.StringVar(value="60")
        self.cam_heading        = tk.StringVar(value="0")
        self.confirm_frames     = tk.IntVar(value=3)
        self._geo_positions     = []
        self._geomap_photo      = None

        # v3 : calibration en MÈTRES
        self.calib_real_width   = tk.StringVar(value="0.5")
        self.calib_ref_distance = tk.StringVar(value="")
        self.calib_pixel_width  = tk.StringVar(value="")
        self.calib_active       = tk.BooleanVar(value=False)

        self.email_enabled      = tk.BooleanVar(value=False)
        self.email_sender       = tk.StringVar(value="")
        self.email_password     = tk.StringVar(value="")
        self.email_recipient    = tk.StringVar(value="")
        self._email_sent_events = set()

        self.ntfy_enabled       = tk.BooleanVar(value=False)
        self.ntfy_channel       = tk.StringVar(value="")

        self._setup_styles()
        self._build_ui()
        self._load_config()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if not PIL_AVAILABLE:
            self._log("[WARN] Pillow non installé — preview désactivé. "
                      "Lancez: pip install Pillow", "warning")
        if not REQUESTS_AVAILABLE:
            self._log("[WARN] requests non installé — Ntfy désactivé. "
                      "Lancez: pip install requests", "warning")

        # ── Message de bienvenue ──────────────────────────────────
        if os.path.exists(_shahed_model):
            self._log("✅ Modèle Shahed chargé — mAP@50=91.1% "
                      "(bird/not/shahed)", "success")
        else:
            self._log("⚠ Modèle Shahed non trouvé — utilisation YOLOv8n "
                      "par défaut", "warning")
        self._log("ℹ v3 : toutes les distances sont en MÈTRES "
                  "(Shahed = 2,5 m d'envergure)", "info")

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG3, background=ACCENT,
                        thickness=8, borderwidth=0)
        style.configure("TScale", background=BG2, troughcolor=BG3,
                        sliderlength=16, sliderrelief="flat")

    CONFIG_FILE = "drone_detector_config.json"

    def _load_config(self):
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            self.CONFIG_FILE)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # v3 : ignorer les configs en cm (anciennes versions)
            if cfg.get("units") != "m":
                logger.warning("[CONFIG] Ancien fichier de config (cm) détecté"
                               " — valeurs de distance ignorées.")
                cfg.pop("danger_dist", None)
                cfg.pop("calib_real_width", None)
                cfg.pop("calib_ref_distance", None)
                cfg.pop("calib_focal", None)
                cfg.pop("calib_was_active", None)
            self.email_enabled.set(cfg.get("email_enabled", False))
            self.email_sender.set(cfg.get("email_sender", ""))
            self.email_password.set(cfg.get("email_password", ""))
            self.email_recipient.set(cfg.get("email_recipient", ""))
            self.ntfy_enabled.set(cfg.get("ntfy_enabled", False))
            self.ntfy_channel.set(cfg.get("ntfy_channel", ""))
            self.observer_lat.set(cfg.get("observer_lat", "48.8566"))
            self.observer_lon.set(cfg.get("observer_lon", "2.3522"))
            self.cam_hfov.set(cfg.get("cam_hfov", "60"))
            self.cam_heading.set(cfg.get("cam_heading", "0"))
            self.confirm_frames.set(cfg.get("confirm_frames", 3))
            if cfg.get("output_dir") and os.path.isdir(cfg["output_dir"]):
                self.output_dir.set(cfg["output_dir"])
            if cfg.get("model_path") and os.path.exists(cfg["model_path"]):
                self.model_path.set(cfg["model_path"])
            self.frame_skip.set(cfg.get("frame_skip", 1))
            self.danger_dist.set(cfg.get("danger_dist", 300))
            self.dist_floor.set(cfg.get("dist_floor", 0))
            global DANGER_DIST, MIN_DIST_FLOOR
            DANGER_DIST     = int(self.danger_dist.get())
            MIN_DIST_FLOOR  = int(self.dist_floor.get())
            self.calib_real_width.set(str(cfg.get("calib_real_width", "0.5")))
            self.calib_ref_distance.set(str(cfg.get("calib_ref_distance", "")))
            self.calib_pixel_width.set(str(cfg.get("calib_pixel_width", "")))
            if cfg.get("calib_focal") and cfg.get("calib_was_active"):
                global KNOWN_WIDTH_M, FOCAL_LENGTH, CALIBRATION_ACTIVE
                KNOWN_WIDTH_M      = float(cfg.get("calib_real_width", 0.5))
                FOCAL_LENGTH       = float(cfg["calib_focal"])
                CALIBRATION_ACTIVE = True
                self.calib_active.set(True)
                self.after(200, self._refresh_calib_status)
        except Exception as e:
            logger.warning(f"[WARN] Config non chargée: {e}")

    def _save_config(self):
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            self.CONFIG_FILE)
        cfg = {
            "units":              "m",          # v3 : marqueur d'unités
            "email_enabled":      self.email_enabled.get(),
            "email_sender":       self.email_sender.get(),
            "email_password":     self.email_password.get(),
            "email_recipient":    self.email_recipient.get(),
            "ntfy_enabled":       self.ntfy_enabled.get(),
            "ntfy_channel":       self.ntfy_channel.get(),
            "observer_lat":       self.observer_lat.get(),
            "observer_lon":       self.observer_lon.get(),
            "cam_hfov":           self.cam_hfov.get(),
            "cam_heading":        self.cam_heading.get(),
            "confirm_frames":     self.confirm_frames.get(),
            "output_dir":         self.output_dir.get(),
            "model_path":         self.model_path.get(),
            "frame_skip":         self.frame_skip.get(),
            "danger_dist":        self.danger_dist.get(),
            "calib_real_width":   self.calib_real_width.get(),
            "calib_ref_distance": self.calib_ref_distance.get(),
            "calib_pixel_width":  self.calib_pixel_width.get(),
            "calib_focal":        FOCAL_LENGTH,
            "calib_was_active":   CALIBRATION_ACTIVE,
            "dist_floor":         self.dist_floor.get(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[WARN] Config non sauvegardée: {e}")

    def _on_close(self):
        self._save_config()
        self.destroy()

    def _apply_calibration(self):
        """
        v3 : calibration en MÈTRES.
        F (pixels) = largeur_px × distance_m / largeur_m
        (les unités de longueur s'annulent, seule la cohérence compte)
        """
        global KNOWN_WIDTH_M, FOCAL_LENGTH, CALIBRATION_ACTIVE
        try:
            rw  = float(self.calib_real_width.get().replace(",", "."))
            rd  = float(self.calib_ref_distance.get().replace(",", "."))
            rpx = float(self.calib_pixel_width.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Calibration",
                                 "Tous les champs doivent être des nombres.")
            return
        if rw <= 0 or rd <= 0 or rpx <= 0:
            messagebox.showerror("Calibration", "Les trois valeurs doivent être > 0.")
            return
        new_focal          = (rpx * rd) / rw
        KNOWN_WIDTH_M      = rw
        FOCAL_LENGTH       = new_focal
        CALIBRATION_ACTIVE = True
        self.calib_active.set(True)
        self.calib_status_lbl.config(
            text=f"✅  F = {new_focal:.0f} px  |  W défaut = {rw:.2f} m", fg=ACCENT)
        self._log(f"[CALIB] ✅ F={new_focal:.1f}px  W défaut={rw:.2f}m "
                  f"(Shahed reste à {class_width_m('shahed'):.2f}m)", "success")

    def _reset_calibration(self):
        global KNOWN_WIDTH_M, FOCAL_LENGTH, CALIBRATION_ACTIVE
        KNOWN_WIDTH_M      = 0.5
        FOCAL_LENGTH       = 2000
        CALIBRATION_ACTIVE = False
        self.calib_active.set(False)
        self.calib_real_width.set("0.5")
        self.calib_ref_distance.set("")
        self.calib_pixel_width.set("")
        self.calib_status_lbl.config(
            text="⚙  Valeurs usine  (F = 2000 px  |  W défaut = 0.50 m)", fg=WARNING)
        self._log("[CALIB] Réinitialisé aux valeurs usine.", "warning")

    def _refresh_calib_status(self):
        if CALIBRATION_ACTIVE:
            self.calib_status_lbl.config(
                text=f"✅  F = {FOCAL_LENGTH:.0f} px  |  "
                     f"W défaut = {KNOWN_WIDTH_M:.2f} m  (restaurée)",
                fg=ACCENT)

    def _test_email(self):
        if not self.email_enabled.get():
            messagebox.showinfo("Info", "Active d'abord les alertes email.")
            return
        self._log("Envoi alerte email test...", "info")
        self.email_status_lbl.config(text="⏳ envoi...", fg=WARNING)
        def _run():
            ok, err = send_email_alert(
                smtp_host="smtp.gmail.com", smtp_port=465,
                sender=self.email_sender.get(),
                password=self.email_password.get(),
                recipient=self.email_recipient.get() or self.email_sender.get(),
                subject="[SHAHED DETECTOR] Alerte sécurité - Test",
                body_html="<html><body style='background:#080c10;color:#c9d1d9;"
                          "font-family:monospace;padding:20px'>"
                          "<h2 style='color:#00ff41'>✅ Shahed Detection System</h2>"
                          "<p>Configuration email <b>OK</b> !</p>"
                          "</body></html>")
            if ok:
                self.after(0, self._log, "[OK] Email test envoyé !", "success")
                self.after(0, self.email_status_lbl.config, {"text":"✅ OK","fg":ACCENT})
            else:
                self.after(0, self._log, f"[ERROR] Email: {err}", "error")
                self.after(0, self.email_status_lbl.config,
                           {"text":"❌ Erreur","fg":ACCENT2})
        threading.Thread(target=_run, daemon=True).start()

    def _send_alert_email(self, subject, body_html, image_path=None):
        if not self.email_enabled.get():
            return
        imgs = [image_path] if image_path else []
        def _run():
            ok, err = send_email_alert(
                smtp_host="smtp.gmail.com", smtp_port=465,
                sender=self.email_sender.get(),
                password=self.email_password.get(),
                recipient=self.email_recipient.get() or self.email_sender.get(),
                subject=subject, body_html=body_html, image_paths=imgs)
            if ok:
                self.after(0, self._log, f"[EMAIL ✅] {subject}", "success")
            else:
                self.after(0, self._log, f"[EMAIL ❌] {err}", "error")
        threading.Thread(target=_run, daemon=True).start()

    def _build_ui(self):
        header = tk.Frame(self, bg=BG2, pady=10,
                          highlightthickness=1, highlightbackground=ACCENT)
        header.pack(side="top", fill="x")
        tk.Label(header, text="◈  SHAHED DETECTION SYSTEM  ◈",
                 font=FONT_TITLE, bg=BG2, fg=ACCENT).pack()
        tk.Label(header,
                 text="AI surveillance · YOLOv8s · mAP@50=91.1% · bird / not / shahed · distances en mètres",
                 font=("Segoe UI", 10), bg=BG2, fg="#888888").pack()

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=6)

        left  = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # ── Ligne 1 — Video + Model ───────────────────────────────
        row_top = tk.Frame(left, bg=BG)
        row_top.pack(fill="x", pady=(0,6))

        vf = tk.Frame(row_top, bg=BG)
        vf.pack(side="left", fill="x", expand=True, padx=(0,6))
        tk.Label(vf, text="▸ VIDEO", font=FONT_CARD, bg=BG, fg="#bbbbbb").pack(anchor="w")
        vc = tk.Frame(vf, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        vc.pack(fill="x")
        vi = tk.Frame(vc, bg=CARD, padx=8, pady=6)
        vi.pack(fill="x")
        # ── Sélecteur de source ─────────────────────────────
        src_row = tk.Frame(vi, bg=CARD)
        src_row.pack(fill="x", pady=(0,4))
        for val, lbl in [("file","FICHIER"), ("webcam","WEBCAM"), ("rtsp","RTSP/IP")]:
            tk.Radiobutton(src_row, text=lbl, variable=self.source_type,
                           value=val, command=self._on_source_type,
                           font=("Segoe UI",9,"bold"), bg=CARD, fg=ACCENT,
                           selectcolor=BG3, activebackground=CARD,
                           activeforeground=ACCENT).pack(side="left", padx=(0,10))
        # Ligne fichier
        self.file_row = tk.Frame(vi, bg=CARD)
        self.file_row.pack(fill="x")
        vrow = tk.Frame(self.file_row, bg=CARD)
        vrow.pack(fill="x")
        self._entry(vrow, self.video_path).pack(side="left", fill="x", expand=True,
                                                ipady=5, padx=(0,6))
        self._btn(vrow, "BROWSE", self._pick_video).pack(side="right")
        self.video_info = tk.Label(self.file_row, text="Aucune vidéo sélectionnée",
                                    font=("Segoe UI",9), bg=CARD, fg="#888888")
        self.video_info.pack(anchor="w")
        # Ligne RTSP
        self.rtsp_row = tk.Frame(vi, bg=CARD)
        rtsp_inner = tk.Frame(self.rtsp_row, bg=CARD)
        rtsp_inner.pack(fill="x")
        tk.Label(rtsp_inner, text="URL :", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="left", padx=(0,4))
        self._entry(rtsp_inner, self.rtsp_url).pack(side="left", fill="x",
                                                     expand=True, ipady=5)
        tk.Label(self.rtsp_row,
                 text="ex: rtsp://192.168.1.10:554/stream  ou  0 pour webcam",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").pack(anchor="w")
        # Ligne webcam info
        self.webcam_row = tk.Frame(vi, bg=CARD)
        tk.Label(self.webcam_row,
                 text="Webcam locale (index 0)  —  appuie sur STOP pour terminer",
                 font=("Segoe UI",9), bg=CARD, fg="#4fc3f7").pack(anchor="w")

        mf = tk.Frame(row_top, bg=BG)
        mf.pack(side="left", fill="x", expand=True)
        tk.Label(mf, text="▸ MODÈLE", font=FONT_CARD, bg=BG, fg="#bbbbbb").pack(anchor="w")
        mc = tk.Frame(mf, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        mc.pack(fill="x")
        mi = tk.Frame(mc, bg=CARD, padx=8, pady=6)
        mi.pack(fill="x")
        mrow = tk.Frame(mi, bg=CARD)
        mrow.pack(fill="x")
        self._entry(mrow, self.model_path).pack(side="left", fill="x", expand=True,
                                                ipady=5, padx=(0,6))
        self._btn(mrow, "SELECT", self._pick_model, fg="#888888", bg=BG3).pack(side="right")

        # ── Ligne 2 — Output ──────────────────────────────────────
        of = tk.Frame(left, bg=BG)
        of.pack(fill="x", pady=(0,6))
        tk.Label(of, text="▸ DOSSIER SORTIE", font=FONT_CARD, bg=BG, fg="#bbbbbb").pack(anchor="w")
        oc = tk.Frame(of, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        oc.pack(fill="x")
        oi = tk.Frame(oc, bg=CARD, padx=8, pady=6)
        oi.pack(fill="x")
        orow = tk.Frame(oi, bg=CARD)
        orow.pack(fill="x")
        self._entry(orow, self.output_dir).pack(side="left", fill="x", expand=True,
                                                ipady=5, padx=(0,6))
        self._btn(orow, "SELECT", self._pick_output, fg="#888888", bg=BG3).pack(side="right")

        # ── Ligne 3 — Settings + GEO + CALIBRATION ───────────────
        row_mid = tk.Frame(left, bg=BG)
        row_mid.pack(fill="x", pady=(0,6))

        sf = tk.Frame(row_mid, bg=BG)
        sf.pack(side="left", fill="x", expand=True, padx=(0,6))
        tk.Label(sf, text="▸ PARAMÈTRES", font=FONT_CARD, bg=BG, fg="#bbbbbb").pack(anchor="w")
        sc = tk.Frame(sf, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        sc.pack(fill="x")
        si = tk.Frame(sc, bg=CARD, padx=8, pady=6)
        si.pack(fill="x")
        sr1 = tk.Frame(si, bg=CARD)
        sr1.pack(fill="x", pady=(0,3))
        tk.Label(sr1, text="Frame skip", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="left")
        self.skip_lbl = tk.Label(sr1, text="1", font=("Segoe UI",10),
                                  bg=CARD, fg=ACCENT, width=2)
        self.skip_lbl.pack(side="right")
        ttk.Scale(si, from_=1, to=8, variable=self.frame_skip, orient="h",
                  command=lambda v: self.skip_lbl.config(
                      text=str(int(float(v))))).pack(fill="x", pady=(0,4))
        sr2 = tk.Frame(si, bg=CARD)
        sr2.pack(fill="x", pady=(0,3))
        tk.Label(sr2, text="Dist. danger (m)", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="left")
        # v3.1 : champ éditable — taper une valeur exacte (ex: 303) + Entrée
        tk.Label(sr2, text="m", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="right")
        self.danger_spin = tk.Spinbox(
            sr2, from_=50, to=5000, increment=10,
            textvariable=self.danger_dist, width=6,
            font=("Segoe UI",10), bg=BG3, fg=ACCENT2,
            buttonbackground=BG3, insertbackground=ACCENT2,
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
            command=self._apply_danger_dist)
        self.danger_spin.pack(side="right", padx=(0,4))
        self.danger_spin.bind("<Return>",   lambda e: self._apply_danger_dist())
        self.danger_spin.bind("<FocusOut>", lambda e: self._apply_danger_dist())
        # v3 : seuil danger en mètres, 50 m → 2000 m
        ttk.Scale(si, from_=50, to=2000, variable=self.danger_dist,
                  orient="h", command=self._on_danger_dist).pack(fill="x")

        # v3.1 : distance plancher ─────────────────────────────────
        tk.Frame(si, bg=BORDER, height=1).pack(fill="x", pady=(6,4))
        sr3 = tk.Frame(si, bg=CARD)
        sr3.pack(fill="x", pady=(0,3))
        tk.Label(sr3, text="Dist. plancher (m)", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="left")
        tk.Label(sr3, text="m", font=("Segoe UI",10),
                 bg=CARD, fg="#aaaaaa").pack(side="right")
        self.floor_spin = tk.Spinbox(
            sr3, from_=0, to=5000, increment=50,
            textvariable=self.dist_floor, width=6,
            font=("Segoe UI",10), bg=BG3, fg="#4fc3f7",
            buttonbackground=BG3, insertbackground="#4fc3f7",
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
            command=self._apply_floor_dist)
        self.floor_spin.pack(side="right", padx=(0,4))
        self.floor_spin.bind("<Return>",   lambda e: self._apply_floor_dist())
        self.floor_spin.bind("<FocusOut>", lambda e: self._apply_floor_dist())
        tk.Label(si, text="0 = désactivé  |  ex: 200 pour vidéos de synthèse",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").pack(anchor="w")


        gf = tk.Frame(row_mid, bg=BG)
        gf.pack(side="left", fill="x", expand=True)
        tk.Label(gf, text="▸ GEO  +  CALIBRATION CAMÉRA", font=FONT_CARD,
                 bg=BG, fg="#bbbbbb").pack(anchor="w")
        gc = tk.Frame(gf, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        gc.pack(fill="x")
        gi = tk.Frame(gc, bg=CARD, padx=8, pady=8)
        gi.pack(fill="x")

        tk.Label(gi, text="GEO", font=("Segoe UI",9,"bold"),
                 bg=CARD, fg="#666666").pack(anchor="w", pady=(0,2))
        gr1 = tk.Frame(gi, bg=CARD)
        gr1.pack(fill="x", pady=(0,4))
        gr1.columnconfigure((1,3), weight=1)
        tk.Label(gr1, text="Lat:", font=("Segoe UI",10), bg=CARD,
                 fg="#aaaaaa").grid(row=0, column=0, sticky="w", padx=(0,4))
        self._entry(gr1, self.observer_lat).grid(row=0, column=1, sticky="ew",
                                                  ipady=4, padx=(0,8))
        tk.Label(gr1, text="Lon:", font=("Segoe UI",10), bg=CARD,
                 fg="#aaaaaa").grid(row=0, column=2, sticky="w", padx=(0,4))
        self._entry(gr1, self.observer_lon).grid(row=0, column=3, sticky="ew", ipady=4)
        gr2 = tk.Frame(gi, bg=CARD)
        gr2.pack(fill="x", pady=(0,8))
        tk.Label(gr2, text="FOV°:", font=("Segoe UI",10), bg=CARD,
                 fg="#aaaaaa").pack(side="left", padx=(0,4))
        self._entry(gr2, self.cam_hfov).pack(side="left", ipady=4, ipadx=12)
        tk.Label(gr2, text="(60=phone, 90=wide)",
                 font=("Segoe UI",9), bg=CARD, fg="#666666").pack(side="left", padx=(6,0))
        gr3 = tk.Frame(gi, bg=CARD)
        gr3.pack(fill="x", pady=(0,4))
        tk.Label(gr3, text="Cap caméra°:", font=("Segoe UI",10), bg=CARD,
                 fg="#aaaaaa").pack(side="left", padx=(0,4))
        self._entry(gr3, self.cam_heading).pack(side="left", ipady=4, ipadx=12)
        tk.Label(gr3, text="0=Nord  90=Est  180=Sud  270=Ouest",
                 font=("Segoe UI",9), bg=CARD, fg="#666666").pack(side="left", padx=(6,0))
        gr4 = tk.Frame(gi, bg=CARD)
        gr4.pack(fill="x", pady=(0,6))
        tk.Label(gr4, text="Confirmation (frames):", font=("Segoe UI",10), bg=CARD,
                 fg="#aaaaaa").pack(side="left", padx=(0,4))
        self.confirm_lbl = tk.Label(gr4, text="3", font=("Segoe UI",10),
                                    bg=CARD, fg=ACCENT, width=2)
        self.confirm_lbl.pack(side="right")
        ttk.Scale(gr4, from_=1, to=15, variable=self.confirm_frames, orient="h",
                  command=lambda v: self.confirm_lbl.config(
                      text=str(int(float(v))))).pack(side="left", fill="x",
                                                     expand=True, padx=(0,6))
        tk.Label(gi,
                 text="1=immédiat  3=anti-faux-positifs  5=très strict",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").pack(anchor="w", pady=(0,4))

        tk.Frame(gi, bg=BORDER, height=1).pack(fill="x", pady=(0,8))
        tk.Label(gi, text="CALIBRATION CAMÉRA  (en mètres)",
                 font=("Segoe UI",9,"bold"), bg=CARD, fg="#666666").pack(anchor="w", pady=(0,4))

        calib_grid = tk.Frame(gi, bg=CARD)
        calib_grid.pack(fill="x", pady=(0,6))
        calib_grid.columnconfigure((1,), weight=1)
        tk.Label(calib_grid, text="Largeur réelle (m) :",
                 font=("Segoe UI",10), bg=CARD, fg="#aaaaaa",
                 anchor="w").grid(row=0, column=0, sticky="w", padx=(0,8), pady=2)
        self._entry(calib_grid, self.calib_real_width).grid(
            row=0, column=1, sticky="ew", ipady=5)
        tk.Label(calib_grid, text="← largeur de l'objet de référence",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").grid(
            row=0, column=2, sticky="w", padx=(6,0))
        tk.Label(calib_grid, text="Distance caméra (m) :",
                 font=("Segoe UI",10), bg=CARD, fg="#aaaaaa",
                 anchor="w").grid(row=1, column=0, sticky="w", padx=(0,8), pady=2)
        self._entry(calib_grid, self.calib_ref_distance).grid(
            row=1, column=1, sticky="ew", ipady=5)
        tk.Label(calib_grid, text="← distance réelle caméra→objet",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").grid(
            row=1, column=2, sticky="w", padx=(6,0))
        tk.Label(calib_grid, text="Largeur pixels (px) :",
                 font=("Segoe UI",10), bg=CARD, fg="#aaaaaa",
                 anchor="w").grid(row=2, column=0, sticky="w", padx=(0,8), pady=2)
        self._entry(calib_grid, self.calib_pixel_width).grid(
            row=2, column=1, sticky="ew", ipady=5)
        tk.Label(calib_grid, text="← pixels dans la frame (ex: VLC)",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").grid(
            row=2, column=2, sticky="w", padx=(6,0))

        btn_row = tk.Frame(gi, bg=CARD)
        btn_row.pack(fill="x", pady=(0,6))
        self._btn(btn_row, "⚙  CALIBRER", self._apply_calibration,
                  bg="#0a3020", fg=ACCENT,
                  font=("Segoe UI",10,"bold")).pack(
            side="left", ipady=7, padx=(0,8), ipadx=6)
        self._btn(btn_row, "RESET", self._reset_calibration,
                  bg=BG3, fg=WARNING,
                  font=("Segoe UI",10,"bold")).pack(side="left", ipady=7, ipadx=6)

        self.calib_status_lbl = tk.Label(
            gi, text="⚙  Valeurs usine  (F = 2000 px  |  W défaut = 0.50 m)",
            font=("Segoe UI",9,"bold"), bg=CARD, fg=WARNING)
        self.calib_status_lbl.pack(anchor="w", pady=(0,2))
        tk.Label(gi, text="ℹ La largeur du Shahed (2,5 m) est codée par classe — "
                          "la calibration ajuste uniquement la focale.",
                 font=("Segoe UI",8), bg=CARD, fg="#555555").pack(anchor="w")

        # ── EMAIL ─────────────────────────────────────────────────
        ef = tk.Frame(left, bg=BG)
        ef.pack(fill="x", pady=(0,6))
        eh = tk.Frame(ef, bg=BG, cursor="hand2")
        eh.pack(fill="x", pady=(0,3))
        email_arrow = tk.StringVar(value="▼")
        tk.Label(eh, textvariable=email_arrow, font=("Segoe UI",9,"bold"),
                 bg=BG, fg=ACCENT, cursor="hand2").pack(side="left", padx=(0,4))
        tk.Label(eh, text="ALERTES EMAIL  —  Gmail", font=FONT_CARD,
                 bg=BG, fg="#bbbbbb", cursor="hand2").pack(side="left")
        ec_outer = tk.Frame(ef, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        ec_outer.pack(fill="x")

        def _toggle_email(c=ec_outer, a=email_arrow):
            if c.winfo_ismapped(): c.pack_forget(); a.set("▶")
            else: c.pack(fill="x"); a.set("▼")
        eh.bind("<Button-1>", lambda e: _toggle_email())
        for w in eh.winfo_children():
            w.bind("<Button-1>", lambda e: _toggle_email())

        ec = tk.Frame(ec_outer, bg=CARD, padx=10, pady=8)
        ec.pack(fill="x")
        tk.Checkbutton(ec, text="Activer alertes email (Shahed détecté + comportement)",
                       variable=self.email_enabled, font=("Segoe UI",10),
                       bg=CARD, fg=ACCENT, selectcolor=BG3,
                       activebackground=CARD, activeforeground=ACCENT).pack(anchor="w", pady=(0,6))
        eg = tk.Frame(ec, bg=CARD)
        eg.pack(fill="x")
        eg.columnconfigure(1, weight=1); eg.columnconfigure(3, weight=1)
        tk.Label(eg, text="Ton Gmail :", font=("Segoe UI",10),
                 bg=CARD, fg="white").grid(row=0, column=0, sticky="w", padx=(0,6), pady=3)
        self._entry(eg, self.email_sender).grid(row=0, column=1, sticky="ew",
                                                 ipady=5, padx=(0,12))
        tk.Label(eg, text="App Password :", font=("Segoe UI",10),
                 bg=CARD, fg="white").grid(row=0, column=2, sticky="w", padx=(0,6))
        tk.Entry(eg, textvariable=self.email_password, font=("Segoe UI",10),
                 bg=BG3, fg=ACCENT, show="●", insertbackground=ACCENT,
                 relief="flat", bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT).grid(
            row=0, column=3, sticky="ew", ipady=5)
        tk.Label(eg, text="Envoyer à :", font=("Segoe UI",10),
                 bg=CARD, fg="white").grid(row=1, column=0, sticky="w", padx=(0,6), pady=3)
        self._entry(eg, self.email_recipient).grid(row=1, column=1, sticky="ew",
                                                    ipady=5, padx=(0,12))
        self.email_status_lbl = tk.Label(eg, text="", font=("Segoe UI",9),
                                          bg=CARD, fg="#888888")
        self.email_status_lbl.grid(row=1, column=2, sticky="w")
        self._btn(eg, "▶ Test email", self._test_email, bg=BG3, fg="#4fc3f7",
                  font=("Segoe UI",10,"bold")).grid(row=1, column=3, sticky="ew", ipady=5)

        # ── NTFY ──────────────────────────────────────────────────
        nf = tk.Frame(left, bg=BG)
        nf.pack(fill="x", pady=(0,6))
        nh = tk.Frame(nf, bg=BG, cursor="hand2")
        nh.pack(fill="x", pady=(0,3))
        ntfy_arrow = tk.StringVar(value="▼")
        tk.Label(nh, textvariable=ntfy_arrow, font=("Segoe UI",9,"bold"),
                 bg=BG, fg=ACCENT, cursor="hand2").pack(side="left", padx=(0,4))
        tk.Label(nh, text="PUSH NOTIFICATIONS  —  Ntfy", font=FONT_CARD,
                 bg=BG, fg="#bbbbbb", cursor="hand2").pack(side="left")
        nc_outer = tk.Frame(nf, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        nc_outer.pack(fill="x")

        def _toggle_ntfy(c=nc_outer, a=ntfy_arrow):
            if c.winfo_ismapped(): c.pack_forget(); a.set("▶")
            else: c.pack(fill="x"); a.set("▼")
        nh.bind("<Button-1>", lambda e: _toggle_ntfy())
        for w in nh.winfo_children():
            w.bind("<Button-1>", lambda e: _toggle_ntfy())

        nc = tk.Frame(nc_outer, bg=CARD, padx=10, pady=8)
        nc.pack(fill="x")
        tk.Checkbutton(nc, text="Activer notifications push",
                       variable=self.ntfy_enabled, font=("Segoe UI",10),
                       bg=CARD, fg=ACCENT, selectcolor=BG3,
                       activebackground=CARD, activeforeground=ACCENT).pack(anchor="w", pady=(0,6))
        ng = tk.Frame(nc, bg=CARD)
        ng.pack(fill="x", pady=(0,4))
        ng.columnconfigure(1, weight=1)
        tk.Label(ng, text="Canal Ntfy :", font=("Segoe UI",10),
                 bg=CARD, fg="white").grid(row=0, column=0, sticky="w", padx=(0,8))
        self._entry(ng, self.ntfy_channel).grid(row=0, column=1, sticky="ew",
                                                 ipady=5, padx=(0,8))
        self._btn(ng, "▶ Test push", self._test_ntfy, bg=BG3, fg="#4fc3f7",
                  font=("Segoe UI",10,"bold")).grid(row=0, column=2, sticky="ew", ipady=5)

        # ── Actions ───────────────────────────────────────────────
        self._card_actions(left)

        # ── Colonne droite ────────────────────────────────────────
        self._card_progress(right)
        self._card_preview(right)
        self._card_geomap(right)
        self._card_logs(right)
        self._card_stats(right)

    def _card_geomap(self, parent):
        card = self._make_card(parent, "▸ GEO MAP  (positions estimées)")
        self.geomap_canvas = tk.Canvas(
            card, bg="#000000", width=220, height=220,
            highlightthickness=1, highlightbackground=BORDER)
        self.geomap_canvas.pack(pady=(0,4))
        self.geomap_canvas.create_text(
            110, 110, text="NO DATA", fill=TEXT_MUTED,
            font=("Courier New",10,"bold"), tags="gm_placeholder")

    def _card_actions(self, parent):
        card = self._make_card(parent, "")
        row  = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(0,8))
        self.btn_start = self._btn(row, "▶  LANCER LA DÉTECTION", self._start,
                                    bg=ACCENT, fg=BG, font=("Segoe UI",12,"bold"))
        self.btn_start.pack(side="left", fill="x", expand=True, ipady=10, padx=(0,6))
        self.btn_stop = self._btn(row, "STOP", self._stop,
                                   bg=ACCENT2, fg="white", state="disabled")
        self.btn_stop.pack(side="right", ipady=10, padx=(6,0))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=(0,8))

        row2 = tk.Frame(card, bg=CARD)
        row2.pack(fill="x", pady=(0,4))
        self.btn_open_video = self._btn(row2, "OUVRIR VIDÉO", self._open_video,
                                         bg=BG3, fg=ACCENT, font=FONT_BTN, state="disabled")
        self.btn_open_video.pack(side="left", fill="x", expand=True, ipady=7, padx=(0,6))
        self.btn_open_pdf = self._btn(row2, "OUVRIR PDF", self._open_pdf,
                                       bg=BG3, fg=WARNING, font=FONT_BTN, state="disabled")
        self.btn_open_pdf.pack(side="right", fill="x", expand=True, ipady=7, padx=(6,0))

        row3 = tk.Frame(card, bg=CARD)
        row3.pack(fill="x", pady=(0,4))
        self._btn(row3, "OUVRIR DOSSIER", self._open_dir,
                  bg=BG3, fg="#888888", font=FONT_BTN
                  ).pack(side="left", fill="x", expand=True, ipady=6, padx=(0,6))
        self.btn_sound = self._btn(row3, "SON ON", self._toggle_sound,
                                    bg=BG3, fg=ACCENT, font=FONT_BTN)
        self.btn_sound.pack(side="right", ipady=6)

        row4 = tk.Frame(card, bg=CARD)
        row4.pack(fill="x")
        self.btn_open_kml = self._btn(row4, "OUVRIR KML (Google Earth)",
                                       self._open_kml, bg=BG3, fg="#4fc3f7",
                                       font=FONT_BTN, state="disabled")
        self.btn_open_kml.pack(side="left", fill="x", expand=True, ipady=7, padx=(0,6))
        self.btn_open_csv = self._btn(row4, "OUVRIR CSV", self._open_csv,
                                       bg=BG3, fg=WARNING, font=FONT_BTN, state="disabled")
        self.btn_open_csv.pack(side="right", fill="x", expand=True, ipady=7, padx=(6,0))

    def _card_progress(self, parent):
        card = self._make_card(parent, "▸ PROGRESSION")
        row  = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(0,6))
        self.pct_lbl   = tk.Label(row, text="0%", font=("Courier New",16,"bold"),
                                   bg=CARD, fg=ACCENT)
        self.pct_lbl.pack(side="left")
        self.frame_lbl = tk.Label(row, text="FRAME 0 / -",
                                   font=FONT_LABEL, bg=CARD, fg=TEXT_MUTED)
        self.frame_lbl.pack(side="right")
        self.progress  = ttk.Progressbar(card, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(0,8))
        row2 = tk.Frame(card, bg=CARD)
        row2.pack(fill="x")
        self.drones_lbl = tk.Label(row2, text="OBJETS : 0",
                                    font=FONT_BTN, bg=CARD, fg=ACCENT)
        self.drones_lbl.pack(side="left")
        self.danger_lbl = tk.Label(row2, text="MENACES : 0",
                                    font=FONT_BTN, bg=CARD, fg=TEXT_MUTED)
        self.danger_lbl.pack(side="right")

    def _card_preview(self, parent):
        card = self._make_card(parent, "▸ LIVE PREVIEW")
        self.preview_canvas = tk.Canvas(
            card, bg="#000000", width=260, height=148,
            highlightthickness=1, highlightbackground=BORDER)
        self.preview_canvas.pack(pady=(0,4))
        self.preview_canvas.create_text(
            130, 74, text="NO SIGNAL", fill=TEXT_MUTED,
            font=FONT_BTN, tags="placeholder")

    def _card_logs(self, parent):
        card = self._make_card(parent, "▸ JOURNAL SYSTÈME")
        self.log_text = tk.Text(card, bg=BG, fg=ACCENT, font=FONT_MONO,
                                 relief="flat", bd=0, wrap="word",
                                 state="disabled", height=8,
                                 insertbackground=ACCENT)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("error",   foreground=ACCENT2)
        self.log_text.tag_config("success", foreground=ACCENT)
        self.log_text.tag_config("warning", foreground=WARNING)
        self.log_text.tag_config("info",    foreground=TEXT_MUTED)

    def _card_stats(self, parent):
        card = self._make_card(parent, "▸ RÉSULTATS")
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x")
        grid.columnconfigure((0,1), weight=1)
        self.stat_total  = self._stat(grid, "OBJETS DÉTECTÉS", "-", 0, 0)
        self.stat_danger = self._stat(grid, "SHAHED / MENACES", "-", 0, 1)
        self.stat_min    = self._stat(grid, "DIST. MIN",        "-", 1, 0)
        self.stat_speed  = self._stat(grid, "VITESSE MAX",      "-", 1, 1)
        grid2 = tk.Frame(card, bg=CARD)
        grid2.pack(fill="x")
        grid2.columnconfigure(0, weight=1)
        self.stat_fichiers = self._stat(grid2, "FICHIERS",      "-", 0, 0)

    def _make_card(self, parent, title, collapsed=False):
        wrap  = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(0,8))
        card  = tk.Frame(wrap, bg=CARD, highlightthickness=1,
                         highlightbackground=BORDER)
        inner = tk.Frame(card, bg=CARD, padx=12, pady=8)
        inner.pack(fill="x")

        if title:
            header    = tk.Frame(wrap, bg=BG, cursor="hand2")
            header.pack(fill="x", pady=(0,3))
            arrow_var = tk.StringVar(value="▼" if not collapsed else "▶")
            arrow_lbl = tk.Label(header, textvariable=arrow_var,
                                 font=("Segoe UI",9,"bold"),
                                 bg=BG, fg=ACCENT, cursor="hand2")
            arrow_lbl.pack(side="left", padx=(0,4))
            tk.Label(header, text=title, font=FONT_CARD,
                     bg=BG, fg="#bbbbbb", cursor="hand2").pack(side="left")

            def _toggle(c=card, a=arrow_var):
                if c.winfo_ismapped(): c.pack_forget(); a.set("▶")
                else: c.pack(fill="x"); a.set("▼")
            header.bind("<Button-1>", lambda e: _toggle())
            arrow_lbl.bind("<Button-1>", lambda e: _toggle())
            for child in header.winfo_children():
                child.bind("<Button-1>", lambda e: _toggle())

        if not collapsed:
            card.pack(fill="x")

        return inner

    def _entry(self, parent, var):
        return tk.Entry(parent, textvariable=var, font=FONT_MONO,
                        bg=BG3, fg=ACCENT, insertbackground=ACCENT,
                        relief="flat", bd=0, highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=ACCENT)

    def _btn(self, parent, text, cmd, bg=BG3, fg=ACCENT, font=FONT_BTN, state="normal"):
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font,
                         relief="flat", bd=0, cursor="hand2",
                         activebackground=ACCENT, activeforeground=BG,
                         state=state, padx=12)

    def _stat(self, parent, label, value, row, col):
        box = tk.Frame(parent, bg=BG3, padx=10, pady=8,
                       highlightthickness=1, highlightbackground=BORDER)
        box.grid(row=row, column=col, sticky="ew",
                 padx=(0, 6 if col==0 else 0), pady=(0,6))
        tk.Label(box, text=label, font=FONT_HINT, bg=BG3, fg="#bbbbbb").pack(anchor="w")
        lbl = tk.Label(box, text=value, font=FONT_STAT, bg=BG3, fg=ACCENT)
        lbl.pack(anchor="w")
        return lbl

    def _on_source_type(self):
        """Affiche/masque les widgets selon la source sélectionnée."""
        st = self.source_type.get()
        self.file_row.pack_forget()
        self.rtsp_row.pack_forget()
        self.webcam_row.pack_forget()
        if st == 'file':
            self.file_row.pack(fill='x')
        elif st == 'rtsp':
            self.rtsp_row.pack(fill='x')
        else:
            self.webcam_row.pack(fill='x')

    def _pick_video(self):
        p = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[("Vidéos","*.mp4 *.avi *.mov *.mkv"),("Tous","*.*")])
        if p:
            self.video_path.set(p)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(p))
            cap   = cv2.VideoCapture(p)
            fps   = int(cap.get(cv2.CAP_PROP_FPS)) or "?"
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            size = os.path.getsize(p)/(1024*1024)
            self.video_info.config(
                text=f"{w}x{h}  ·  {fps}fps  ·  {total} frames  ·  {size:.1f}Mo",
                fg=ACCENT)

    def _pick_model(self):
        p = filedialog.askopenfilename(title="Sélectionner un modèle .pt",
                                        filetypes=[("PyTorch model","*.pt"),("Tous","*.*")])
        if p:
            self.model_path.set(p)

    def _pick_output(self):
        p = filedialog.askdirectory(title="Dossier de sortie")
        if p:
            self.output_dir.set(p)

    def _on_danger_dist(self, val):
        """Slider → l'IntVar partagé met aussi à jour le champ chiffré."""
        global DANGER_DIST
        DANGER_DIST = int(float(val))

    def _apply_danger_dist(self):
        """Saisie clavier / flèches du Spinbox → valeur exacte (ex: 303)."""
        global DANGER_DIST
        try:
            v = max(1, int(self.danger_dist.get()))
        except Exception:
            return
        DANGER_DIST = v

    def _apply_floor_dist(self):
        """Applique la distance plancher (0 = désactivé)."""
        global MIN_DIST_FLOOR
        try:
            v = max(0, int(self.dist_floor.get()))
        except Exception:
            return
        MIN_DIST_FLOOR = v
        if v > 0:
            self._log(f"[INFO] Distance plancher activée : {v} m "
                      f"(toute détection < {v}m sera remontée à {v}m)", "warning")
        else:
            self._log("[INFO] Distance plancher désactivée.", "info")

    def _toggle_sound(self):
        self.sound_on = not self.sound_on
        if self.sound_on:
            self.btn_sound.config(text="SON ON",  fg=ACCENT)
        else:
            stop_alarm()
            self.btn_sound.config(text="SON OFF", fg=TEXT_MUTED)

    def _open_video(self):
        if self._last_video_out and os.path.exists(self._last_video_out):
            if sys.platform == "darwin":
                subprocess.call(["open", self._last_video_out])
            else:
                os.startfile(self._last_video_out)

    def _open_pdf(self):
        if self._last_pdf_out and os.path.exists(self._last_pdf_out):
            if sys.platform == "darwin":
                subprocess.call(["open", self._last_pdf_out])
            else:
                os.startfile(self._last_pdf_out)

    def _open_dir(self):
        d = self._last_run_dir or self.output_dir.get() or os.getcwd()
        if os.path.isdir(d):
            if sys.platform == "darwin":
                subprocess.call(["open", d])
            else:
                os.startfile(d)

    def _open_kml(self):
        if not (self._last_kml_out and os.path.exists(self._last_kml_out)):
            messagebox.showinfo("Info", "Aucun fichier KML.\nLancez d'abord une analyse.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.call(["open", self._last_kml_out])
            else:
                os.startfile(self._last_kml_out)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _open_csv(self):
        if self._last_csv_out and os.path.exists(self._last_csv_out):
            if sys.platform == "darwin":
                subprocess.call(["open", self._last_csv_out])
            else:
                os.startfile(self._last_csv_out)

    def _test_ntfy(self):
        ch = self.ntfy_channel.get().strip()
        if not ch:
            messagebox.showerror("Erreur", "Entre un nom de canal Ntfy.")
            return
        self._log("Envoi notification push test...", "info")
        def _run():
            ok, err = self._ntfy_post(ch,
                "[OK] Shahed Detector", "Configuration Ntfy OK !")
            if ok:
                self.after(0, self._log, "[NTFY ✅] Test envoyé !", "success")
            else:
                self.after(0, self._log, f"[NTFY ❌] {err}", "error")
        threading.Thread(target=_run, daemon=True).start()

    def _ntfy_post(self, channel, title, message, priority="high", image_path=None):
        try:
            url = f"https://ntfy.sh/{channel.strip()}"
            safe_message = message.replace("\n", " | ").replace("\r", "")
            safe_title   = title.replace("\n", " ").replace("\r", "")
            headers = {
                "Title":    safe_title.encode("utf-8"),
                "Priority": priority,
                "Tags":     "rotating_light",
                "Message":  safe_message.encode("utf-8"),
            }
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_data = f.read()
                headers["Filename"] = os.path.basename(image_path)
                resp = requests.post(url, data=img_data, headers=headers, timeout=10)
            else:
                resp = requests.post(url, data=safe_message.encode("utf-8"),
                                    headers=headers, timeout=10)
            return (True, "") if resp.status_code == 200 else (False, f"HTTP {resp.status_code}")
        except Exception as e:
            return False, str(e)

    def _send_ntfy_alert(self, title, message, image_path=None):
        if not self.ntfy_enabled.get():
            return
        ch = self.ntfy_channel.get().strip()
        if not ch:
            return
        def _run():
            ok, err = self._ntfy_post(ch, title, message, image_path=image_path)
            if ok:
                self.after(0, self._log, f"[NTFY ✅] {title}", "success")
            else:
                self.after(0, self._log, f"[NTFY ❌] {err}", "error")
        threading.Thread(target=_run, daemon=True).start()

    def _log(self, msg, level="info"):
        self.log_text.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n", level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _update_progress(self, pct, fid, total, n_drones, n_danger):
        if pct < 0:  # mode live
            self.progress["value"] = 0
            self.pct_lbl.config(text="LIVE")
            self.frame_lbl.config(text=f"FRAME {fid}")
        else:
            self.progress["value"] = pct
            self.pct_lbl.config(text=f"{pct}%")
            self.frame_lbl.config(text=f"FRAME {fid} / {total}")
        self.drones_lbl.config(text=f"OBJETS : {n_drones}")
        col = ACCENT2 if n_danger > 0 else TEXT_MUTED
        self.danger_lbl.config(text=f"MENACES : {n_danger}", fg=col)

    def _set_running(self, running):
        self.is_running = running
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal"    if running else "disabled")

    def _preview_cb_safe(self, frame, geo_positions=None):
        self._preview_counter += 1
        if self._preview_counter % 4 == 0:
            self.after(0, self._draw_preview, frame.copy())
        if geo_positions and self._preview_counter % 30 == 0:
            try:
                lat = float(self.observer_lat.get())
                lon = float(self.observer_lon.get())
            except ValueError:
                lat, lon = 48.8566, 2.3522
            self.after(0, self._update_geomap_live,
                       list(geo_positions), lat, lon)

    def _draw_preview(self, frame):
        if not PIL_AVAILABLE:
            return
        try:
            h, w     = frame.shape[:2]
            canvas_w = 260; canvas_h = 148
            scale    = min(canvas_w/w, canvas_h/h)
            nw, nh   = int(w*scale), int(h*scale)
            small    = cv2.resize(frame, (nw, nh))
            rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_img  = PILImage.fromarray(rgb)
            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self.preview_canvas.config(width=nw, height=nh)
            self.preview_canvas.delete("placeholder")
            self.preview_canvas.delete("img")
            self.preview_canvas.create_image(0, 0, anchor="nw",
                                              image=self._preview_photo, tags="img")
        except Exception:
            pass

    def _clear_preview(self):
        self.preview_canvas.delete("img")
        self.preview_canvas.config(width=260, height=148)
        self.preview_canvas.create_text(
            130, 74, text="NO SIGNAL", fill=TEXT_MUTED,
            font=FONT_BTN, tags="placeholder")

    def _update_geomap_live(self, geo_positions, observer_lat, observer_lon):
        if not PIL_AVAILABLE or not geo_positions:
            return
        try:
            import io
            # v3 : cercle de danger = vrai seuil DANGER_DIST en mètres
            all_dists_live = [p["dist"] for p in geo_positions if p["dist"] > 0]
            max_dist_live  = max(all_dists_live) if all_dists_live else float(DANGER_DIST)
            danger_m    = float(DANGER_DIST)
            view_m_live = max(max_dist_live * 1.2, danger_m * 1.5)
            size     = 280
            fig, ax  = plt.subplots(figsize=(size/72, size/72), dpi=72)
            fig.patch.set_facecolor("#000000")
            ax.set_facecolor("#0a0f14")

            def to_xy(lat, lon):
                dy = (lat-observer_lat) * EARTH_R * np.pi / 180
                dx = (lon-observer_lon) * EARTH_R * \
                     np.cos(np.radians(observer_lat)) * np.pi / 180
                return dx, dy

            for r in [view_m_live*0.25, view_m_live*0.5, view_m_live]:
                ax.add_patch(plt.Circle((0,0), r, color="#1a2332",
                             fill=False, linewidth=0.6, linestyle="--"))
            ax.add_patch(plt.Circle((0,0), danger_m, color="#ff3c3c",
                         fill=True, alpha=0.07, linewidth=1,
                         linestyle="-", edgecolor="#ff3c3c"))

            by_tid = {}
            for p in geo_positions:
                by_tid.setdefault(p["tid"], []).append(p)
            colors_plt = ["#00ff41","#00c8ff","#ffc800","#ff64c8","#64ffc8"]
            for i, (tid, pts) in enumerate(by_tid.items()):
                col = colors_plt[i % len(colors_plt)]
                xs  = [to_xy(p["lat"], p["lon"])[0] for p in pts]
                ys  = [to_xy(p["lat"], p["lon"])[1] for p in pts]
                ax.plot(xs, ys, color=col, lw=0.8, alpha=0.5)
                colors_pts = ["#ff0000" if p.get("name","")=="shahed" else
                              "#ff3c3c" if p["dist"] < DANGER_DIST else col
                              for p in pts]
                ax.scatter(xs, ys, c=colors_pts, s=4, zorder=3, alpha=0.6)
                if xs:
                    ax.scatter(xs[-1], ys[-1], color=col, s=40,
                               marker="^", zorder=5,
                               edgecolors="white", linewidths=0.4)

            ax.scatter(0, 0, color="white", s=60, zorder=6,
                       marker="o", edgecolors=ACCENT, linewidths=1.5)
            for lbl, x, y in [("N",0,view_m_live*0.9),("S",0,-view_m_live*0.9),
                               ("E",view_m_live*0.9,0),("W",-view_m_live*0.9,0)]:
                ax.text(x, y, lbl, color="#484f58", fontsize=5,
                        ha="center", va="center", fontweight="bold")

            lim = view_m_live
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal"); ax.axis("off")
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72,
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            pil_img = PILImage.open(buf).resize((size,size), PILImage.LANCZOS)
            self._geomap_photo = ImageTk.PhotoImage(pil_img)
            self.geomap_canvas.delete("gm_placeholder")
            self.geomap_canvas.delete("gm_img")
            self.geomap_canvas.create_image(0, 0, anchor="nw",
                                             image=self._geomap_photo,
                                             tags="gm_img")
        except Exception:
            pass

    def _clear_geomap(self):
        self.geomap_canvas.delete("gm_img")
        self.geomap_canvas.create_text(
            110, 110, text="NO DATA", fill=TEXT_MUTED,
            font=("Courier New",10,"bold"), tags="gm_placeholder")

    def _start(self):
        st = self.source_type.get()
        mp = self.model_path.get().strip()
        if not mp or not os.path.exists(mp):
            messagebox.showerror("Erreur", "Modèle .pt introuvable.")
            return
        # Déterminer la source vidéo
        if st == 'file':
            vp = self.video_path.get().strip()
            if not vp or not os.path.exists(vp):
                messagebox.showerror("Erreur", "Vidéo invalide.")
                return
            od = self.output_dir.get().strip() or os.path.dirname(vp)
        elif st == 'webcam':
            vp = 0   # index webcam
            od = self.output_dir.get().strip() or os.path.expanduser('~')
        else:  # rtsp
            vp = self.rtsp_url.get().strip()
            if not vp or vp == 'rtsp://':
                messagebox.showerror("Erreur", "Entrez une URL RTSP valide.")
                return
            od = self.output_dir.get().strip() or os.path.expanduser('~')
        self.output_dir.set(od)
        os.makedirs(od, exist_ok=True)
        self._apply_danger_dist()
        self._apply_floor_dist()
        self.stop_event.clear()
        self._preview_counter = 0
        self._set_running(True)
        mode_txt = {"file": "fichier", "webcam": "webcam", "rtsp": "flux RTSP"}[st]
        self._log(f"Démarrage — source : {mode_txt}", "info")
        for s in (self.stat_total, self.stat_danger, self.stat_min,
                  self.stat_speed, self.stat_fichiers):
            s.config(text="...")
        self._live_mode = (st in ('webcam', 'rtsp'))
        threading.Thread(target=self._pipeline, args=(vp, mp, od), daemon=True).start()

    def _stop(self):
        self.stop_event.set()
        self._log("Arrêt demandé...", "warning")

    def _pipeline(self, vp, mp, od):
        is_live   = getattr(self, '_live_mode', False)
        base      = (os.path.splitext(os.path.basename(str(vp)))[0]
                     if not is_live else
                     f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = os.path.join(od, f"{base}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        v_out      = os.path.join(run_dir, f"{base}_detected.mp4")
        pdf_out    = os.path.join(run_dir, f"{base}_report.pdf")
        png_out    = os.path.join(run_dir, f"{base}_charts.png")
        geomap_out = os.path.join(run_dir, f"{base}_geomap.png")

        def log(m, l="info"): self.after(0, self._log, m, l)
        def prog(p,f,t,n,d):  self.after(0, self._update_progress, p,f,t,n,d)

        log(f"Dossier : {run_dir}", "info")

        try:
            try:
                obs_lat  = float(self.observer_lat.get())
                obs_lon  = float(self.observer_lon.get())
                hfov     = float(self.cam_hfov.get())
                heading  = float(self.cam_heading.get())
            except ValueError:
                obs_lat, obs_lon, hfov, heading = 48.8566, 2.3522, 60.0, 0.0

            _geo_acc = []

            def _preview_with_geo(frame):
                self._preview_cb_safe(frame, _geo_acc)

            all_dist, timeline, all_det, danger_crops, geo_positions, behaviour_events = analyze_video(
                vp, v_out, mp, self.frame_skip.get(),
                log, prog, self.stop_event,
                sound_enabled=self.sound_on,
                preview_cb=_preview_with_geo,
                observer_lat=obs_lat,
                observer_lon=obs_lon,
                cam_hfov_deg=hfov,
                cam_heading_deg=heading,
                confirm_frames=self.confirm_frames.get())

            _geo_acc.clear()
            _geo_acc.extend(geo_positions)
            self.after(0, self._clear_preview)

            if not all_det:
                log("Aucune détection.", "warning")
                self.after(0, self._set_running, False)
                return

            log("Génération des graphiques...", "info")
            generate_charts(all_dist, timeline, png_out)

            geo_ok = False
            if geo_positions:
                log("Génération de la carte géo...", "info")
                geo_ok = generate_geomap(geo_positions, obs_lat, obs_lon,
                                         DANGER_DIST, geomap_out)
                if geo_ok:
                    self.after(0, self._update_geomap_live,
                               list(geo_positions), obs_lat, obs_lon)

            kml_out = os.path.join(run_dir, f"{base}_trajectory.kml")
            csv_out = os.path.join(run_dir, f"{base}_trajectory.csv")
            if geo_positions:
                try:
                    fps_val = int(cv2.VideoCapture(vp).get(cv2.CAP_PROP_FPS)) or 25
                    export_kml(geo_positions, obs_lat, obs_lon, fps_val, kml_out)
                    export_csv(geo_positions, fps_val, csv_out)
                    self._last_kml_out = kml_out
                    self._last_csv_out = csv_out
                    self.after(0, self.btn_open_kml.config, {"state":"normal"})
                    self.after(0, self.btn_open_csv.config, {"state":"normal"})
                    log(f"[OK] KML + CSV → {os.path.basename(kml_out)}", "success")
                except Exception as e:
                    log(f"[WARN] KML/CSV : {e}", "warning")

            log("Génération du rapport PDF...", "info")
            export_pdf(all_dist, all_det, png_out, pdf_out,
                       danger_crops=danger_crops,
                       crops_dir=run_dir,
                       geomap_path=geomap_out if geo_ok else None)

            # ── Alertes email ─────────────────────────────────────
            if self.email_enabled.get():
                shahed_crops = {tid: v for tid,v in danger_crops.items()
                                if v.get("name","") == "shahed"}
                alert_crops  = shahed_crops if shahed_crops else danger_crops
                if alert_crops:
                    best     = min(alert_crops.values(), key=lambda x: x["dist"])
                    crop_img = os.path.join(run_dir, "danger_drone_best.png")
                    ts_str   = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    threat_label = "⚠ SHAHED DÉTECTÉ" if best.get("name","")=="shahed" \
                                   else "🚨 MENACE DÉTECTÉE"
                    html = f"""
                    <html><body style="background:#080c10;color:#c9d1d9;font-family:monospace">
                    <h2 style="color:#ff3c3c">{threat_label}</h2>
                    <p><b>Heure :</b> {ts_str}</p>
                    <p><b>Classe :</b> {best['name'].upper()}</p>
                    <p><b>Distance :</b> {best['dist']:.0f} m</p>
                    <p><b>Frame :</b> {best['frame']}</p>
                    <hr/>
                    <img src="cid:img0" width="400" style="border:2px solid #ff3c3c"/>
                    <p style="color:#484f58;font-size:11px">
                    Shahed Detection System — alexandre196</p>
                    </body></html>"""
                    self._send_alert_email(
                        f"[SHAHED ALERT] {best['name'].upper()} — {best['dist']:.0f}m",
                        html, crop_img)

                for ev in behaviour_events:
                    ts_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    html = f"""
                    <html><body style="background:#080c10;color:#c9d1d9;font-family:monospace">
                    <h2 style="color:#ffb700">⚠️ COMPORTEMENT SUSPECT: {ev['behaviour']}</h2>
                    <p><b>Heure :</b> {ts_str}</p>
                    <p><b>ID :</b> #{ev['tid']} — {ev['name'].upper()}</p>
                    <p><b>Distance :</b> {ev['dist']:.0f} m</p>
                    </body></html>"""
                    self._send_alert_email(
                        f"[SHAHED ALERT] Comportement: {ev['behaviour']}",
                        html)

            # ── Ntfy ──────────────────────────────────────────────
            if self.ntfy_enabled.get() and self.ntfy_channel.get().strip():
                if danger_crops:
                    best     = min(danger_crops.values(), key=lambda x: x["dist"])
                    crop_img = os.path.join(run_dir, "danger_drone_best.png")
                    ts_str   = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self._send_ntfy_alert(
                        title=f"[{best['name'].upper()}] DÉTECTÉ — {best['dist']:.0f}m",
                        message=f"{ts_str}\nClasse: {best['name']}\n"
                                f"Distance: {best['dist']:.0f} m",
                        image_path=crop_img if os.path.exists(crop_img) else None)

            # ── Stats finales ─────────────────────────────────────
            valid = [d for d in all_dist if d > 0]
            from collections import Counter
            frames_count   = Counter(d["frame"] for d in all_det)
            drones_uniques = Counter(frames_count.values()).most_common(1)[0][0] \
                             if frames_count else 0
            danger_per_frame = Counter(d["frame"] for d in all_det
                                       if 0 < d["distance"] < DANGER_DIST)
            drones_danger  = Counter(danger_per_frame.values()).most_common(1)[0][0] \
                             if danger_per_frame else 0
            shahed_count   = sum(1 for d in all_det if d.get("name","")=="shahed")
            min_d  = min(valid) if valid else 0
            max_spd = max((p.get("speed_kmh",0) for p in geo_positions), default=0)

            self._last_video_out = v_out
            self._last_pdf_out   = pdf_out
            self._last_run_dir   = run_dir
            self.after(0, self.btn_open_video.config, {"state":"normal"})
            self.after(0, self.btn_open_pdf.config,   {"state":"normal"})
            self.after(0, self.stat_total.config,
                       {"text": str(drones_uniques), "fg": ACCENT})
            self.after(0, self.stat_danger.config,
                       {"text": f"{shahed_count} shahed / {drones_danger} menaces",
                        "fg": ACCENT2 if shahed_count > 0 else (
                              WARNING if drones_danger > 0 else ACCENT)})
            self.after(0, self.stat_min.config,
                       {"text": f"{min_d:.0f} m", "fg": ACCENT})
            self.after(0, self.stat_speed.config,
                       {"text": f"{max_spd:.1f} km/h",
                        "fg": ACCENT2 if max_spd > 30 else ACCENT})
            n_files = (3 + (1 if danger_crops else 0)
                       + (1 if geo_ok else 0)
                       + (2 if geo_positions else 0))
            self.after(0, self.stat_fichiers.config,
                       {"text": f"{n_files} fichiers", "fg": ACCENT})
            self.after(0, self._update_progress,
                       100, len(all_det), len(all_det), 0, 0)
            log(f"TERMINÉ — {drones_uniques} objets · {shahed_count} shahed · "
                f"dist min {min_d:.0f}m", "success")

        except Exception as exc:
            import traceback
            log(f"[ERREUR] {exc}", "error")
            log(traceback.format_exc(), "error")
            self.after(0, self._clear_preview)
        finally:
            self.after(0, self._set_running, False)


if __name__ == "__main__":
    try:
        app = DroneApp()
        app.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nERREUR - Appuyez sur Entrée pour fermer...")