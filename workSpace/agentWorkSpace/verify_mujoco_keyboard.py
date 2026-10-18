"""End-to-end check of the MuJoCo keyboard control (change 7).

Runs the REAL deploy script (deploy.deploy_mujoco.run with the shipped go2.yaml,
viewer and all) and drives it with real X11 key events via XTest, then reports
what the robot actually did in each time window. Nothing is stubbed: we only
wrap mj_step / build_observation to record the trajectory.

Needs a working DISPLAY and libXtst. The keys are addressed to whatever window
has the input focus, so this script gives the viewer the focus first and refuses
to send anything if that fails.

    DISPLAY=:1 python verify_mujoco_keyboard.py
"""
import ctypes
import sys
import threading
import time

sys.path.insert(0, "/home/ubuntu/B4_2025/unitree_rl_gym")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import deploy.deploy_mujoco.deploy_mujoco as dm  # noqa: E402

# ---------------------------------------------------------------- recording ---
latest = {}
records = []          # (sim_time, cmd, v_body, x, y, z, yaw)

_orig_step = mujoco.mj_step


def _step(m, d):
    _orig_step(m, d)
    latest["d"] = d


mujoco.mj_step = _step

_orig_build = dm.build_observation


def _build(config, action, cmd, qj, dqj, quat, omega, base_lin_vel, sim_time):
    d = latest.get("d")
    if d is not None:
        qpos = np.array(d.qpos)
        v_body = dm.quat_rotate_inverse(qpos[3:7], np.array(d.qvel[0:3]))
        qw, qx, qy, qz = qpos[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        records.append((sim_time, np.array(cmd, dtype=float), v_body,
                        qpos[0], qpos[1], qpos[2], yaw))
    return _orig_build(config, action, cmd, qj, dqj, quat, omega, base_lin_vel, sim_time)


dm.build_observation = _build

# ------------------------------------------------------------------- XTest ----
x11 = ctypes.CDLL("libX11.so.6")
xtst = ctypes.CDLL("libXtst.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
x11.XKeysymToKeycode.restype = ctypes.c_ubyte
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
x11.XGetInputFocus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int)]
x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                           ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                           ctypes.POINTER(ctypes.c_uint)]
x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x11.XFree.argtypes = [ctypes.c_void_p]
x11.XFlush.argtypes = [ctypes.c_void_p]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

dpy = x11.XOpenDisplay(None)


def _find(win, depth=0):
    out = []
    if depth > 4:
        return out
    root, parent = ctypes.c_ulong(), ctypes.c_ulong()
    kids = ctypes.POINTER(ctypes.c_ulong)()
    n = ctypes.c_uint()
    if not x11.XQueryTree(dpy, win, ctypes.byref(root), ctypes.byref(parent),
                          ctypes.byref(kids), ctypes.byref(n)):
        return out
    if kids:
        for i in range(n.value):
            nm = ctypes.c_char_p()
            if x11.XFetchName(dpy, kids[i], ctypes.byref(nm)) and nm.value:
                out.append((kids[i], nm.value.decode("utf-8", "replace")))
            out.extend(_find(kids[i], depth + 1))
        x11.XFree(kids)
    return out


def _focus():
    w, r = ctypes.c_ulong(), ctypes.c_int()
    x11.XGetInputFocus(dpy, ctypes.byref(w), ctypes.byref(r))
    return w.value


def _key(name):
    return x11.XKeysymToKeycode(dpy, x11.XStringToKeysym(name.encode()))


def send(name, down):
    xtst.XTestFakeKeyEvent(dpy, _key(name), 1 if down else 0, 0)
    x11.XFlush(dpy)


# ------------------------------------------------------------------ schedule --
# (simulated second, action, keys) -- driven by SIM time so it is immune to the
# loop lagging behind the wall clock.
SCHEDULE = [
    (2.0, True, ["Up"]),
    (5.0, False, ["Up"]),
    (7.0, True, ["Right"]),
    (10.0, False, ["Right"]),
    (12.0, True, ["k"]),
    (15.0, False, ["k"]),
    (17.0, True, ["Up", "Left"]),
    (20.0, False, ["Up", "Left"]),
]

WINDOWS = [
    ("idle (start)", 1.0, 2.0, None),
    ("Up  = forward", 2.5, 5.0, "Up"),
    ("idle after Up", 5.5, 7.0, "release"),
    ("Right = strafe right", 7.5, 10.0, "Right"),
    ("idle after Right", 10.5, 12.0, "release"),
    ("k   = turn left", 12.5, 15.0, "k"),
    ("idle after k", 15.5, 17.0, "release"),
    ("Up+Left = diagonal", 17.5, 20.0, "Up+Left"),
    ("idle after diagonal", 20.5, 22.0, "release"),
]

cfg = dm.load_config("go2.yaml")
cfg["simulation_duration"] = 22.0

result = {}


def run_deploy():
    result["d"] = dm.run(cfg, headless=False)


thread = threading.Thread(target=run_deploy)
thread.start()

# find + focus the viewer window
win = None
for _ in range(200):
    for w, n in _find(x11.XDefaultRootWindow(dpy)):
        if "mujoco" in n.lower():
            win, name = w, n
    if win:
        break
    time.sleep(0.1)

print("viewer window:", repr(name), flush=True)
x11.XSetInputFocus(dpy, win, 1, 0)
x11.XFlush(dpy)
time.sleep(0.3)
assert _focus() == win, "viewer does not have the keyboard focus; refusing to send keys"

print("driving with XTest key events...", flush=True)
pending = list(SCHEDULE)
while thread.is_alive():
    d = latest.get("d")
    if d is not None and pending:
        sim_time = float(d.time)
        while pending and pending[0][0] <= sim_time:
            _, down, keys = pending.pop(0)
            for k in keys:
                send(k, down)
            print("  t=%5.2fs  %-7s %s" % (sim_time, "PRESS" if down else "RELEASE",
                                           " ".join(keys)), flush=True)
    time.sleep(0.01)
thread.join()

for k in ["Up", "Down", "Left", "Right", "k", "l"]:
    send(k, False)  # never leave a key stuck

# ------------------------------------------------------------------ report ----
print("\n=== what the robot did (measured from the recorded trajectory) ===", flush=True)
print("%-22s %-22s %-20s %s" % ("window", "command [vx vy wz]", "measured [vx vy]", "yaw rate / travel"))
for label, t0, t1, _ in WINDOWS:
    sel = [r for r in records if t0 <= r[0] <= t1]
    if not sel:
        print("%-22s (no samples)" % label)
        continue
    arr = np.array([r[1] for r in sel])
    vel = np.array([r[2] for r in sel])
    yaw = np.array([r[6] for r in sel])
    dx = sel[-1][3] - sel[0][3]
    dy = sel[-1][4] - sel[0][4]
    dyaw = np.degrees(yaw[-1] - yaw[0])
    print("%-22s [%+.2f %+.2f %+.2f]      [%+.2f %+.2f]        %+6.1f deg  (%+.2f,%+.2f) m"
          % (label, arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(),
             vel[:, 0].mean(), vel[:, 1].mean(), dyaw, dx, dy))

d = result["d"]
print("\nfinal base position: [%+.3f %+.3f %+.3f]" % tuple(d.qpos[:3]))
print("total travel: %.2f m" % np.hypot(d.qpos[0], d.qpos[1]))
