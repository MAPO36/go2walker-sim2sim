"""Which keyboard does MuJoCo leave to us? (used to pick the key map in change 7)

(a) is X11's XQueryKeymap a reliable 'is the key held down' signal?
(b) which keys does the MuJoCo viewer consume for its own shortcuts?

(a) press Up and poll XQueryKeymap at ~200 Hz for 2 s while it is held.
(b) for each candidate key, snapshot the viewer's own state (render flags,
    geom groups, camera, UI panels) before and after, and report changes.
"""
import ctypes
import sys
import threading
import time

import mujoco
import mujoco.viewer

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
x11.XFetchName.restype = ctypes.c_int
x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
x11.XQueryTree.restype = ctypes.c_int
x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                           ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                           ctypes.POINTER(ctypes.c_uint)]
x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x11.XQueryKeymap.restype = ctypes.c_int
x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
x11.XFree.argtypes = [ctypes.c_void_p]
x11.XFlush.argtypes = [ctypes.c_void_p]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

dpy = x11.XOpenDisplay(None)
KEYS = ctypes.create_string_buffer(32)


def key_down(code):
    x11.XQueryKeymap(dpy, KEYS)
    return (KEYS.raw[code >> 3] >> (code & 7)) & 1


def kc(name):
    return x11.XKeysymToKeycode(dpy, x11.XStringToKeysym(name.encode()))


def press(name):
    xtst.XTestFakeKeyEvent(dpy, kc(name), 1, 0)
    x11.XFlush(dpy)


def release(name):
    xtst.XTestFakeKeyEvent(dpy, kc(name), 0, 0)
    x11.XFlush(dpy)


def find_window(win, depth=0):
    root, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    n = ctypes.c_uint()
    out = []
    if depth > 4 or not x11.XQueryTree(dpy, win, ctypes.byref(root), ctypes.byref(parent),
                                       ctypes.byref(children), ctypes.byref(n)):
        return out
    if children:
        for i in range(n.value):
            nm = ctypes.c_char_p()
            if x11.XFetchName(dpy, children[i], ctypes.byref(nm)) and nm.value:
                out.append((children[i], nm.value.decode("utf-8", "replace")))
            out.extend(find_window(children[i], depth + 1))
        x11.XFree(children)
    return out


def current_focus():
    w, r = ctypes.c_ulong(), ctypes.c_int()
    x11.XGetInputFocus(dpy, ctypes.byref(w), ctypes.byref(r))
    return w.value


def snapshot(h):
    sim = h._get_sim()
    return {
        "opt.flags": tuple(int(v) for v in h.opt.flags),
        "geomgroup": tuple(int(g) for g in h.opt.geomgroup),
        "sitegroup": tuple(int(g) for g in h.opt.sitegroup),
        "cam.type": int(h.cam.type),
        "cam.fixedcamid": int(h.cam.fixedcamid),
        "ui_left": bool(sim.ui0_enable),
        "ui_right": bool(sim.ui1_enable),
        "refresh_rate": float(sim.refresh_rate),
        "busywait": bool(sim.busywait),
    }


model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data, key_callback=lambda c: None) as h:
    time.sleep(2.0)
    win = nm = None
    for w, n in find_window(x11.XDefaultRootWindow(dpy)):
        if "mujoco" in n.lower():
            win, nm = w, n
            break
    print("window:", repr(nm), flush=True)
    if win is None:
        print("no window; abort", flush=True)
        sys.exit(1)
    x11.XSetInputFocus(dpy, win, 1, 0)
    x11.XFlush(dpy)
    time.sleep(0.3)
    if current_focus() != win:
        print("focus not on viewer (%s); abort" % hex(current_focus()), flush=True)
        sys.exit(1)

    # ---------------- (a) hold test -------------------------------------
    code = kc("Up")
    print("\n=== XQueryKeymap reliability, key 'Up' keycode=%d ===" % code, flush=True)
    press("Up")
    t0 = time.time()
    samples = []
    while time.time() - t0 < 2.0:
        samples.append((time.time() - t0, key_down(code)))
        time.sleep(0.005)
    release("Up")
    tail = []
    t1 = time.time()
    while time.time() - t1 < 0.5:
        tail.append(key_down(code))
        time.sleep(0.005)
    down = sum(1 for _, v in samples if v)
    print("while held : %d/%d samples reported DOWN (%.1f%%)"
          % (down, len(samples), 100.0 * down / len(samples)), flush=True)
    worst = 0.0
    streak = None
    for t, v in samples:
        if not v:
            streak = streak or t
        elif streak is not None:
            worst = max(worst, t - streak)
            streak = None
    print("longest reported-UP gap while physically held: %.1f ms" % (worst * 1000), flush=True)
    print("after release: %d/%d samples DOWN (expect 0)" % (sum(1 for v in tail if v), len(tail)), flush=True)

    # ---------------- (b) which keys does the viewer consume ------------
    print("\n=== keys the viewer itself reacts to ===", flush=True)
    for name in ["Up", "Down", "Left", "Right", "q", "e", "z", "x", "c", "v",
                 "b", "n", "m", "comma", "period", "slash", "semicolon", "space",
                 "Tab", "bracketleft", "bracketright", "h", "i", "p", "s", "f", "g"]:
        before = snapshot(h)
        press(name)
        time.sleep(0.05)
        release(name)
        time.sleep(0.15)
        h.sync()
        after = snapshot(h)
        diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        if diff:
            print("  %-14s CHANGED %s" % (name, diff), flush=True)
        time.sleep(0.05)
    print("  (keys not listed above had no observable effect)", flush=True)
print("done", flush=True)
