#!/usr/bin/env python3
"""Flash an image on screen for a moment, then close — a demo aid for look().

Borderless, always-on-top, auto-closes after N ms. Meant to be spawned detached
by see_camera._show_annotated so the assistant loop is never blocked:

    python3 _popup.py <image.png> [ms] [x] [y] [scale]

x/y place the window's top-left corner. A NEGATIVE value anchors to the opposite
screen edge instead (like CSS): x=0 is the left edge, y=-60 is 60 px up from the
bottom — so "0 -60" is the lower-left corner. scale enlarges/shrinks the image
(default 1.0), applied via Tk zoom/subsample so no Pillow is needed here.

Leading underscore keeps the assistant's tool loader from importing it as a tool.
Tk PhotoImage reads PNG on Tcl/Tk >= 8.6, so no Pillow/ImageTk is needed here.
"""
import sys
import tkinter as tk
from fractions import Fraction


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    x = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    y = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    root = tk.Tk()
    root.overrideredirect(True)              # no title bar / borders
    root.attributes("-topmost", True)        # above the terminal/video
    try:
        img = tk.PhotoImage(file=path)
    except tk.TclError:
        root.destroy()
        return
    # Resize via integer zoom/subsample (Tk has no float scale): approximate the
    # factor as a small fraction, e.g. 1.5 -> zoom(3).subsample(2).
    if abs(scale - 1.0) > 1e-3:
        fr = Fraction(scale).limit_denominator(8)
        if fr.numerator != 1:
            img = img.zoom(fr.numerator)
        if fr.denominator != 1:
            img = img.subsample(fr.denominator)
    # Negative x/y anchor to the far edge: px = screen - image + x (x<0).
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    px = x if x >= 0 else max(0, sw - img.width() + x)
    py = y if y >= 0 else max(0, sh - img.height() + y)
    root.geometry(f"+{px}+{py}")
    tk.Label(root, image=img, borderwidth=0, highlightthickness=0).pack()
    root.after(ms, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
