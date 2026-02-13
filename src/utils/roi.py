def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return (x1, y1, x2, y2)

def inside_ratio(bbox, safe):
    inter = _intersection(bbox, safe)
    return _area(inter) / (_area(bbox) + 1e-9)

def safe_box(frame_shape, border_pct: float):
    H, W = frame_shape[:2]
    mx = int(border_pct * W)
    my = int(border_pct * H)
    return (mx, my, W - mx, H - my)
