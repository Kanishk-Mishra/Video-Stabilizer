import numpy as np
from scipy.signal import savgol_filter, medfilt

def robust_smooth(xs, median_ks=5, savgol_win=31, savgol_poly=3):
    n = len(xs)
    if n == 0:
        return xs
    mk = min(median_ks, n if n%2==1 else n-1)
    if mk < 1: mk = 1
    xs_med = medfilt(xs, kernel_size=mk) if mk>1 else xs.copy()
    win = min(savgol_win, n if n%2==1 else n-1)
    if win < 3: return xs_med
    poly = min(savgol_poly, win-1)
    try:
        xs_sg = savgol_filter(xs_med, window_length=win, polyorder=poly)
    except Exception:
        return xs_med
    return xs_sg

def compute_shifts(centers, W, H, stabilize_x_only=False, target="center"):
    N = len(centers)
    xs = centers[:,0].astype(np.float32)
    ys = centers[:,1].astype(np.float32)
    xs_s = robust_smooth(xs, median_ks=5, savgol_win=max(5, min(51, N//3|1)), savgol_poly=3)
    ys_s = ys.copy() if stabilize_x_only else robust_smooth(ys, median_ks=5, savgol_win=max(5, min(51, N//3|1)), savgol_poly=3)
    if target=="center":
        tx, ty = W//2, H//2
    else:
        tx, ty = target
    shifts = np.zeros((N,2), dtype=np.float32)
    shifts[:,0] = tx - xs_s
    shifts[:,1] = ty - ys_s
    return shifts
