import base64
import io
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# renderer: lightweight Catmull-Rom routines (same idea as earlier)
def catmull_rom_segment(p0, p1, p2, p3, n_points=20):
    t = np.linspace(0, 1, n_points)[:, None]
    a = 2 * p1
    b = -p0 + p2
    c = 2*p0 - 5*p1 + 4*p2 - p3
    d = -p0 + 3*p1 - 3*p2 + p3
    points = 0.5 * (a + b * t + c * (t**2) + d * (t**3))
    return points

def catmull_rom_chain(points, n_points_per_seg=24, closed=True):
    pts = np.asarray(points)
    if pts.shape[0] < 2:
        return pts.copy()
    if closed:
        extended = np.vstack([pts[-1], pts, pts[0], pts[1]])
        seg_indices = range(1, len(pts)+1)
    else:
        extended = np.vstack([pts[0], pts, pts[-1]])
        seg_indices = range(1, len(pts))
    chain = []
    for i in seg_indices:
        p0 = extended[i-1]
        p1 = extended[i]
        p2 = extended[i+1]
        p3 = extended[i+2]
        seg = catmull_rom_segment(p0, p1, p2, p3, n_points=n_points_per_seg)
        chain.append(seg)
    return np.vstack(chain)

def grid_points(rows, cols, spacing=1.0, offset=(0,0)):
    sx = spacing
    ys = (np.arange(rows) - (rows-1)/2.0) * sx + offset[1]
    xs = (np.arange(cols) - (cols-1)/2.0) * sx + offset[0]
    pts = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append([xs[c], -ys[r]])
        pts.append(row)
    return np.array(pts)

def ring_around_dots(bin_matrix, spacing=1.0, radius=0.3, n_circle_pts=24):
    rows, cols = bin_matrix.shape
    gp = grid_points(rows, cols, spacing=spacing)
    curves = []
    angles = np.linspace(0, 2*np.pi, n_circle_pts, endpoint=False)
    for i in range(rows):
        for j in range(cols):
            if int(bin_matrix[i, j]) == 1:
                cx, cy = gp[i, j]
                circle = np.stack([cx + radius*np.cos(angles), cy + radius*np.sin(angles)], axis=1)
                curves.append(catmull_rom_chain(circle, n_points_per_seg=4, closed=True))
    return curves

def draw_curves_to_image(curves, show_grid=None, figsize=(6,6), linewidth=2.0):
    fig, ax = plt.subplots(figsize=figsize)
    for c in curves:
        ax.plot(c[:,0], c[:,1], color='black', linewidth=linewidth, solid_capstyle='round')
    if show_grid is not None:
        rows, cols, spacing = show_grid
        gp = grid_points(rows, cols, spacing=spacing)
        ax.plot(gp[:,:,0].ravel(), gp[:,:,1].ravel(), 'ro', markersize=3, alpha=0.7)
    ax.set_aspect('equal')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    return img

def image_to_base64(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return 'data:image/png;base64,' + b64

def load_image_to_tensor(path, size=384, device='cpu'):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().to(device)
    return tensor

def tensor_to_pil(img_tensor):
    t = img_tensor.detach().cpu().squeeze(0)
    t = t.permute(1,2,0).numpy()
    t = (np.clip(t, -1, 1) + 1) / 2.0 if t.min() < -0.1 else t
    t = (t*255).astype(np.uint8)
    return Image.fromarray(t)

def heuristic_symmetry_classifier(bin_matrix):
    m = np.array(bin_matrix)
    r,c = m.shape
    score = 0.0
    # horizontal
    if r>1:
        hsim = np.mean(m == np.flipud(m))
        score += hsim
    # vertical
    if c>1:
        vsim = np.mean(m == np.fliplr(m))
        score += vsim
    score /= 2.0 if (r>1 and c>1) else 1.0
    label = 'symmetric' if score > 0.75 else 'asymmetric'
    return {'label': label, 'score': float(score)}
