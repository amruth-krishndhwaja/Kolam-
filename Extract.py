import cv2
import numpy as np
from skimage.morphology import skeletonize

def _binarize_image(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def _find_dot_centroids(bin_img, min_area=8, max_area=5000):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
        centers.append((cx, cy))
    if len(centers) == 0:
        return np.empty((0,2))
    return np.array(centers)

def _cluster_rows_cols(pts, eps_factor=0.5):
    if pts.shape[0] == 0:
        return np.empty((0,2))
    ys = np.sort(pts[:,1])
    diffs = np.diff(ys)
    if diffs.size == 0:
        row_spacing = 1.0
    else:
        row_spacing = np.median(diffs[diffs>0]) if np.any(diffs>0) else np.mean(diffs)
    row_thresh = row_spacing * (0.8 + eps_factor*0.2)
    rows = []
    sorted_idx = np.argsort(pts[:,1])
    current = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if abs(pts[idx,1] - pts[current[-1],1]) <= row_thresh:
            current.append(idx)
        else:
            rows.append(current)
            current = [idx]
    rows.append(current)
    grid = []
    for r in rows:
        row_pts = pts[r]
        order = np.argsort(row_pts[:,0])
        row_sorted = row_pts[order]
        grid.append(row_sorted)
    return np.array(grid, dtype=object)

def _make_regular_grid(grid_rows):
    nrows = len(grid_rows)
    ncols = max(len(r) for r in grid_rows) if nrows>0 else 0
    arr = np.full((nrows, ncols, 2), np.nan, dtype=float)
    for i,r in enumerate(grid_rows):
        for j,p in enumerate(r):
            arr[i,j,:] = p
    for j in range(ncols):
        col = arr[:,j,0]
        mask = ~np.isnan(col)
        if np.sum(mask) > 0:
            median_x = np.median(col[mask])
            for i in range(nrows):
                if np.isnan(arr[i,j,0]):
                    arr[i,j,0] = median_x
    for i in range(nrows):
        rowy = arr[i,:,1]
        mask = ~np.isnan(rowy)
        if np.sum(mask)>0:
            median_y = np.median(rowy[mask])
            for j in range(ncols):
                if np.isnan(arr[i,j,1]):
                    arr[i,j,1] = median_y
    return arr

def _compute_intersection_centers(dot_grid):
    r,c,_ = dot_grid.shape
    if r<2 or c<2:
        return np.empty((0,2))
    centers = []
    for i in range(r-1):
        for j in range(c-1):
            p1 = dot_grid[i,j]
            p2 = dot_grid[i,j+1]
            p3 = dot_grid[i+1,j]
            p4 = dot_grid[i+1,j+1]
            center = np.mean([p1,p2,p3,p4], axis=0)
            centers.append(center)
    return np.array(centers).reshape(r-1, c-1, 2)

def _skeletonize_bin(bin_img):
    bin_small = (bin_img > 0).astype(np.uint8)
    sk = skeletonize(bin_small==1)
    return sk.astype(np.uint8)

def _classify_intersections(intersections, skeleton, spacing):
    rows, cols, _ = intersections.shape
    binmat = np.zeros((rows, cols), dtype=int)
    h, w = skeleton.shape
    radius = max(1, int(round(spacing * 0.22)))
    for i in range(rows):
        for j in range(cols):
            cx, cy = intersections[i,j]
            px = int(round(cx))
            py = int(round(cy))
            y0 = max(0, py-radius)
            y1 = min(h, py+radius+1)
            x0 = max(0, px-radius)
            x1 = min(w, px+radius+1)
            patch = skeleton[y0:y1, x0:x1]
            if patch.size==0:
                val = 0
            else:
                val = 1 if patch.sum() > 0 else 0
            binmat[i,j] = val
    return binmat

def binary_matrix_to_hex(bin_matrix):
    flat = ''.join(map(str, bin_matrix.flatten()))
    pad = (4 - (len(flat) % 4)) % 4
    flat = flat + ('0'*pad)
    hex_str = ''
    for i in range(0, len(flat), 4):
        nibble = flat[i:i+4]
        val = int(nibble, 2)
        hex_str += format(val, 'X')
    return hex_str

def extract_hex_from_image(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = _binarize_image(gray)
    dots = _find_dot_centroids(th, min_area=8, max_area=5000)
    if dots.shape[0] < 4:
        raise RuntimeError("Too few dots detected. Try a cleaner image.")
    grid_rows = _cluster_rows_cols(dots, eps_factor=0.5)
    dot_grid = _make_regular_grid(grid_rows)
    first_row = None
    for r in dot_grid:
        if not np.isnan(r[:,0]).all():
            first_row = r
            break
    if first_row is None:
        raise RuntimeError("Couldn't form dot grid")
    xs = first_row[:,0]
    diffs_x = np.diff(xs)
    spacing = float(np.median(diffs_x[diffs_x>0])) if np.any(diffs_x>0) else 20.0
    intersections = _compute_intersection_centers(dot_grid)
    sk = _skeletonize_bin(th)
    binmat = _classify_intersections(intersections, sk, spacing)
    hexcode = binary_matrix_to_hex(binmat)
    if debug:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,3, figsize=(14,5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title('original'); ax[0].axis('off')
        ax[1].imshow(th, cmap='gray'); ax[1].set_title('binarized'); ax[1].axis('off')
        ax[2].imshow(sk, cmap='gray'); ax[2].scatter(dots[:,0], dots[:,1], c='r', s=6)
        r,c,_ = intersections.shape
        for i in range(r):
            for j in range(c):
                cx,cy = intersections[i,j]
                ax[2].plot(cx, cy, 'go', markersize=4)
                ax[2].text(cx+2, cy+2, str(binmat[i,j]), color='yellow', fontsize=8)
        ax[2].set_title('skeleton + dots + intersections'); ax[2].axis('off')
        plt.show()
    return {
        'hex': hexcode,
        'binary_matrix': binmat,
        'dot_grid': dot_grid,
        'intersections': intersections
    }
