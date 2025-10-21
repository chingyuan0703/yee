# -*- coding: utf-8 -*-
"""
Lane detection (integrated):
- 轉灰階 → GaussianBlur → (可選 close) → Dynamic Canny
- ROI (梯形) 遮罩（快取）
- HoughLinesP 找線 → 依角度分左右 → 收集端點做最小平方法擬合
- 影格間 EMA 平滑 (s, b)
- 疊圖：半透明綠區 + 中心偏移 + 右側顯示 Canny（可選只顯示 ROI）
- 參數比例化（隨影像高 HH 調整 Hough）
"""

import cv2
import numpy as np

# ===== 全域參數 =====
WW, HH = 640, 400        # 縮放後寬高
RH, R  = 0.60, 3         # ROI 高度比例、底部邊界內縮像素
VIDEO_PATH = 'LaneVideo.mp4'  # ← 改成你的影片路徑

# Hough 參數（比例化）
HOUGH_THRESHOLD = 40                      # 票數門檻
HOUGH_MIN_LINE_LENGTH_RATIO = 0.08        # 以 HH 為基礎
HOUGH_MAX_LINE_GAP_RATIO   = 0.15

# 額外選項
EDGES_USE_ROI = False     # True: 右側只顯示 ROI 內的 Canny
SHOW_OVERLAY  = True      # True: 左側疊合綠區/中心線

# 角度與平滑
ANGLE_LEFT  = (-85, -25)  # 左車道線角度範圍（度）
ANGLE_RIGHT = (25, 85)    # 右車道線角度範圍（度）
EMA_ALPHA   = 0.7         # 越大越穩但反應慢 (0~1)

# ===== 狀態快取 =====
_prev_left  = None   # (s, b)
_prev_right = None   # (s, b)
_roi_cache  = None   # ROI mask


# ---------- 工具函式 ----------
def dynamic_canny(img_gray_blurred: np.ndarray) -> np.ndarray:
    """依亮度自動設定 Canny 門檻。"""
    v = float(np.median(img_gray_blurred))
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(img_gray_blurred, lo, hi)


def build_roi_mask(width: int, height: int, rh: float, margin: int) -> np.ndarray:
    """梯形 ROI；同尺寸只建一次並快取。"""
    global _roi_cache
    if _roi_cache is not None:
        return _roi_cache

    xx1, yy1 = int(width * 0.40), int(height * rh)
    xx2, yy2 = int(width * 0.60), int(height * rh)
    p1 = (margin, height - margin)
    p2 = (width - margin, height - margin)
    p3 = (xx2, yy2)
    p4 = (xx1, yy2)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([p1, p2, p3, p4], np.int32)], 255)
    _roi_cache = mask
    return mask


def split_and_fit_lines(lines: np.ndarray, width: int, height: int):
    """
    將 Hough 直線以角度分成左右，彙整所有端點做最小平方法擬合 y = s*x + b
    回傳: (done, (sL, bL), (sR, bR))；done: bit1=左, bit2=右
    """
    left_pts, right_pts = [], []
    if lines is None:
        return 0, (0.0, 0.0), (0.0, 0.0)

    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        dx, dy = (x2 - x1), (y2 - y1)
        ang = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)

        # 過濾太靠邊或太短的線
        if min(x1, x2) < 20 or max(x1, x2) > width - 20 or length < 10:
            continue

        if ANGLE_LEFT[0] <= ang <= ANGLE_LEFT[1]:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif ANGLE_RIGHT[0] <= ang <= ANGLE_RIGHT[1]:
            right_pts.extend([(x1, y1), (x2, y2)])

    def fit(points):
        if len(points) < 4:
            return None
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        A = np.vstack([xs, np.ones_like(xs)]).T
        s, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        return float(s), float(b)

    L = fit(left_pts)
    R = fit(right_pts)
    done = (1 if L is not None else 0) | (2 if R is not None else 0)
    return done, (L if L else (0.0, 0.0)), (R if R else (0.0, 0.0))


def ema(prev, cur):
    """(s, b) 的指數移動平均。"""
    if prev is None:
        return cur
    p = np.array(prev, dtype=np.float32)
    c = np.array(cur,  dtype=np.float32)
    return (EMA_ALPHA * p + (1 - EMA_ALPHA) * c).tolist()


# ---------- 單幀處理 ----------
def process_frame(frame: np.ndarray) -> np.ndarray:
    global _prev_left, _prev_right

    # 縮放
    img1 = cv2.resize(frame, (WW, HH))

    # 前處理：灰階 → Blur → (可選 close) → 動態 Canny
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    blur  = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=1)  # 視噪聲情況可註解
    edge  = dynamic_canny(blur)

    # ROI 遮罩
    roi_mask = build_roi_mask(WW, HH, RH, R)
    edge_roi = cv2.bitwise_and(edge, roi_mask)

    # Hough 參數（比例化）
    min_len = int(HH * HOUGH_MIN_LINE_LENGTH_RATIO)
    max_gap = int(HH * HOUGH_MAX_LINE_GAP_RATIO)

    lines = cv2.HoughLinesP(edge_roi, 1, np.pi/180,
                            HOUGH_THRESHOLD, None,
                            min_len, max_gap)

    done, (s1, b1), (s2, b2) = split_and_fit_lines(lines, WW, HH)

    # 影格間平滑與回退
    if done & 1: _prev_left  = ema(_prev_left,  (s1, b1))
    if done & 2: _prev_right = ema(_prev_right, (s2, b2))
    if not (done & 1) and _prev_left  is not None:  s1, b1 = _prev_left
    if not (done & 2) and _prev_right is not None:  s2, b2 = _prev_right
    done = (1 if _prev_left  is not None else 0) | (2 if _prev_right is not None else 0)

    # 左側：原圖或疊圖
    left = img1.copy()
    if SHOW_OVERLAY:
        # 底部刻度
        cx, cy = int(WW / 2) - 1, HH - R
        cv2.line(left, (cx, cy), (cx, cy - 12), (255, 0, 0), 2)
        for i in range(1, 10):
            cv2.line(left, (cx - i * 15, cy), (cx - i * 15, cy - 3), (0, 255, 0), 2)
            cv2.line(left, (cx + i * 15, cy), (cx + i * 15, cy - 3), (0, 255, 0), 2)

        if done == 3 and abs(s1) > 1e-3 and abs(s2) > 1e-3:
            y_bottom = HH - R
            y_top    = int(HH - HH * 0.175)

            # 上下水平線與左右線的交點
            xl_b = (y_bottom - b1) / s1
            xr_b = (y_bottom - b2) / s2
            xl_t = (y_top    - b1) / s1
            xr_t = (y_top    - b2) / s2

            poly = np.array([
                (int(xl_b), y_bottom),
                (int(xr_b), y_bottom),
                (int(xr_t), y_top),
                (int(xl_t), y_top)
            ], dtype=np.int32)

            shade = np.zeros_like(left)
            cv2.fillPoly(shade, [poly], (0, 80, 0))
            left = cv2.addWeighted(left, 1.0, shade, 0.35, 0.0)  # 0.3~0.5 較自然

            # 中心紅線與像素偏移
            x_center_b = int((xl_b + xr_b) / 2)
            x_center_t = int((xl_t + xr_t) / 2)
            cv2.line(left, (x_center_b, y_bottom), (x_center_t, y_top), (0, 0, 255), 3)

            offset_px = int(x_center_b - (WW / 2))
            cv2.putText(left, f"offset: {offset_px:+d}px", (12, HH - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 右側：Canny 畫面
    edges_to_show = edge_roi if EDGES_USE_ROI else edge
    right = cv2.cvtColor(edges_to_show, cv2.COLOR_GRAY2BGR)

    # 標題
    cv2.putText(left,  "Original / Overlay", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "Canny Edges" + (" (ROI)" if EDGES_USE_ROI else ""),
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 左右並排
    combined = np.hstack([left, right])
    return combined


# ---------- 主程式 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f'無法開啟影片：{VIDEO_PATH}')

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('讀取影片第一幀失敗')

    out_size = (WW * 2, HH)  # 並排寬度 ×2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter('output_side_by_side.mp4', fourcc, fps, out_size)

    # 第一幀
    out.write(process_frame(frame))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = process_frame(frame)
        out.write(result)

        cv2.imshow('lane (left: overlay, right: canny)', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('完成：output_side_by_side.mp4')

if __name__ == '__main__':
    main()
