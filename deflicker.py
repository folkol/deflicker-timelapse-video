"""Normalize per-frame exposure of time-lapse video

"""
import cv2
import numpy as np

MARKER_COLOR = [255, 0, 0]
SAMPLE_SIZE = 300


def adjust_gamma(image, gamma=1.0):
    gamma_inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** gamma_inv) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def estimate_gamma(frame_brightness, target_brightness, epsilon=0.00000001):
    gamma_a, gamma_b = 1 / 1000, 1000
    while True:
        gamma = (gamma_a + gamma_b) / 2
        invGamma = 1.0 / gamma
        y = ((frame_brightness / 255.0) ** invGamma) * 255
        if abs(target_brightness - y) < epsilon:
            return gamma
        if y < target_brightness:
            gamma_a, gamma_b = (gamma_a + gamma_b) / 2, gamma_b
        else:
            gamma_a, gamma_b = gamma_a, (gamma_a + gamma_b) / 2
        # print(gamma_a, gamma_b, abs(target_brightness - y), epsilon)


indata = cv2.VideoCapture('norrsken.mp4')
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'avc1'), 20.0, (1920, 1080))

target_brightness = None
while indata.isOpened():
    ret, frame = indata.read()
    if not ret:
        break
    target_area = frame[-SAMPLE_SIZE:, -SAMPLE_SIZE:]
    sample_area_brightness = target_area.sum() / target_area.size
    if target_brightness is None:
        target_brightness = sample_area_brightness
    gamma = estimate_gamma(sample_area_brightness, target_brightness)
    adjusted_frame = adjust_gamma(frame, gamma)
    adjusted_frame[-SAMPLE_SIZE::2, -SAMPLE_SIZE::2] = [255, 0, 0]
    output.write(adjusted_frame)

indata.release()
output.release()
cv2.destroyAllWindows()
