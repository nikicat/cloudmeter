import sys
import cv2
import numpy as np

def main(fps: int):
    template_names = ['victory', 'defeat']
    templates = [cv2.imread(f'gr_{name}.bmp', 0) for name in template_names]
    width = 284
    height = 160
    frame_size = width * height * 3
    threshold = 0.8
    f = 0
    prev_f = -100
    while True:
        frame_bytes = sys.stdin.buffer.read(frame_size)
        if len(frame_bytes) != frame_size:
            break
        frame = (
            np
            .frombuffer(frame_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        for name, template in zip(template_names, templates):
            gr_frame = cv2.cvtColor(frame[21:30,10:39,:], cv2.COLOR_BGR2GRAY)
            m = cv2.matchTemplate(gr_frame, template, cv2.TM_CCOEFF_NORMED)
            sim = cv2.minMaxLoc(m)[1]
            if sim > threshold:
                if f - prev_f > 10:
                    output(name, f, frame, fps)
                prev_f = f
                break
        f += 1


def output(name: str, f: int, frame: np.array, fps: float):
    secs = int(f / fps % 60)
    mins = int(f / fps // 60 % 60)
    hours = int(f / fps // 3600)
    print(f'{hours}:{mins:02d}:{secs:02d} {name}', flush=True)


if __name__ == '__main__':
    main(float(sys.argv[1]))
