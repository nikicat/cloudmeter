import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import cv2
    import ffmpeg
    import numpy as np
    import marimo as mo
    from ipywidgets import interact
    from matplotlib import pyplot as plt
    from detect import Analyzer, Frame
    return Analyzer, Frame, cv2, ffmpeg, interact, mo, np, plt


@app.cell
def __(Frame):
    f = Frame.from_file('dumps/1:54:06.966.bmp') # zero-defeat
    f = Frame.from_file('dumps/1:25:07.700.bmp') # defeat
    f = Frame.from_file('dumps/1:03:54.533.bmp') # victory
    f = Frame.from_file('dumps/defeat-t500.bmp') # defeat-t500
    return (f,)


@app.cell
def __(Frame):
    gm4d = Frame.from_file('./samples/gm4-defeat-2.bmp')
    return (gm4d,)


@app.cell
def __(np):
    # Define the color range for white
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Define the color range for green
    lower_green = np.array([0, 128, 0], dtype=np.uint8)
    upper_green = np.array([128, 255, 128], dtype=np.uint8)
    return lower_green, lower_white, upper_green, upper_white


@app.cell
def __(f, mo):
    mo.image(f.f)
    return


@app.cell
def __(cv2, f, lower_green, lower_white, upper_green, upper_white):
    # Create masks for white and green colors
    white_mask = cv2.inRange(f.progress_img().f, lower_white, upper_white)
    green_mask = cv2.inRange(f.progress_img().f, lower_green, upper_green)
    combined_mask = cv2.bitwise_or(white_mask, green_mask)
    return combined_mask, green_mask, white_mask


@app.cell
def __(combined_mask, mo):
    mo.image(combined_mask, height=10)
    return


@app.cell
def __(combined_mask, cv2, np):
    np.max(cv2.findNonZero(combined_mask))
    return


@app.cell
def __(f, mo):
    mo.image(f.progress_img().f, height=10)
    return


@app.cell
def __(cv2, f, mo):
    mo.image(cv2.threshold(f.progress_img().gray().f, 160, 255, cv2.THRESH_BINARY_INV)[1], height=10)
    return


@app.cell
def __(cv2, f, mo):
    mo.image(cv2.threshold(f.progress_img().gray().f, 150, 255, cv2.THRESH_BINARY)[1])
    return


@app.cell
def __():
    def division_img(f):
        return f.f[48:82,123:162,:]
    return (division_img,)


@app.cell
def __(Frame, division_img, mo):
    mo.image(division_img(Frame.from_file('samples/plat4-defeat.bmp')), width=50)
    return


@app.cell
def __(Frame, division_img, mo):
    mo.image(division_img(Frame.from_file('samples/gm4-defeat.bmp')), width=50)
    return


@app.cell
def __(Frame, mo):
    mo.image(Frame.from_file('samples/plat4-defeat.bmp').f[88:95,140:145,:], width=50)
    return


@app.cell
def __(Frame, mo):
    mo.image(Frame.from_file('samples/gm4-defeat.bmp').rank_img().f, width=50)
    return


@app.cell
def __(f):
    f.gray().progress_defeat()
    return


app._unparsable_cell(
    r"""
    capture = cv(2.VideoCapture('in.mp4')
    """,
    name="__"
)


@app.cell
def __(capture, cv2, np):
    def get_frame(num: int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, num)
        _, buf = capture.read()
        return np.frombuffer(buf, np.uint8).reshape([160, 284, 3])[:, :, ::-1]

    def sec_to_frame(sec: float):
        return int(sec * 30)

    def get_rank_img(f):
        return f[92:99,140:145,:]

    def get_progress_img(f):
        return f[128:129,70:215,:]

    def gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def binary(img):
        _, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return b

    def rank_progres(binary_img):
        return int(np.min(cv2.findNonZero(255-binary_img)[:,:,0])/len(binary_img[0])*100)

    def save_rank(n_frame, rank):
        f = get_frame(n_frame)
        rank_img = gray(get_rank_img(f))
        cv2.imwrite(f'ranks/{rank}.bmp', rank_img)

    def load_ranks():
        return [cv2.imread(f'ranks/{rank}.bmp', 0) for rank in range(1,6)]

    def matches(img, template) -> bool:
        m = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        s = cv2.minMaxLoc(m)[1]
        return s > 0.8
    return (
        binary,
        get_frame,
        get_progress_img,
        get_rank_img,
        gray,
        load_ranks,
        matches,
        rank_progres,
        save_rank,
        sec_to_frame,
    )


@app.cell
def __(load_ranks):
    ranks = load_ranks()
    return (ranks,)


@app.cell
def __(
    binary,
    get_frame,
    get_progress_img,
    get_rank_img,
    gray,
    matches,
    mo,
    rank_progres,
    ranks,
):
    def find_rank(img) -> int | None:
        for i, rank_img in enumerate(ranks):
            if rank_img is None:
                continue
            if matches(img, rank_img):
                return i

    def analyze(num: int):
        f = get_frame(num)
        rank_img = gray(get_rank_img(f))
        pr = rank_progres(binary(gray(get_progress_img(f))))
        rank = find_rank(rank_img)
        return mo.vstack([
            mo.image(f, width=700),
            mo.image(rank_img, width=70),
            pr,
            rank,
        ])
    return analyze, find_rank


@app.cell
def __(get_frame, sec_to_frame):
    f_vict = get_frame(sec_to_frame(3600+180+52))
    #mo.image(f_vict, width=700)
    return (f_vict,)


@app.cell
def __(save_rank, sec_to_frame):
    save_rank(sec_to_frame(3600+180+52), 2)
    return


@app.cell
def __(get_frame, sec_to_frame):
    f_defeat = get_frame(sec_to_frame(3600+25*60+7))
    #mo.image(f_defeat, width=700)
    return (f_defeat,)


@app.cell
def __(find_rank, get_frame, get_rank_img, gray, sec_to_frame):
    #f = get_frame(sec_to_frame(3600+25*60+7))
    #rank_img = gray(get_rank_img(f))
    find_rank(gray(get_rank_img(get_frame(sec_to_frame(3600+35*60+7)))))
    return


@app.cell
def __(analyze, sec_to_frame):
    analyze(sec_to_frame(3600+25*60+7))
    return


@app.cell
def __(analyze, sec_to_frame):
    analyze(sec_to_frame(3600+46*60+50))
    return


@app.cell
def __(f_vict, mo):
    mo.image(f_vict[92:99,140:145,:], width=70)
    return


@app.cell
def __(f_vict, get_progress_img, mo):
    mo.image(get_progress_img(f_vict), height=40)
    return


@app.cell
def __(rank_progres):
    rank_progres()
    return


@app.cell
def __(f_defeat, get_progress_img, gray, mo):
    mo.image(gray(get_progress_img(f_defeat)), width=700)
    return


@app.cell
def __(binary, cv2, f_defeat, get_progress_img, gray, np):
    defeat_pr_gray = gray(get_progress_img(f_defeat))
    defeat_pr_binary = binary(defeat_pr_gray)
    np.min(cv2.findNonZero(255-defeat_pr_binary)[:,:,0])/len(defeat_pr_binary[0])
    return defeat_pr_binary, defeat_pr_gray


@app.cell
def __(b, cv2, gr_defeat, gr_victory, num_frames):
    vfs = []
    dfs = []
    for frame in range(num_frames):
        for arr, template in [(vfs, gr_victory), (dfs, gr_defeat)]:
            m = cv2.matchTemplate(cv2.cvtColor(b[frame,21:30,10:39,:], cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
            s = cv2.minMaxLoc(m)[1]
            if s > 0.8 and (len(arr) == 0 or frame - arr[-1] > 10):
                arr.append(frame)
    return arr, dfs, frame, m, s, template, vfs


@app.cell
def __(dfs, vfs):
    dfs, vfs
    return


@app.cell
def __(dfs, f):
    for _f in dfs:
        _secs = f * 2 % 60
        _mins = f * 2 // 60 % 60
        _hours = f * 2 // 3600
        print(f'{_hours}:{_mins:02d}:{_secs:02d}')
    return


@app.cell
def __(f, vfs):
    for _f in vfs:
        _secs = f * 2 % 60
        _mins = f * 2 // 60 % 60
        _hours = f * 2 // 3600
        print(f'{_hours}:{_mins:02d}:{_secs:02d}')
    return


@app.cell
def __(cv2):
    cv2.imread('dumps/1:03:54.533.bmp', 1)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
