import sys
import cv2
import numpy as np
from dataclasses import dataclass
import click
import logging


width = 284
height = 160
frame_size = width * height * 3

@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option('--fps', help='fps', type=float)
@click.option('--skip', help='skip seconds', default=0., type=float)
@click.option('--dump', help='dump images', default=None)
def run(fps: float, skip: float, dump: str):
    a = Analyzer(fps, skip, dump)
    a.run()


@cli.command()
@click.argument('filename')
def oneshot(filename):
    a = Analyzer(1., 0)
    f = Frame.from_file(filename)
    a.oneshot(f).output()


@cli.command()
@click.argument('filename')
@click.argument('output')
def extract_high_rank(filename: str, output: str):
    f = Frame.from_file(filename)
    f.rank_high_img().gray().save(output)


@cli.command()
@click.argument('filename')
@click.argument('output')
def extract_low_rank(filename: str, output: str):
    f = Frame.from_file(filename)
    f.rank_low_img().gray().save(output)


@cli.command()
@click.argument('filename')
@click.argument('output')
def extract_division(filename: str, output: str):
    f = Frame.from_file(filename)
    f.division_img().gray().save(output)


@dataclass
class Frame:
    f: np.array

    @staticmethod
    def from_stdin() -> 'Frame':
        frame_bytes = sys.stdin.buffer.read(frame_size)
        if len(frame_bytes) == frame_size:
            return Frame(
                np
                .frombuffer(frame_bytes, np.uint8)
                .reshape([height, width, 3])
            )

    @staticmethod
    def from_file(filename: str) -> 'Frame':
        return Frame(cv2.imread(filename, cv2.IMREAD_COLOR)).bgr()

    def result_img(self) -> 'Frame':
        return Frame(self.f[21:30,10:39,:])

    def division_img(self) -> 'Frame':
        return Frame(self.f[48:82,123:162,:])

    def rank_high_img(self) -> 'Frame':
        return Frame(self.f[92:99,140:145,:])

    def rank_low_img(self) -> 'Frame':
        return Frame(self.f[88:95,140:145,:])

    def progress_img(self) -> 'Frame':
        return Frame(self.f[128:129,70:215,:])

    def gray(self) -> 'Frame':
        return Frame(cv2.cvtColor(self.f, cv2.COLOR_BGR2GRAY))

    def binary(self) -> 'Frame':
        return Frame(b)

    def inverse(self) -> 'Frame':
        return Frame(255-self.f)

    def progress_victory(self) -> int:
        _, b = cv2.threshold(self.f, 150, 255, cv2.THRESH_BINARY)
        return percent(np.max(cv2.findNonZero(b)[:,:,0])/self.width())

    def progress_defeat(self) -> int:
        _, b = cv2.threshold(self.f, 160, 255, cv2.THRESH_BINARY_INV)
        return percent(np.min(cv2.findNonZero(b)[:,:,0])/self.width())

    def width(self) -> int:
        return len(self.f[0])

    def progress(self, a: 'Analyzer') -> int:
        white_mask = cv2.inRange(self.f, a.lower_white, a.upper_white)
        green_mask = cv2.inRange(self.f, a.lower_green, a.upper_green)
        combined_mask = cv2.bitwise_or(white_mask, green_mask)
        right_boundary = np.max(cv2.findNonZero(combined_mask)) or 0
        return percent(right_boundary/self.width())

    def bgr(self) -> 'Frame':
        return Frame(cv2.cvtColor(self.f, cv2.COLOR_RGB2BGR))

    def save(self, filename: str):
        cv2.imwrite(filename, self.bgr().f)

    def matches(self, template) -> bool:
        m = cv2.matchTemplate(self.f, template, cv2.TM_CCOEFF_NORMED)
        s = cv2.minMaxLoc(m)[1]
        return s > 0.8

    def find_match(self, templates) -> int | None:
        for i, template in enumerate(templates):
            if template is None:
                continue
            if self.matches(template):
                return i

    def find_result(self, templates) -> str:
        for name, template in templates.items():
            if self.match_result(template):
                return name

    def match_result(self, template) -> bool:
        threshold = 0.8
        m = cv2.matchTemplate(self.f, template, cv2.TM_CCOEFF_NORMED)
        sim = cv2.minMaxLoc(m)[1]
        return sim > threshold


class Video:
    def __init__(self, filename: str):
        self.capture = cv2.VideoCapture(filename)

    def get_frame(num: int) -> Frame:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, num)
        _, buf = self.capture.read()
        return Frame(np.frombuffer(buf, np.uint8).reshape([160, 284, 3])[:, :, ::-1])


class Analyzer:
    def __init__(self, fps: float, skip_secs: float = 0., dump: str = None):
        self.fps = fps
        self.dump = dump
        self.results = load_results()
        self.ranks = load_ranks()
        self.divisions = load_divisions()
        self.skip = int(fps * skip_secs)
        # Define the color range for white
        self.lower_white = np.array([200, 200, 200], dtype=np.uint8)
        self.upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # Define the color range for green
        self.lower_green = np.array([0, 128, 0], dtype=np.uint8)
        self.upper_green = np.array([128, 255, 128], dtype=np.uint8)
        logging.info("skip every %d frames", self.skip)

    def run(self):
        i = -1
        a = None
        while True:
            i += 1
            f = Frame.from_stdin()
            if a is not None and (a.frame + 10 * self.fps < i or f is None):
                a.output()
                if self.dump:
                    a.f.save(f'dumps/{self.dump}-{a.timestr()}.bmp')
                a = None
            if f is None:
                break
            if self.skip != 0 and i % self.skip != 1 and a is None:
                continue

            if result := self.get_result(f):
                prev_result = None
                prev_f = None
                while True: 
                    prev_result = result
                    prev_f = f
                    f = Frame.from_stdin()
                    i += 1
                    result = self.get_result(f)
                    if result is None:
                        a = self.analyze(prev_result, prev_f, i-1)
                        break

    def get_result(self, f: Frame) -> str:
        return f.result_img().gray().find_result(self.results)

    def analyze(self, result: str, f: Frame, frame: int) -> 'Analysis':
        division = f.division_img().gray().find_match(self.divisions)
        if division is not None and division <= 3: # <= plat
            rank_img = f.rank_low_img()
        else:
            rank_img = f.rank_high_img()
        rank = rank_img.gray().find_match(self.ranks)
        if rank is not None:
            rank += 1
        progress = f.progress_img().progress(self)
        return Analysis(self.fps, result, division, rank, progress, frame, f)

    def oneshot(self, f: Frame) -> 'Analysis':
        result = f.result_img().gray().find_result(self.results)
        return self.analyze(result, f, 0)


@dataclass
class Analysis:
    fps: float
    result: str
    division: int
    rank: int
    progress: int
    frame: int
    f: Frame

    def output(self):
        print(f'{self.timestr()}\t{self.result}\t{self.divname()}\t{self.rank}\t{self.progress}%', flush=True)

    def divname(self) -> str:
        return '?' if self.division is None else div_names()[self.division]

    def timestr(self) -> str:
        secs = int(self.frame / self.fps % 60)
        mins = int(self.frame / self.fps // 60 % 60)
        hours = int(self.frame / self.fps // 3600)
        millis = int(self.frame % self.fps * 1_000 / self.fps)
        return f'{hours}:{mins:02d}:{secs:02d}.{millis:03d}'


def save_rank(n_frame: int, rank: int):
    f = get_frame(n_frame)
    rank_img = f.get_rank_img().gray().save(f'ranks/{rank}.bmp')

def sec_to_frame(sec: float) -> int:
    return int(sec * 30)

def div_names():
    return ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'master', 'grandmaster', 'champion']

def load_divisions() -> list[np.array]:
    return [cv2.imread(f'divisions/{div}.bmp', 0) for div in div_names()]

def load_ranks() -> list[np.array]:
    return [cv2.imread(f'ranks/{rank}.bmp', 0) for rank in range(1,6)]

def load_results() -> dict[str, np.array]:
    # TODO: add draw
    return {name: cv2.imread(f'results/{name}.bmp', 0) for name in ['victory', 'defeat']}

def percent(n: float) -> int:
    return int(n*100)

if __name__ == '__main__':
    cli()
