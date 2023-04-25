import os
from collections import Counter
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor
from torch import nn
from tqdm.auto import tqdm

ENCODING_AC = {
    '0': [3, 2, 1, 1],
    '1': [2, 2, 2, 1],
    '2': [2, 1, 2, 2],
    '3': [1, 4, 1, 1],
    '4': [1, 1, 3, 2],
    '5': [1, 2, 3, 1],
    '6': [1, 1, 1, 4],
    '7': [1, 3, 1, 2],
    '8': [1, 2, 1, 3],
    '9': [3, 1, 1, 2]
}
ENCODING_B = {k: v[::-1] for k, v in ENCODING_AC.items()}
FIRST_SYM_ENCODING = {
    '00000': '0',
    '01011': '1',
    '01101': '2',
    '01110': '3',
    '10011': '4',
    '11001': '5',
    '11100': '6',
    '10101': '7',
    '10110': '8',
    '11010': '9'
}
FIRST_SYM_DECODING = {v: k for k, v in FIRST_SYM_ENCODING.items()}
START_END_PATTERN = [1, 1, 1]
MID_PATTERN = [1, 1, 1, 1, 1]


def bits_repr(code: List[int], start_sym: int = 0) -> str:
    s = ''
    cur_sym = start_sym
    for i in range(len(code)):
        s += str(cur_sym) * code[i]
        cur_sym = 1 - cur_sym
    return s


def decode_barcode(code: str) -> str:
    result = '101'
    first_sym_code = FIRST_SYM_DECODING[code[0]]
    result += first_sym_code
    result += bits_repr(ENCODING_AC[code[1]])
    for i in range(2, 7):
        result += bits_repr(ENCODING_AC[code[i]] if first_sym_code[i - 2] == '0' else ENCODING_B[code[i]])
    result += '01010'
    for i in range(7, 13):
        result += bits_repr(ENCODING_AC[code[i]], start_sym=1)
    result += '101'
    return result


def get_diff_metric(counters: List[int], pattern: List[int]) -> float:
    assert len(counters) == len(pattern)
    unit_len = counters[0] / pattern[0]
    return (np.array(counters) / (np.array(pattern) * unit_len)).var()


def find_start_pattern(counters: List[int], start_pos: int) -> int:
    cur_pos = start_pos
    while cur_pos < len(counters) - 2:
        if get_diff_metric(counters[cur_pos:cur_pos + 3], START_END_PATTERN) < 0.5:
            break
        cur_pos += 1
    return cur_pos


def find_closes_symbol(counters: List[int], encoding: Dict[str, List[int]]):
    symbol_diff = {}
    for symbol, pattern in encoding.items():
        symbol_diff[symbol] = get_diff_metric(counters, pattern)
    return min(symbol_diff.items(), key=lambda x: x[1])


def check_sum(result: str) -> bool:
    sum = 0
    result = list(map(int, result))
    for i in range(12):
        sum += result[i] + (result[i] * 2 if i % 2 != 0 else 0)
    return result[-1] == (10 - (sum % 10)) % 10


def decode_row(row: np.ndarray) -> str:
    cur_counter = 1
    counters = []
    first_color = prev_color = row[0]
    for cur_color in row[1:]:
        if cur_color == prev_color:
            cur_counter += 1
        else:
            if cur_counter < 3:
                if len(counters) == 0:
                    cur_counter += 1
                else:
                    cur_counter += counters.pop() + 1
            else:
                counters.append(cur_counter)
                cur_counter = 1
        prev_color = cur_color

    if cur_counter < 3:
        counters[-1] += cur_counter
    else:
        counters.append(cur_counter)

    cur_pos = 0 if first_color == 0 else 1
    cur_pos = find_start_pattern(counters, cur_pos) + 3
    if cur_pos + 4 * 12 + 8 > len(counters):
        return ''

    result = find_closes_symbol(counters[cur_pos: cur_pos + 4], ENCODING_AC)[0]
    cur_pos += 4
    first_sym_code = ''
    for i in range(2, 7):
        AC_result = find_closes_symbol(counters[cur_pos: cur_pos + 4], ENCODING_AC)
        B_result = find_closes_symbol(counters[cur_pos: cur_pos + 4], ENCODING_B)
        if AC_result[1] < B_result[1]:
            result += AC_result[0]
            first_sym_code += '0'
        else:
            result += B_result[0]
            first_sym_code += '1'
        cur_pos += 4

    if first_sym_code not in FIRST_SYM_ENCODING or get_diff_metric(counters[cur_pos:cur_pos + 5], MID_PATTERN) >= 1.0:
        return ''
    cur_pos += 5

    result = FIRST_SYM_ENCODING[first_sym_code] + result

    for i in range(7, 13):
        result += find_closes_symbol(counters[cur_pos:cur_pos + 4], ENCODING_AC)[0]
        cur_pos += 4

    if not check_sum(result) or get_diff_metric(counters[cur_pos:cur_pos + 3], START_END_PATTERN) >= 0.5:
        return ''

    return result


def decode_image(image: np.ndarray) -> str:
    results = Counter()
    for row in image:
        row_result = decode_row(row)
        if len(row_result) == 13:
            results.update([row_result])
    for row in image[:, ::-1]:
        row_result = decode_row(row)
        if len(row_result) == 13:
            results.update([row_result])

    return '1' * 13 if len(results) == 0 else results.most_common(1)[0][0]


def init_segmentor(config, checkpoint=None, device='cuda') -> nn.Module:
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def infer(config_path: str, images_dir: str, pred_path: str = 'predictions.csv', device: str = 'cuda') -> None:
    config = Config.fromfile(config_path)
    model = init_segmentor(config, fr'workdir\hrnet_v1\latest.pth', device=device)

    image_names = os.listdir(images_dir)

    data = []
    for image_name in tqdm(image_names):
        image_result = inference_segmentor(model, os.path.join(images_dir, image_name))[0].astype(np.uint8)
        contours = cv2.findContours(image_result, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
        max_contour = max(contours, key=lambda c: cv2.contourArea(c))
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect).reshape((4, 2))

        out_height = 64
        out_width = 728
        if ((box[1] - box[0]) ** 2).sum() > ((box[2] - box[1]) ** 2).sum():
            out_image_pts = [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]]
        else:
            out_image_pts = [[0, out_height], [0, 0], [out_width, 0], [out_width, out_height]]
        out_image_pts = np.float32(out_image_pts)

        transform = cv2.getPerspectiveTransform(box, out_image_pts)
        image = cv2.imread(os.path.join(images_dir, image_name), cv2.IMREAD_COLOR)
        image = cv2.warpPerspective(image, transform, (out_width, out_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, 3)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
        image = cv2.medianBlur(image, 3)
        # cv2.imwrite(os.path.join('target', image_name), image[:, ::-1])
        code = decode_image(image)
        data.append([code] + box.astype(int).ravel().tolist() + [decode_barcode(code)])

    pd.DataFrame(data=data, index=image_names).to_csv(pred_path, header=False, encoding='utf-16')
