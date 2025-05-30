import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import traceback
import os
import random
import time
import io
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.font_manager as fm
import discord
from discord.ext import commands
from discord import app_commands
from collections import deque, Counter
import typing

DATA_FILENAME = "makemeahanzi/graphics.txt"
POEMS_SOURCES = {'poems.json': '唐詩三百首', 'easypoems.json': '常見唐詩(較簡單)'}
DEFAULT_POEMS_SOURCE = 'poems.json'

ALL_CHARACTERS_DATA: typing.Dict[str, typing.List[str]] = {}
ALL_CHARACTERS_DTW_DATA: typing.Dict[str, typing.List[typing.Tuple[np.ndarray,
                                                                   int]]] = {}
VALID_POEM_LINES_MAP: typing.Dict[str, typing.List[str]] = {}
POEM_INFO_MAP_MAP: typing.Dict[str, typing.Dict[str,
                                                typing.Dict[str,
                                                            typing.Any]]] = {}
GAME_LOAD_ERROR: typing.Optional[str] = None

RECENT_LINES_LIMIT = 10

FONT_FILE = 'NotoSansTC-Regular.ttf'
FONT_PATH = os.path.join(os.path.dirname(__file__), 'fonts', FONT_FILE)

if os.path.exists(FONT_PATH):
    try:
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"成功設定中文字體: {font_prop.get_name()}")
    except Exception as e:
        print(f"警告: 設定中文字體時發生錯誤: {e}. 中文字符可能無法正常顯示.")
        traceback.print_exc()
else:
    print(f"警告: 中文字體檔案未找到於 {FONT_PATH}. 中文字符可能無法正常顯示.")

DIFFICULTY_THRESHOLDS = {
    1: [500, 1000],
    2: [1000, 1500],
    3: [2000, 2500],
    4: [4000, 6000],
    5: [8000, 10000],
    6: [12000, 20000]
}


def parse_svg_path(
    path_string: str,
    num_curve_points: int = 30
) -> typing.List[typing.List[typing.List[float]]]:
    points_list: typing.List[typing.List[typing.List[float]]] = []
    current_point: typing.Optional[np.ndarray] = None
    subpath_start_point: typing.Optional[np.ndarray] = None
    tokens = re.findall(
        r'[MLQCSZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?\s*', path_string)
    i = 0
    num_tokens = len(tokens)

    def get_coords(count: int) -> typing.List[float]:
        nonlocal i
        coords = []
        if i + count > num_tokens:
            command_context = tokens[i - 1].strip() if i > 0 else "START"
            remaining = num_tokens - i
            raise ValueError(
                f"座標不足 (需要 {count}, 剩餘 {remaining}) 於指令 '{command_context}' 之後 (索引 {i})."
            )
        for k in range(count):
            coord_str = tokens[i].strip()
            coords.append(float(coord_str))
            i += 1
        return coords

    try:
        while i < num_tokens:
            command = tokens[i].strip()
            if not re.match(r'[MLQCSZz]', command):
                i += 1
                continue
            i += 1
            if command == 'M':
                coords = get_coords(2)
                x, y = coords
                current_point = np.array([x, y])
                subpath_start_point = current_point.copy()
                points_list.append([current_point.tolist()])
                while i + 1 < num_tokens and re.match(
                        r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                        tokens[i].strip()) and re.match(
                            r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                            tokens[i + 1].strip()):
                    coords = get_coords(2)
                    x, y = coords
                    current_point = np.array([x, y])
                    points_list[-1].append(current_point.tolist())
            elif command == 'L':
                if current_point is None:
                    raise ValueError(f"L 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.")
                while True:
                    coords = get_coords(2)
                    x, y = coords
                    current_point = np.array([x, y])
                    points_list[-1].append(current_point.tolist())
                    if i + 1 >= num_tokens or not (re.match(
                            r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                            tokens[i].strip()) and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 1].strip())):
                        break
            elif command == 'Q':
                if current_point is None:
                    raise ValueError(f"Q 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.")
                while True:
                    coords = get_coords(4)
                    p0 = current_point
                    p1 = np.array(coords[:2])
                    p2 = np.array(coords[2:])
                    curve_points = []
                    if num_curve_points >= 1:
                        for t in np.linspace(0, 1, num_curve_points):
                            curve_points.append(
                                ((1 - t)**2 * p0 + 2 * (1 - t) * t * p1 +
                                 t**2 * p2).tolist())
                    if curve_points:
                        if points_list and points_list[-1] and (
                                abs(points_list[-1][-1][0] -
                                    curve_points[0][0]) > 1e-6
                                or abs(points_list[-1][-1][1] -
                                       curve_points[0][1]) > 1e-6):
                            points_list[-1].extend(curve_points)
                        elif points_list:
                            points_list[-1].extend(curve_points[1:])
                    current_point = p2
                    if i + 3 >= num_tokens or not (
                            re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 1].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 2].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 3].strip())):
                        break
            elif command == 'C':
                if current_point is None:
                    raise ValueError(f"C 指令於索引 {i-1} 需要先前的 M, L, Q, 或 C 指令.")
                while True:
                    coords = get_coords(6)
                    p0 = current_point
                    p1 = np.array(coords[:2])
                    p2_ctl = np.array(coords[2:4])
                    p3 = np.array(coords[4:])
                    curve_points = []
                    if num_curve_points >= 1:
                        for t in np.linspace(0, 1, num_curve_points):
                            curve_points.append(
                                ((1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 +
                                 3 * (1 - t) * t**2 * p2_ctl +
                                 t**3 * p3).tolist())
                    if curve_points:
                        if points_list and points_list[-1] and (
                                abs(points_list[-1][-1][0] -
                                    curve_points[0][0]) > 1e-6
                                or abs(points_list[-1][-1][1] -
                                       curve_points[0][1]) > 1e-6):
                            points_list[-1].extend(curve_points)
                        elif points_list:
                            points_list[-1].extend(curve_points[1:])
                    current_point = p3
                    if i + 5 >= num_tokens or not (
                            re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 1].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 2].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 3].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 4].strip())
                            and re.match(
                                r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',
                                tokens[i + 5].strip())):
                        break
            elif command.lower() == 'z':
                if current_point is not None and subpath_start_point is not None and points_list:
                    if points_list[-1] and (
                            abs(current_point[0] - subpath_start_point[0])
                            > 1e-6 or abs(current_point[1] -
                                          subpath_start_point[1]) > 1e-6):
                        points_list[-1].append(subpath_start_point.tolist())
                    current_point = subpath_start_point.copy()
                subpath_start_point = None
            else:
                pass
    except Exception as e:
        print(f"解析 SVG 路徑時發生錯誤: {e}")
        traceback.print_exc()
        return [
            seg_list for seg_list in points_list
            if seg_list and isinstance(seg_list, list)
        ]
    return [
        segment_points_list for segment_points_list in points_list
        if segment_points_list
    ]


def calculate_stroke_centroid(path_string: str) -> np.ndarray:
    try:
        segments = parse_svg_path(path_string, num_curve_points=5)
        all_points = []
        for segment_list in segments:
            if isinstance(segment_list, list):
                all_points.extend(segment_list)
        all_points_np: np.ndarray = np.array(all_points)
        if all_points_np.shape[0] > 0:
            return np.mean(all_points_np, axis=0)
        else:
            first_point = None
            for seg_list in parse_svg_path(path_string, num_curve_points=2):
                if seg_list and len(seg_list) > 0:
                    first_point = np.array(seg_list[0])
                    break
            return first_point if first_point is not None and first_point.shape == (
                2, ) else np.array([512.0, 512.0])
    except Exception as e:
        print(f"計算質心時發生錯誤: {e}")
        return np.array([512.0, 512.0])


def load_character_data(
    filepath: str
) -> typing.Tuple[typing.Dict[str, typing.List[str]], typing.Optional[str]]:
    character_data = {}
    print(f"嘗試載入筆畫資料檔案於: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(
                            data, dict
                    ) and 'character' in data and 'strokes' in data:
                        char = data['character']
                        strokes = data['strokes']
                        if isinstance(char, str) and len(char) == 1 and \
                           isinstance(strokes, list) and all(isinstance(s, str) for s in strokes):
                            character_data[char] = strokes
                        else:
                            print(
                                f"警告: 略過檔案中的無效資料格式 (字元或筆畫格式錯誤) 於行 {line_num}: {line}"
                            )
                    else:
                        print(
                            f"警告: 略過檔案中的無效行 (缺少鍵 'character' 或 'strokes' 或非字典) {line_num}: {line}"
                        )
                except json.JSONDecodeError as e:
                    print(f"警告: 解碼 JSON 錯誤於行 {line_num}: {line} - {e}")
                except Exception as e:
                    print(f"處理行 {line_num} 時發生未知錯誤: {e}")
    except FileNotFoundError:
        error_msg = f"錯誤: 筆畫資料檔案未找到於 {filepath}. 請確認檔案存在."
        print(error_msg)
        return {}, error_msg
    except Exception as e:
        error_msg = f"讀取筆畫資料檔案 '{filepath}' 時發生錯誤: {e}"
        print(error_msg)
        traceback.print_exc()
        return {}, error_msg
    print(f"成功載入 {len(character_data)} 個漢字的筆畫資料.")
    return character_data, None


def load_poems_from_source(
    filepath: str, char_data: typing.Dict[str, typing.List[str]]
) -> typing.Tuple[typing.List[str], typing.Dict[str, typing.Dict[
        str, typing.Any]], typing.Optional[str]]:
    valid_lines = []
    poem_info_map = {}
    print(f"嘗試載入詩詞檔案於: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            poems_data = json.load(f)
        if not isinstance(poems_data, list):
            error_msg = f"詩詞檔案格式錯誤 ('{filepath}'): 根元素不是列表."
            print(error_msg)
            return [], {}, error_msg
        for poem in poems_data:
            if not isinstance(
                    poem, dict
            ) or 'title' not in poem or 'content' not in poem or not isinstance(
                    poem['content'], list):
                print(
                    f"警告: 略過無效格式的詩詞條目 ('{filepath}'): {poem.get('title', '未知標題')}"
                )
                continue
            title = poem.get('title', '未知詩名')
            content = poem['content']
            for line in content:
                if isinstance(line, str) and len(line) == 5:
                    is_valid_line = True
                    for char in line:
                        if char not in char_data:
                            is_valid_line = False
                            break
                    if is_valid_line:
                        valid_lines.append(line)
                        poem_info_map[line] = {
                            'title':
                            title,
                            'content':
                            [str(l) for l in content if isinstance(l, str)]
                        }
                    else:
                        pass
        valid_lines = list(set(valid_lines))
        print(f"成功載入並驗證 {len(valid_lines)} 條有效 (五言且字元存在) 詩句 來自 '{filepath}'.")
        if not valid_lines:
            return [], {}, f"沒有載入到任何來源的有效五言詩句 來自 '{filepath}'."
        return valid_lines, poem_info_map, None
    except FileNotFoundError:
        error_msg = f"錯誤: 詩詞檔案未找到於 {filepath}. 請確認檔案存在."
        print(error_msg)
        return [], {}, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"錯誤: 解碼詩詞 JSON 檔案 '{filepath}' 錯誤: {e}"
        print(error_msg)
        return [], {}, error_msg
    except Exception as e:
        error_msg = f"讀取詩詞檔案 '{filepath}' 時發生錯誤: {e}"
        print(error_msg)
        traceback.print_exc()
        return [], {}, error_msg


def get_stroke_point_sequences_with_original_index(
        char: str,
        char_data: typing.Dict[str, typing.List[str]],
        num_curve_points: int = 7
) -> typing.List[typing.Tuple[np.ndarray, int]]:
    stroke_point_sequences_with_original_index = []
    stroke_paths = char_data.get(char, [])

    if not stroke_paths:
        return []

    for original_stroke_index, path_str in enumerate(stroke_paths):
        try:
            segments_for_this_stroke = parse_svg_path(
                path_str, num_curve_points=num_curve_points)

            all_points_for_stroke_list: typing.List[typing.List[float]] = []
            if segments_for_this_stroke:
                all_points_for_stroke_list = [
                    point for segment in segments_for_this_stroke
                    if isinstance(segment, list) for point in segment
                    if isinstance(point, list) and len(point) == 2
                ]

            all_points_for_stroke: np.ndarray = np.array(
                all_points_for_stroke_list)

            if all_points_for_stroke.shape[0] >= 2:
                center = np.mean(all_points_for_stroke, axis=0)
                processed_sequence: np.ndarray = all_points_for_stroke - center
                stroke_point_sequences_with_original_index.append(
                    (processed_sequence, original_stroke_index))
            else:
                pass

        except Exception as e:
            print(
                f"Warning: 處理字元 '{char}' 原始筆畫 {original_stroke_index+1} 以準備 DTW 時發生錯誤: {e}"
            )
            traceback.print_exc()
            pass

    return stroke_point_sequences_with_original_index


def precompute_dtw_data(
    char_data: typing.Dict[str, typing.List[str]]
) -> typing.Dict[str, typing.List[typing.Tuple[np.ndarray, int]]]:
    dtw_data = {}
    num_curve_points = 7
    for char in char_data:
        try:
            dtw_data[char] = get_stroke_point_sequences_with_original_index(
                char, char_data, num_curve_points)
        except Exception as e:
            print(
                f"Warning: Precomputing DTW data for character '{char}' failed: {e}"
            )
            traceback.print_exc()
            dtw_data[char] = []
    print(
        f"Completed precomputation of DTW data for {len(dtw_data)} characters."
    )
    return dtw_data


def get_path_outline_points(path_string: str,
                            num_curve_points: int = 30) -> np.ndarray:
    segments = parse_svg_path(path_string, num_curve_points=num_curve_points)
    all_points_list: typing.List[typing.List[float]] = []
    if segments:
        all_points_list = [
            point for segment in segments if isinstance(segment, list)
            for point in segment if isinstance(point, list) and len(point) == 2
        ]
    all_points: np.ndarray = np.array(all_points_list)
    return all_points


def plot_solid_black_character(
        target_char: str,
        char_data: typing.Dict[str,
                               typing.List[str]]) -> typing.Optional[bytes]:
    if target_char not in char_data:
        print(f"繪圖錯誤: 目標字元 '{target_char}' 不在筆畫資料中，無法繪製黑色圖片.")
        return None
    target_stroke_paths = char_data.get(target_char, [])
    if not target_stroke_paths:
        print(f"繪圖錯誤: 目標字元 '{target_char}' 沒有筆畫資料，無法繪製黑色圖片.")
        return None

    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    padding = 100
    ax.set_xlim(0 - padding, 1024 + padding)
    ax.set_ylim(0 - padding, 1024 + padding)

    color_pure_black = 0.0

    for path_str in target_stroke_paths:
        try:
            outline_points = get_path_outline_points(path_str,
                                                     num_curve_points=30)
            if isinstance(
                    outline_points, np.ndarray
            ) and outline_points.ndim == 2 and outline_points.shape[0] > 0:
                ax.fill(outline_points[:, 0],
                        outline_points[:, 1],
                        color=str(color_pure_black),
                        zorder=2)
                ax.plot(outline_points[:, 0],
                        outline_points[:, 1],
                        color=str(color_pure_black),
                        linewidth=0.5,
                        zorder=3,
                        solid_capstyle='round',
                        solid_joinstyle='round')
            else:
                print(f"Warning: 字元 '{target_char}' 的筆畫路徑解析失敗或無效點 (用於繪製黑色圖).")

        except Exception as e:
            print(f"Warning: 解析或繪製字元 '{target_char}' 的筆畫時發生錯誤 (用於繪製黑色圖): {e}")
            traceback.print_exc()
            pass

    buf = io.BytesIO()
    try:
        plt.tight_layout(pad=0)
        fig.savefig(buf,
                    format='png',
                    dpi=200,
                    bbox_inches='tight',
                    pad_inches=0)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"錯誤: 儲存字元 '{target_char}' 的黑色圖片時發生錯誤: {e}")
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)


def plot_character_colored_by_history(
        target_char: str, char_history: typing.Dict[str, typing.Any],
        thresholds: typing.Dict[str, float]) -> typing.Optional[bytes]:
    if target_char not in ALL_CHARACTERS_DATA:
        print(f"繪圖錯誤: 目標字元 '{target_char}' 不在筆畫資料中.")
        return None
    target_stroke_paths = ALL_CHARACTERS_DATA.get(target_char, [])
    target_stroke_histories = char_history.get('stroke_histories', [])

    if not target_stroke_paths:
        print(f"繪圖錯誤: 目標字元 '{target_char}' 沒有筆畫資料.")
        return None

    if len(target_stroke_paths) != len(target_stroke_histories):
        print(
            f"繪圖錯誤: 目標字元 '{target_char}' 的筆畫資料 ({len(target_stroke_paths)}) 或歷史記錄 ({len(target_stroke_histories)}) 不一致. 正在嘗試重設歷史記錄."
        )
        char_history['stroke_histories'] = [{
            'min_dist': float('inf'),
            'best_guess_char': None,
            'best_guess_stroke_index': None
        } for _ in range(len(target_stroke_paths))]
        target_stroke_histories = char_history['stroke_histories']

    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    padding = 100
    ax.set_xlim(0 - padding, 1024 + padding)
    ax.set_ylim(0 - padding, 1024 + padding)

    threshold1 = thresholds.get('thresh1', 10000)
    threshold2 = thresholds.get('thresh2', 25000)
    color_pure_black = 0.0
    color_fixed_gray = 0.5

    for target_stroke_index, hist_info in enumerate(target_stroke_histories):
        historical_min_dist = hist_info.get('min_dist', float('inf'))
        historical_guess_char = hist_info.get('best_guess_char')
        historical_guess_stroke_index = hist_info.get(
            'best_guess_stroke_index')

        plot_path_str = None
        color_to_use = None
        historical_guess_centroid = None
        target_stroke_path_str = target_stroke_paths[target_stroke_index]
        target_centroid = calculate_stroke_centroid(target_stroke_path_str)

        if historical_min_dist < threshold1:
            if historical_guess_char is not None and historical_guess_char in ALL_CHARACTERS_DATA:
                historical_guess_stroke_paths = ALL_CHARACTERS_DATA.get(
                    historical_guess_char, [])
                if historical_guess_stroke_paths and historical_guess_stroke_index is not None and 0 <= historical_guess_stroke_index < len(
                        historical_guess_stroke_paths):
                    plot_path_str = historical_guess_stroke_paths[
                        historical_guess_stroke_index]
                    historical_guess_centroid = calculate_stroke_centroid(
                        plot_path_str)
                    color_to_use = str(color_pure_black)
                else:
                    print(
                        f"Warning: 位置 '{target_char}' ({target_stroke_index+1}) 歷史猜測字 '{historical_guess_char}' 或筆畫索引 {historical_guess_stroke_index} 無效，即使距離 < threshold1 ({historical_min_dist})."
                    )

        elif historical_min_dist >= threshold1 and historical_min_dist < threshold2:
            if historical_guess_char is not None and historical_guess_char in ALL_CHARACTERS_DATA:
                historical_guess_stroke_paths = ALL_CHARACTERS_DATA.get(
                    historical_guess_char, [])
                if historical_guess_stroke_paths and historical_guess_stroke_index is not None and 0 <= historical_guess_stroke_index < len(
                        historical_guess_stroke_paths):
                    plot_path_str = historical_guess_stroke_paths[
                        historical_guess_stroke_index]
                    historical_guess_centroid = calculate_stroke_centroid(
                        plot_path_str)
                    color_to_use = str(color_fixed_gray)
                else:
                    print(
                        f"Warning: 位置 '{target_char}' ({target_stroke_index+1}) 歷史猜測字 '{historical_guess_char}' 或筆畫索引 {historical_guess_stroke_index} 無效，即使距離 < threshold2 ({historical_min_dist})."
                    )

        if plot_path_str is not None and color_to_use is not None:
            try:
                translation = np.array([0.0, 0.0])
                if isinstance(target_centroid, np.ndarray) and isinstance(
                        historical_guess_centroid,
                        np.ndarray) and target_centroid.shape == (
                            2, ) and historical_guess_centroid.shape == (2, ):
                    translation = target_centroid - historical_guess_centroid
                else:
                    translation = np.array([0.0, 0.0])

                outline_points = get_path_outline_points(plot_path_str,
                                                         num_curve_points=30)

                if isinstance(
                        outline_points, np.ndarray
                ) and outline_points.ndim == 2 and outline_points.shape[0] > 0:
                    translated_outline = outline_points + translation

                    ax.fill(translated_outline[:, 0],
                            translated_outline[:, 1],
                            color=color_to_use,
                            zorder=2)
                    ax.plot(translated_outline[:, 0],
                            translated_outline[:, 1],
                            color=color_to_use,
                            linewidth=0.5,
                            zorder=3,
                            solid_capstyle='round',
                            solid_joinstyle='round')
                else:
                    print(
                        f"Warning: 字元 '{target_char}' ({target_stroke_index+1}) 的歷史猜測筆畫路徑解析失敗或無效點."
                    )

            except Exception as e:
                print(
                    f"Warning: 解析、處理或繪製目標字 '{target_char}' 筆畫 {target_stroke_index+1} 的歷史猜測筆畫時發生錯誤: {e}"
                )
                traceback.print_exc()
                pass

    buf = io.BytesIO()
    try:
        plt.tight_layout(pad=0)
        fig.savefig(buf,
                    format='png',
                    dpi=200,
                    bbox_inches='tight',
                    pad_inches=0)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"錯誤: 儲存字元 '{target_char}' 的著色圖片時發生錯誤: {e}")
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)


def generate_stats_plot_buffer(
        guess_counts: typing.List[int]) -> typing.Optional[bytes]:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    if not guess_counts:
        ax.text(0.5,
                0.5,
                '尚未完成任何詩詞猜測',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12)
        ax.set_title("猜測次數統計圖")
        ax.axis('off')
    else:
        count_map = Counter(guess_counts)
        sorted_counts = sorted(count_map.items())
        guess_numbers = [item[0] for item in sorted_counts]
        frequencies = [item[1] for item in sorted_counts]
        bars = ax.bar(guess_numbers, frequencies, color='#007bff')
        max_frequency = max(frequencies) if frequencies else 0
        ax.set_xlabel("猜測次數")
        ax.set_ylabel("完成次數")
        ax.set_title("猜測次數統計圖")
        ax.set_xticks(guess_numbers)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2,
                     yval + 0.1,
                     str(int(yval)),
                     ha='center',
                     va='bottom')
        y_upper_limit = max_frequency + 0.1 + 0.5
        ax.set_ylim(0, y_upper_limit)

    buf = io.BytesIO()
    try:
        plt.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"生成統計圖表時發生錯誤: {e}")
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)


ALL_CHARACTERS_DATA, char_load_error = load_character_data(DATA_FILENAME)
if ALL_CHARACTERS_DATA:
    ALL_CHARACTERS_DTW_DATA = precompute_dtw_data(ALL_CHARACTERS_DATA)
    print("DTW 資料預計算完成.")

GLOBAL_POEMS_LOAD_ERRORS = {}
for source_key, source_name in POEMS_SOURCES.items():
    filepath = source_key
    VALID_POEM_LINES, POEM_INFO_MAP, poems_load_error = load_poems_from_source(
        filepath, ALL_CHARACTERS_DATA)
    VALID_POEM_LINES_MAP[source_key] = VALID_POEM_LINES
    POEM_INFO_MAP_MAP[source_key] = POEM_INFO_MAP
    if poems_load_error:
        GLOBAL_POEMS_LOAD_ERRORS[source_key] = poems_load_error

if char_load_error:
    GAME_LOAD_ERROR = char_load_error
elif not VALID_POEM_LINES_MAP or all(
        not lines for lines in VALID_POEM_LINES_MAP.values()):
    GAME_LOAD_ERROR = "沒有載入到任何來源的有效五言詩句，無法開始遊戲。"
else:
    print("應用程式啟動資料載入完成.")

games: typing.Dict[int, typing.Dict[str, typing.Any]] = {}


def get_or_initialize_game_state(
    channel_id: int,
    preferred_source: str = DEFAULT_POEMS_SOURCE,
    force_new: bool = False
) -> typing.Tuple[typing.Optional[typing.Dict[str, typing.Any]],
                  typing.Optional[str], str]:
    global games, VALID_POEM_LINES_MAP, POEM_INFO_MAP_MAP, ALL_CHARACTERS_DATA

    state = games.get(channel_id)
    current_source_in_state = state.get(
        'current_poem_source',
        DEFAULT_POEMS_SOURCE) if state else DEFAULT_POEMS_SOURCE

    if not force_new:
        is_valid = False
        if state:
            try:
                target_line: typing.Optional[str] = state.get('target_line')
                char_histories: typing.Optional[typing.List[typing.Dict[
                    str, typing.Any]]] = state.get('char_histories')
                guess_count: typing.Optional[int] = state.get('guess_count')
                guess_count_history: typing.Optional[
                    typing.List[int]] = state.get('guess_count_history')
                recent_lines: typing.Any = state.get('recent_lines')
                thresholds: typing.Optional[typing.Dict[
                    str, float]] = state.get('thresholds')
                current_source: typing.Optional[str] = state.get(
                    'current_poem_source')
                target_poem_info: typing.Any = state.get('target_poem_info')

                if isinstance(target_line, str) and len(target_line) == 5 and \
                   isinstance(char_histories, list) and len(char_histories) == 5 and \
                   isinstance(guess_count, int) and isinstance(guess_count_history, list) and \
                   isinstance(recent_lines, (deque, list)) and isinstance(thresholds, dict) and \
                   isinstance(current_source, str) and isinstance(target_poem_info, dict):

                    chars_in_history_match = True
                    for i in range(5):
                        char_history = char_histories[i]
                        target_char = target_line[i]
                        if not isinstance(char_history,
                                          dict) or char_history.get(
                                              'target_char') != target_char:
                            chars_in_history_match = False
                            break
                        if target_char not in ALL_CHARACTERS_DATA:
                            chars_in_history_match = False
                            break
                        target_char_stroke_count = len(
                            ALL_CHARACTERS_DATA.get(target_char, []))
                        stroke_hists = char_history.get('stroke_histories')
                        if not isinstance(stroke_hists, list) or len(
                                stroke_hists) != target_char_stroke_count:
                            chars_in_history_match = False
                            break
                        if not all(
                                isinstance(sh, dict) and 'min_dist' in sh
                                and 'best_guess_char' in sh
                                and 'best_guess_stroke_index' in sh
                                for sh in stroke_hists):
                            chars_in_history_match = False
                            break

                    if chars_in_history_match:
                        is_valid = True
                        if not isinstance(recent_lines, deque):
                            state['recent_lines'] = deque(
                                recent_lines, maxlen=RECENT_LINES_LIMIT)

            except Exception as e:
                print(f"Channel {channel_id} 的遊戲狀態驗證失敗: {e}")
                traceback.print_exc()
                is_valid = False

        if is_valid:
            print(
                f"頻道 {channel_id} 找到有效遊戲狀態，目標詩句: '{state.get('target_line', '未知')}' 使用題庫: '{POEMS_SOURCES.get(state.get('current_poem_source', DEFAULT_POEMS_SOURCE), '未知')}'"
            )
            return state, None, state.get('current_poem_source',
                                          DEFAULT_POEMS_SOURCE)

    print(f"為頻道 {channel_id} 初始化或重置遊戲狀態 (force_new={force_new}).")
    current_source = preferred_source
    valid_lines = VALID_POEM_LINES_MAP.get(current_source)
    poem_info_map = POEM_INFO_MAP_MAP.get(current_source)

    if not valid_lines or poem_info_map is None:
        print(
            f"當前選擇的題庫 ('{POEMS_SOURCES.get(current_source, current_source)}') 沒有可用的詩句或資訊地圖載入失敗，嘗試回退到預設題庫."
        )
        if current_source != DEFAULT_POEMS_SOURCE:
            current_source = DEFAULT_POEMS_SOURCE
            valid_lines = VALID_POEM_LINES_MAP.get(current_source)
            poem_info_map = POEM_INFO_MAP_MAP.get(current_source)
            if not valid_lines or poem_info_map is None:
                print(
                    f"預設題庫 ('{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}') 也沒有可用的詩句或資訊地圖載入失敗. 無法初始化遊戲."
                )
                return None, f"無法初始化遊戲：沒有可用的詩句來源或資料不完整。", current_source
        else:
            print(
                f"預設題庫 ('{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}') 沒有可用的詩句或資訊地圖載入失敗. 無法初始化遊戲."
            )
            return None, f"無法初始化遊戲：沒有可用的詩句來源或資料不完整。", current_source

    guess_count_history = games.get(channel_id,
                                    {}).get('guess_count_history', [])
    recent_lines_list = games.get(channel_id,
                                  {}).get('recent_lines',
                                          deque(maxlen=RECENT_LINES_LIMIT))
    if not isinstance(recent_lines_list, deque):
        recent_lines_list = deque(recent_lines_list, maxlen=RECENT_LINES_LIMIT)
    recent_lines = recent_lines_list

    available_lines = [
        line for line in valid_lines if line not in recent_lines
    ]

    if not available_lines:
        print(f"頻道 {channel_id}: 沒有新的詩句可用 (所有詩句都在最近列表中)，重用最近的詩句.")
        if not valid_lines:
            return None, f"無法初始化遊戲：沒有可用的詩句來源。", current_source
        target_line = random.choice(valid_lines)
    else:
        target_line = random.choice(available_lines)

    target_poem_info = poem_info_map.get(target_line, {
        'title': '未知詩名',
        'content': [target_line]
    })

    recent_lines.append(target_line)

    char_histories = []
    for char in target_line:
        stroke_paths = ALL_CHARACTERS_DATA.get(char, [])
        stroke_histories = []
        for _ in stroke_paths:
            stroke_histories.append({
                'min_dist': float('inf'),
                'best_guess_char': None,
                'best_guess_stroke_index': None
            })
        char_histories.append({
            'target_char': char,
            'stroke_histories': stroke_histories
        })

    state = {
        'target_line': target_line,
        'target_poem_info': target_poem_info,
        'char_histories': char_histories,
        'recent_lines': recent_lines,
        'guess_count': 0,
        'guess_count_history': guess_count_history,
        'current_poem_source': current_source,
        'thresholds': {
            'thresh1': 10000,
            'thresh2': 25000
        }
    }
    games[channel_id] = state
    print(
        f"頻道 {channel_id} 遊戲已初始化，目標詩句: '{state.get('target_line', '未知')}' 使用題庫: '{POEMS_SOURCES.get(state.get('current_poem_source', '未知'))}'"
    )
    return state, None, current_source


bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())
tree = app_commands.CommandTree(bot)


@bot.event
async def on_ready():
    if GAME_LOAD_ERROR:
        print(f"嚴重錯誤: 遊戲資料載入失敗 - {GAME_LOAD_ERROR}")
        print("機器人將無法正常運作.")
    else:
        if bot.user is not None:
            print(f'{bot.user.name} has connected to Discord!')
        else:
            print('Bot has connected to Discord, but user is None.')
        print(
            f'Loaded data for {len(ALL_CHARACTERS_DATA)} characters and {sum(len(VALID_POEM_LINES_MAP.get(source, [])) for source in POEMS_SOURCES)} poem lines across {len(POEMS_SOURCES)} sources.'
        )
        if GLOBAL_POEMS_LOAD_ERRORS:
            print("詩詞檔案載入時發生以下錯誤:")
            for source, err in GLOBAL_POEMS_LOAD_ERRORS.items():
                print(f"- {source}: {err}")
        print("機器人已準備就緒，正在同步斜線指令...")
        try:
            await tree.sync()
            print("斜線指令同步完成.")
        except Exception as e:
            print(f"同步斜線指令失敗: {e}")
            traceback.print_exc()


@tree.command(name='poem', description='顯示當前詩句提示圖片，如果沒有遊戲則開始新遊戲')
async def poem(interaction: discord.Interaction):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法開始或顯示遊戲: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer()

    game_state, init_error, current_source = get_or_initialize_game_state(
        interaction.channel_id)

    if init_error or game_state is None:
        await interaction.followup.send(init_error or "無法獲取或初始化遊戲狀態.")
        return

    target_line = game_state['target_line']
    char_histories = game_state['char_histories']
    thresholds = game_state.get('thresholds', {
        'thresh1': 10000,
        'thresh2': 25000
    })
    guess_count = game_state.get('guess_count', 0)

    files: typing.List[discord.File] = []
    messages: typing.List[str] = []

    if guess_count == 0:
        messages.append(
            f"新的猜詩遊戲開始了！ (題庫: {POEMS_SOURCES.get(current_source, current_source)})"
        )
        messages.append(f"這是一句五言詩。請使用 `/guess [你的猜測]` 來猜測。")
    else:
        messages.append(f"這是目前的遊戲狀態。")
        messages.append(f"你已經猜了 {guess_count} 次。")

    messages.append("提示圖片如下：")

    for i in range(5):
        target_char = target_line[i]
        if target_char not in ALL_CHARACTERS_DATA:
            messages.append(
                f"內部錯誤: 目標字 '{target_char}' ({i+1}) 筆畫資料遺失. 請使用 `/newpoem` 重新開始."
            )
            continue

        char_history = char_histories[i]
        try:
            image_bytes = plot_character_colored_by_history(
                target_char, char_history, thresholds)
            if image_bytes:
                files.append(
                    discord.File(io.BytesIO(image_bytes),
                                 filename=f'char_{i}.png'))
            else:
                messages.append(f"位置 {i+1} 字 '{target_char}' 的圖片生成失敗。")
        except Exception as e:
            print(
                f"Error plotting image for char {target_char} at pos {i}: {e}")
            traceback.print_exc()
            messages.append(f"位置 {i+1} 字 '{target_char}' 的圖片生成時發生錯誤。")

    await interaction.followup.send('\n'.join(messages), files=files)


@tree.command(name='guess', description='猜測當前詩句。請輸入五個漢字。例如: /guess 花落知多少')
@app_commands.describe(guess_line="你的五字猜測")
async def guess(interaction: discord.Interaction, guess_line: str):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法進行猜測: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer()

    guess_line = guess_line.strip()

    if len(guess_line) != 5:
        await interaction.followup.send(
            f"請輸入剛好五個漢字進行猜測 (你輸入了 {len(guess_line)} 個字).")
        return

    missing_chars = [
        char for char in guess_line if char not in ALL_CHARACTERS_DATA
    ]
    if missing_chars:
        await interaction.followup.send(
            f"你的猜測詩句中包含不合法的字元: {''.join(missing_chars)}. 請輸入常見漢字.")
        return

    game_state = games.get(interaction.channel_id)

    if game_state is None or any(
            k not in game_state
            for k in ['target_line', 'char_histories', 'thresholds']):
        await interaction.followup.send("此頻道目前沒有正在進行的遊戲。請先使用 `/poem` 開始新遊戲。")
        return

    target_line: str = game_state['target_line']
    char_histories: typing.List[typing.Dict[
        str, typing.Any]] = game_state['char_histories']
    thresholds: typing.Dict[str,
                            float] = game_state.get('thresholds', {
                                'thresh1': 10000,
                                'thresh2': 25000
                            })

    if any(char not in ALL_CHARACTERS_DATA for char in target_line):
        await interaction.followup.send(
            f"內部錯誤: 當前目標詩句 '{target_line}' 包含無效字元或不在當前題庫中. 請使用 `/newpoem` 重新開始遊戲."
        )
        if interaction.channel_id in games:
            del games[interaction.channel_id]
        return

    game_state['guess_count'] = game_state.get('guess_count', 0) + 1
    current_guess_count = game_state['guess_count']

    messages: typing.List[str] = []
    files: typing.List[discord.File] = []
    partial_failure = False

    num_curve_points_for_dtw = 7

    if len(char_histories) != 5:
        messages.append(f"警告: 遊戲狀態字元歷史記錄長度異常 ({len(char_histories)}). 重置歷史記錄.")
        char_histories = []
        for char in target_line:
            stroke_paths = ALL_CHARACTERS_DATA.get(char, [])
            stroke_histories = [{
                'min_dist': float('inf'),
                'best_guess_char': None,
                'best_guess_stroke_index': None
            } for _ in range(len(stroke_paths))]
            char_histories.append({
                'target_char': char,
                'stroke_histories': stroke_histories
            })
        game_state['char_histories'] = char_histories

    for i in range(5):
        target_char = target_line[i]
        guess_char = guess_line[i]

        if i >= len(char_histories) or not isinstance(
                char_histories[i],
                dict) or char_histories[i].get('target_char') != target_char:
            messages.append(f"內部錯誤: 位置 {i+1} 的字元歷史記錄結構無效或與目標字元不匹配. 無法處理.")
            partial_failure = True
            if target_char in ALL_CHARACTERS_DATA:
                stroke_paths = ALL_CHARACTERS_DATA[target_char]
                char_histories[i] = {
                    'target_char':
                    target_char,
                    'stroke_histories': [{
                        'min_dist': float('inf'),
                        'best_guess_char': None,
                        'best_guess_stroke_index': None
                    } for _ in range(len(stroke_paths))]
                }
                messages.append(f"嘗試為位置 {i+1} 的字元 '{target_char}' 重設歷史記錄結構.")
            else:
                print(
                    f"Error: Cannot re-initialize history for {target_char} at pos {i}, data missing."
                )
                continue

        char_history = char_histories[i]

        if target_char not in ALL_CHARACTERS_DATA:
            messages.append(f"錯誤: 目標字 '{target_char}' ({i+1}) 筆畫資料遺失.")
            partial_failure = True
            continue

        target_char_stroke_count = len(ALL_CHARACTERS_DATA.get(
            target_char, []))
        if len(char_history.get('stroke_histories',
                                [])) != target_char_stroke_count:
            messages.append(
                f"警告: 位置 {i+1} 目標字 '{target_char}' 的筆畫歷史記錄長度異常 ({len(char_history.get('stroke_histories', []))}). 應為 {target_char_stroke_count}. 正在重設."
            )
            char_history['stroke_histories'] = [{
                'min_dist': float('inf'),
                'best_guess_char': None,
                'best_guess_stroke_index': None
            } for _ in range(target_char_stroke_count)]

        if guess_char == target_char:
            print(f"位置 {i+1}: 字元 '{target_char}' 猜對了，繪製黑色圖.")
            try:
                image_bytes = plot_solid_black_character(
                    target_char, ALL_CHARACTERS_DATA)
                if image_bytes:
                    files.append(
                        discord.File(io.BytesIO(image_bytes),
                                     filename=f'char_{i}.png'))
                else:
                    messages.append(
                        f"位置 {i+1} 字 '{target_char}': 正確字元的黑色圖片生成失敗.")
                    partial_failure = True
            except Exception as plot_e:
                print(
                    f"錯誤: 繪製位置 {i} 正確字元 '{target_char}' 的黑色圖片時發生錯誤: {plot_e}")
                traceback.print_exc()
                messages.append(f"位置 {i+1} 字 '{target_char}': 黑色圖片生成時發生意外錯誤.")
                partial_failure = True
            continue

        print(f"位置 {i+1}: 字元 '{target_char}' vs 猜測 '{guess_char}', 進行筆畫比較.")

        try:
            target_stroke_sequences_with_original_index = ALL_CHARACTERS_DTW_DATA.get(
                target_char, [])
            guess_stroke_sequences_with_original_index = ALL_CHARACTERS_DTW_DATA.get(
                guess_char, [])

            if not target_stroke_sequences_with_original_index:
                msg = f"位置 {i+1} ('{target_char}'): 目標字元筆畫資料無效或點數不足 (<2 點). 無法計算相似度."
                messages.append(msg)
                partial_failure = True
            elif not guess_stroke_sequences_with_original_index:
                msg = f"位置 {i+1} ('{guess_char}'): 猜測字元筆畫資料無效或點數不足 (<2 點). 無法計算相似度."
                messages.append(msg)
                partial_failure = True
            else:
                stroke_dtw_matrix = np.full(
                    (len(guess_stroke_sequences_with_original_index),
                     len(target_stroke_sequences_with_original_index)), np.inf)

                for seq_g_idx, (seq_g, original_g_idx) in enumerate(
                        guess_stroke_sequences_with_original_index):
                    for seq_t_idx, (seq_t, original_t_idx) in enumerate(
                            target_stroke_sequences_with_original_index):
                        if not isinstance(seq_g, np.ndarray) or seq_g.shape[0] < 2 or \
                           not isinstance(seq_t, np.ndarray) or seq_t.shape[0] < 2:
                            print(
                                f"Warning: 無效或過短的筆劃序列用於 FastDTW 於位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}). 跳過."
                            )
                            stroke_dtw_matrix[seq_g_idx,
                                              seq_t_idx] = float('inf')
                            continue
                        try:
                            dtw_distance, path = fastdtw(seq_g,
                                                         seq_t,
                                                         dist=euclidean)
                            stroke_dtw_matrix[seq_g_idx,
                                              seq_t_idx] = dtw_distance
                        except Exception as dtw_e:
                            print(
                                f"Warning: FastDTW 計算失敗於位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}): {dtw_e}"
                            )
                            traceback.print_exc()
                            stroke_dtw_matrix[seq_g_idx,
                                              seq_t_idx] = float('inf')

                for target_original_index in range(target_char_stroke_count):
                    target_comp_index = next(
                        (idx for idx, (seq, orig_idx) in enumerate(
                            target_stroke_sequences_with_original_index)
                         if orig_idx == target_original_index), None)

                    min_dist_for_this_target_stroke = float('inf')
                    best_matching_guess_original_index_for_this_target_stroke: typing.Optional[
                        int] = None

                    if target_comp_index is not None and target_comp_index < stroke_dtw_matrix.shape[
                            1]:
                        col_distances = stroke_dtw_matrix[:, target_comp_index]

                        if col_distances.size > 0 and np.min(
                                col_distances) != np.inf:
                            min_dist_in_col = np.min(col_distances)
                            min_dist_for_this_target_stroke = min_dist_in_col

                            min_row_comp_index = np.argmin(col_distances)

                            if min_row_comp_index < len(
                                    guess_stroke_sequences_with_original_index
                            ):
                                best_matching_guess_original_index_for_this_target_stroke = guess_stroke_sequences_with_original_index[
                                    min_row_comp_index][1]

                    if target_original_index < len(
                            char_history['stroke_histories']):
                        hist_stroke_info = char_history['stroke_histories'][
                            target_original_index]
                        if min_dist_for_this_target_stroke < hist_stroke_info.get(
                                'min_dist', float('inf')):
                            hist_stroke_info[
                                'min_dist'] = min_dist_for_this_target_stroke
                            hist_stroke_info['best_guess_char'] = guess_char
                            hist_stroke_info[
                                'best_guess_stroke_index'] = best_matching_guess_original_index_for_this_target_stroke
                    else:
                        messages.append(
                            f"內部錯誤: 位置 {i+1} 目標字 '{target_char}' 的筆畫歷史記錄索引異常 ({target_original_index})."
                        )
                        partial_failure = True

        except Exception as comp_e:
            print(
                f"錯誤: 處理位置 {i} 字元 '{target_char}' vs '{guess_char}' 的比較邏輯時發生錯誤: {comp_e}"
            )
            traceback.print_exc()
            messages.append(f"位置 {i+1} 字元比較失敗: {comp_e}")
            partial_failure = True

        try:
            image_bytes = plot_character_colored_by_history(
                target_char, char_history, thresholds)
            if image_bytes:
                files.append(
                    discord.File(io.BytesIO(image_bytes),
                                 filename=f'char_{i}.png'))
            else:
                messages.append(f"位置 {i+1} 字 '{target_char}': 圖片生成失敗.")
                partial_failure = True
        except Exception as plot_e:
            print(f"錯誤: 繪製位置 {i} 的圖片時發生錯誤: {plot_e}")
            traceback.print_exc()
            messages.append(f"位置 {i+1} 字 '{target_char}': 圖片生成時發生意外錯誤.")
            partial_failure = True

    is_correct_guess = (guess_line == target_line)

    response_message = f"你的猜測: **{guess_line}** (第 {current_guess_count} 次猜測)\n"
    if partial_failure:
        response_message += "圖片或比較過程中發生部分錯誤.\n"

    if is_correct_guess:
        response_message += "恭喜你，猜對了整句詩!\n"
        poem_info = game_state.get('target_poem_info')
        if poem_info and isinstance(poem_info, dict):
            response_message += f"這是出自 **《{poem_info.get('title', '未知詩名')}》**：\n"
            content = poem_info.get('content', [])
            if isinstance(content, list):
                response_message += '\n'.join(content) + '\n'
            else:
                response_message += '詩詞內容無效.\n'

        game_state['guess_count_history'] = game_state.get(
            'guess_count_history', [])
        game_state['guess_count_history'].append(current_guess_count)
        del games[interaction.channel_id]

        response_message += "使用 `/poem` 開始新一輪遊戲，或使用 `/newpoem` 切換詩句並開始新遊戲，或使用 `/stats` 查看統計。\n"

        files = []
        for i in range(5):
            target_char = target_line[i]
            if target_char not in ALL_CHARACTERS_DATA:
                messages.append(
                    f"內部錯誤: 字元 '{target_char}' ({i+1}) 筆畫資料遺失，無法繪製最終圖片.")
                continue
            try:
                image_bytes = plot_solid_black_character(
                    target_char, ALL_CHARACTERS_DATA)
                if image_bytes:
                    files.append(
                        discord.File(io.BytesIO(image_bytes),
                                     filename=f'char_{i}.png'))
                else:
                    messages.append(
                        f"位置 {i+1} 字 '{target_char}': 正確字元的黑色圖片生成失敗.")
                    partial_failure = True
            except Exception as plot_e:
                print(
                    f"錯誤: 繪製位置 {i} 正確字元 '{target_char}' 的黑色圖片時發生錯誤: {plot_e}")
                traceback.print_exc()
                messages.append(f"位置 {i+1} 字 '{target_char}': 黑色圖片生成時發生意外錯誤.")
                partial_failure = True

    elif current_guess_count >= 15:
        response_message += f"猜測次數過多 ({current_guess_count} 次), 本輪遊戲結束。\n"
        response_message += f"正確詩句是: **{target_line}**\n"
        poem_info = game_state.get('target_poem_info')
        if poem_info and isinstance(poem_info, dict):
            response_message += f"這是出自 **《{poem_info.get('title', '未知詩名')}》**：\n"
            content = poem_info.get('content', [])
            if isinstance(content, list):
                response_message += '\n'.join(content) + '\n'
            else:
                response_message += '詩詞內容無效.\n'

        game_state['guess_count_history'] = game_state.get(
            'guess_count_history', [])
        game_state['guess_count_history'].append(current_guess_count)
        del games[interaction.channel_id]

        response_message += "使用 `/poem` 開始新一輪遊戲，或使用 `/newpoem` 切換詩句並開始新遊戲，或使用 `/stats` 查看統計。\n"

        files = []
        for i in range(5):
            target_char = target_line[i]
            if target_char not in ALL_CHARACTERS_DATA:
                messages.append(
                    f"內部錯誤: 字元 '{target_char}' ({i+1}) 筆畫資料遺失，無法繪製最終圖片.")
                continue
            try:
                image_bytes = plot_solid_black_character(
                    target_char, ALL_CHARACTERS_DATA)
                if image_bytes:
                    files.append(
                        discord.File(io.BytesIO(image_bytes),
                                     filename=f'char_{i}.png'))
                else:
                    messages.append(
                        f"位置 {i+1} 字 '{target_char}': 正確字元的黑色圖片生成失敗.")
                    partial_failure = True
            except Exception as plot_e:
                print(
                    f"錯誤: 繪製位置 {i} 正確字元 '{target_char}' 的黑色圖片時發生錯誤: {plot_e}")
                traceback.print_exc()
                messages.append(f"位置 {i+1} 字 '{target_char}': 黑色圖片生成時發生意外錯誤.")
                partial_failure = True

    else:
        response_message += f"繼續努力! 這是你第 {current_guess_count} 次猜測。\n"
        response_message += f"(閾值: 黑 < {thresholds.get('thresh1', 10000)}, 灰 < {thresholds.get('thresh2', 25000)})\n"
        response_message += "提示圖片已更新："

    if messages:
        await interaction.followup.send('\n'.join(messages))

    if files:
        await interaction.followup.send(response_message, files=files)
    elif not messages:
        await interaction.followup.send(response_message + " (沒有圖片需要發送)")


@tree.command(
    name='newpoem',
    description=f'開始一個新的詩詞遊戲。可以指定題庫: {", ".join(POEMS_SOURCES.keys())}')
@app_commands.describe(source=f'選擇題庫來源: {", ".join(POEMS_SOURCES.keys())}')
async def newpoem(interaction: discord.Interaction,
                  source: str = DEFAULT_POEMS_SOURCE):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法開始新遊戲: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer()

    if source not in POEMS_SOURCES:
        available_sources = ", ".join(
            [f"{k} ({v})" for k, v in POEMS_SOURCES.items()])
        await interaction.followup.send(
            f"無效的題庫來源 '{source}'. 可用的題庫: {available_sources}. 使用預設題庫 `{DEFAULT_POEMS_SOURCE}`."
        )
        source = DEFAULT_POEMS_SOURCE

    print(f"頻道 {interaction.channel_id}: 收到開始新遊戲指令，來源: {source}")

    old_history: typing.List[int] = games.get(interaction.channel_id,
                                              {}).get('guess_count_history',
                                                      [])
    games[interaction.channel_id] = {'guess_count_history': old_history}

    game_state, init_error, current_source_used = get_or_initialize_game_state(
        interaction.channel_id, preferred_source=source, force_new=True)

    if init_error or game_state is None:
        await interaction.followup.send(f"無法初始化新遊戲: {init_error or '狀態無效'}")
        if interaction.channel_id in games:
            del games[interaction.channel_id]
        return

    target_line = game_state['target_line']
    char_histories = game_state['char_histories']
    thresholds = game_state.get('thresholds', {
        'thresh1': 10000,
        'thresh2': 25000
    })

    files: typing.List[discord.File] = []
    messages: typing.List[str] = [
        f"新的猜詩遊戲開始了！ (題庫: {POEMS_SOURCES.get(current_source_used, current_source_used)})"
    ]
    messages.append(f"這是一句五言詩。請使用 `/guess [你的猜測]` 來猜測。")
    messages.append("提示圖片如下：")

    for i in range(5):
        target_char = target_line[i]
        if target_char not in ALL_CHARACTERS_DATA:
            messages.append(
                f"內部錯誤: 目標字 '{target_char}' ({i+1}) 筆畫資料遺失. 請使用 `/newpoem` 重新開始."
            )
            continue

        if i >= len(char_histories) or not isinstance(
                char_histories[i],
                dict) or char_histories[i].get('target_char') != target_char:
            messages.append(f"內部錯誤: 位置 {i+1} 的字元歷史記錄結構無效或與目標字元不匹配. 無法繪製圖片.")
            continue
        char_history = char_histories[i]

        try:
            image_bytes = plot_character_colored_by_history(
                target_char, char_history, thresholds)
            if image_bytes:
                files.append(
                    discord.File(io.BytesIO(image_bytes),
                                 filename=f'char_{i}.png'))
            else:
                messages.append(f"位置 {i+1} 字 '{target_char}' 的圖片生成失敗。")
        except Exception as e:
            print(
                f"Error plotting initial image for char {target_char} at pos {i}: {e}"
            )
            traceback.print_exc()
            messages.append(f"位置 {i+1} 字 '{target_char}' 的圖片生成時發生錯誤。")

    await interaction.followup.send('\n'.join(messages), files=files)


@tree.command(name='stats', description='顯示你的猜測次數統計圖')
async def stats(interaction: discord.Interaction):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法顯示統計: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer()

    game_state = games.get(interaction.channel_id)

    guess_count_history = game_state.get('guess_count_history',
                                         []) if game_state else []

    if not guess_count_history:
        await interaction.followup.send("目前沒有完成的詩詞統計資料。完成一首詩後再試試！")
        return

    try:
        image_bytes = generate_stats_plot_buffer(guess_count_history)
        if image_bytes:
            await interaction.followup.send("你的猜測次數統計圖：",
                                            file=discord.File(
                                                io.BytesIO(image_bytes),
                                                filename='stats.png'))
        else:
            await interaction.followup.send("無法生成統計圖表。")
    except Exception as e:
        print(f"生成並發送統計圖表時發生錯誤: {e}")
        traceback.print_exc()
        await interaction.followup.send("生成統計圖表時發生錯誤。")


@tree.command(name='difficulty', description='設定遊戲難度 (1-6)')
@app_commands.describe(level=f'難度等級 (1-6)')
async def difficulty(interaction: discord.Interaction, level: int):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法設定難度: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    if level not in DIFFICULTY_THRESHOLDS:
        await interaction.followup.send(f"無效的難度等級。請輸入 1 到 6 之間的數字。")
        return

    game_state = games.get(interaction.channel_id)

    if game_state is None or any(
            k not in game_state
            for k in ['target_line', 'char_histories', 'thresholds']):
        await interaction.followup.send("此頻道目前沒有正在進行的遊戲。請先使用 `/poem` 開始新遊戲。")
        return

    thresh1, thresh2 = DIFFICULTY_THRESHOLDS[level]
    game_state['thresholds'] = {'thresh1': thresh1, 'thresh2': thresh2}
    games[interaction.channel_id] = game_state

    await interaction.followup.send(
        f"已將遊戲難度設定為等級 {level} (閾值: 黑 < {thresh1}, 灰 < {thresh2}).")


@tree.command(name='set_thresholds', description='手動設定相似度閾值')
@app_commands.describe(thresh1='黑色的閾值 (距離小於此值為黑色)',
                       thresh2='灰色的閾值 (距離小於此值且大於等於黑色閾值為灰色)')
@app_commands.checks.has_permissions(manage_channels=True)
async def set_thresholds_cmd(interaction: discord.Interaction, thresh1: float,
                             thresh2: float):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法設定閾值: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    if thresh1 < 0 or thresh2 < 0 or thresh1 > thresh2:
        await interaction.followup.send("無效的閾值輸入。請確保 0 ≤ 黑閾值 ≤ 灰閾值。")
        return

    game_state = games.get(interaction.channel_id)

    if game_state is None or any(
            k not in game_state
            for k in ['target_line', 'char_histories', 'thresholds']):
        await interaction.followup.send("此頻道目前沒有正在進行的遊戲狀態。請先使用 `/poem` 開始新遊戲。")
        return

    game_state['thresholds'] = {'thresh1': thresh1, 'thresh2': thresh2}
    games[interaction.channel_id] = game_state

    await interaction.followup.send(f"已將閾值設定為：黑 < {thresh1}, 灰 < {thresh2}")


@set_thresholds_cmd.error
async def set_thresholds_cmd_error(interaction: discord.Interaction,
                                   error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("你沒有權限設定閾值。", ephemeral=True)
    elif isinstance(error, app_commands.BadArgument):
        await interaction.response.send_message(
            "無效的閾值格式。請輸入兩個數字。例如: `/set_thresholds 10000 25000`",
            ephemeral=True)
    else:
        print(f"Unexpected error in set_thresholds_cmd: {error}")
        traceback.print_exc()
        await interaction.response.send_message("設定閾值時發生未知錯誤。", ephemeral=True)


BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
if not BOT_TOKEN:
    print("錯誤: 未找到環境變數 'DISCORD_BOT_TOKEN'. 請在 Replit Secrets 或環境變數中設定.")
else:
    try:
        bot.run(BOT_TOKEN)
    except Exception as e:
        print(f"運行 Discord 機器人時發生錯誤: {e}")
        traceback.print_exc()
