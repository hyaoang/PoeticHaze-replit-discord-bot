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
from discord import app_commands, ui
from collections import deque, Counter
import typing

GameState = typing.Dict[str, typing.Any]
PoemInfo = typing.Dict[str, typing.Dict[str, typing.Any]]
CharData = typing.Dict[str, typing.List[str]]
DtwData = typing.Dict[str, typing.List[typing.Tuple[np.ndarray, int]]]
CharHistory = typing.List[typing.Dict[str, typing.Any]]
Thresholds = typing.Dict[str, float]

DATA_FILENAME = "makemeahanzi/graphics.txt"
POEMS_SOURCES = {'poems.json': '唐詩三百首', 'easypoems.json': '常見唐詩(較簡單)'}

POEMS_SOURCE_MAP = {'1': 'poems.json', '2': 'easypoems.json'}
POEMS_SOURCE_MAP.update(POEMS_SOURCES)
DEFAULT_POEMS_SOURCE = 'poems.json'

ALL_CHARACTERS_DATA: CharData = {}
ALL_CHARACTERS_DTW_DATA: DtwData = {}
VALID_POEM_LINES_MAP: typing.Dict[str, typing.List[str]] = {}
POEM_INFO_MAP_MAP: typing.Dict[str, PoemInfo] = {}
GAME_LOAD_ERROR: typing.Optional[str] = None

RECENT_LINES_LIMIT = 10

FONT_FILE = 'NotoSansTC-Regular.ttf'
FONT_PATH = os.path.join(os.path.dirname(__file__), 'fonts', FONT_FILE)

channel_preferred_thresholds: typing.Dict[int, Thresholds] = {}

channel_last_used_source: typing.Dict[int, str] = {}

try:
    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"成功設定中文字體: {font_prop.get_name()}")
    else:
        print(f"警告: 中文字體檔案未找到於 {FONT_PATH}. 中文字符可能無法正常顯示.")
except Exception as e:
    print(f"警告: 設定中文字體時發生錯誤: {e}. 中文字符可能無法正常顯示.")
    traceback.print_exc()

DIFFICULTY_THRESHOLDS = {
    1: [500, 1000],
    2: [1000, 1500],
    3: [2000, 2500],
    4: [4000, 6000],
    5: [8000, 10000],
    6: [12000, 20000]
}

channel_preferred_thresholds: typing.Dict[int, Thresholds] = {}


def get_difficulty_display(thresholds: Thresholds) -> str:
    t1, t2 = thresholds.get('thresh1', 0), thresholds.get('thresh2', 0)
    t1_int, t2_int = int(t1), int(t2)
    for level, (d1, d2) in DIFFICULTY_THRESHOLDS.items():
        if t1_int == d1 and t2_int == d2:
            return str(level)
    return f"自定義 ({int(t1)}, {int(t2)})"


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
        traceback.print_exc()
        return np.array([512.0, 512.0])


def load_character_data(
        filepath: str) -> typing.Tuple[CharData, typing.Optional[str]]:
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
                    traceback.print_exc()
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
    filepath: str, char_data: CharData
) -> typing.Tuple[typing.List[str], PoemInfo, typing.Optional[str]]:
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
        char_data: CharData,
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


def precompute_dtw_data(char_data: CharData) -> DtwData:
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


def plot_character_strokes_on_ax(ax: plt.Axes, target_char: str,
                                 char_history: typing.Dict[str, typing.Any],
                                 thresholds: Thresholds):
    target_stroke_paths = ALL_CHARACTERS_DATA.get(target_char, [])
    target_stroke_histories = char_history.get('stroke_histories', [])

    if not target_stroke_paths:
        ax.text(0.5,
                0.5,
                '無筆畫',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=10,
                color='red')
        return

    if len(target_stroke_paths) != len(target_stroke_histories):
        print(
            f"Warning: Character '{target_char}' stroke count ({len(target_stroke_paths)}) mismatch with history ({len(target_stroke_histories)}). Resetting history for plotting."
        )
        char_history['stroke_histories'] = [{
            'min_dist': float('inf'),
            'best_guess_char': None,
            'best_guess_stroke_index': None
        } for _ in range(len(target_stroke_paths))]
        target_stroke_histories = char_history['stroke_histories']

    threshold1 = thresholds.get('thresh1', float('inf'))
    threshold2 = thresholds.get('thresh2', float('inf'))
    color_pure_black = '0.0'
    color_fixed_gray = '0.5'

    for target_stroke_index, hist_info in enumerate(target_stroke_histories):
        historical_min_dist = hist_info.get('min_dist', float('inf'))
        historical_guess_char = hist_info.get('best_guess_char')
        historical_guess_stroke_index = hist_info.get(
            'best_guess_stroke_index')

        plot_path_str = None
        color_to_use = None
        historical_guess_centroid = None

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
                    color_to_use = color_pure_black

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
                    color_to_use = color_fixed_gray

        if plot_path_str is not None and color_to_use is not None:
            try:
                if target_stroke_index < len(target_stroke_paths):
                    target_stroke_path_str = target_stroke_paths[
                        target_stroke_index]
                    target_centroid = calculate_stroke_centroid(
                        target_stroke_path_str)
                else:
                    target_centroid = np.array([512.0, 512.0])

                translation = np.array([0.0, 0.0])
                if isinstance(target_centroid, np.ndarray) and isinstance(
                        historical_guess_centroid,
                        np.ndarray) and target_centroid.shape == (
                            2, ) and historical_guess_centroid.shape == (2, ):
                    translation = target_centroid - historical_guess_centroid
                else:
                    pass

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

            except Exception as e:
                print(
                    f"Warning: Error plotting character '{target_char}' stroke {target_stroke_index+1}'s history: {e}"
                )
                traceback.print_exc()
                pass


def plot_combined_character_image(
        game_state: GameState,
        is_game_over: bool = False) -> typing.Optional[bytes]:
    target_line: str = game_state.get('target_line', '')
    char_histories: CharHistory = game_state.get('char_histories', [])
    thresholds: Thresholds = game_state.get('thresholds', {
        'thresh1': 10000,
        'thresh2': 25000
    })

    if len(target_line) != 5 or len(char_histories) != 5:
        print(
            "Error: Invalid game state for plotting combined image (line or history length)."
        )
        return None

    fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.5), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, ax in enumerate(axes):
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        padding = 50
        ax.set_xlim(0 - padding, 1024 + padding)
        ax.set_ylim(0 - padding, 1024 + padding)

        target_char = target_line[i]
        if target_char not in ALL_CHARACTERS_DATA:
            ax.text(0.5,
                    0.5,
                    '資料遺失',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=10,
                    color='red')
            continue

        if i >= len(char_histories) or not isinstance(
                char_histories[i],
                dict) or char_histories[i].get('target_char') != target_char:
            print(
                f"Error: Character history mismatch for plotting character {target_char} at pos {i}. History target char: {char_histories[i].get('target_char')}, Expected: {target_char}"
            )
            ax.text(0.5,
                    0.5,
                    '歷史錯誤',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=10,
                    color='red')
            continue

        char_history = char_histories[i]

        if is_game_over:
            target_stroke_paths = ALL_CHARACTERS_DATA.get(target_char, [])
            if not target_stroke_paths:
                ax.text(0.5,
                        0.5,
                        '無筆畫',
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=10,
                        color='red')
                continue
            for path_str in target_stroke_paths:
                try:
                    outline_points = get_path_outline_points(
                        path_str, num_curve_points=30)
                    if isinstance(
                            outline_points, np.ndarray
                    ) and outline_points.ndim == 2 and outline_points.shape[
                            0] > 0:
                        ax.fill(outline_points[:, 0],
                                outline_points[:, 1],
                                color='0.0',
                                zorder=2)
                except Exception as e:
                    print(
                        f"Warning: Error plotting final character '{target_char}' stroke: {e}"
                    )
                    traceback.print_exc()
                    pass

        else:
            plot_character_strokes_on_ax(ax, target_char, char_history,
                                         thresholds)

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
        print(f"錯誤: 儲存合併圖片時發生錯誤: {e}")
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)


def generate_stats_plot_buffer(
        guess_counts: typing.List[int]) -> typing.Optional[bytes]:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    if len(guess_counts) < 3:
        ax.text(0.5,
                0.5,
                f'完成詩詞數量不足 ({len(guess_counts)} 首)，無法生成統計圖表',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=10)
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
        all_possible_counts = list(
            range(1, (max(guess_numbers) if guess_numbers else 1) + 1))
        ax.set_xticks(all_possible_counts)
        ax.set_xticklabels([str(x) for x in all_possible_counts])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width() / 2,
                         yval + 0.1,
                         str(int(yval)),
                         ha='center',
                         va='bottom')
        y_upper_limit = max_frequency + 0.1 + (max_frequency * 0.1)
        ax.set_ylim(0, y_upper_limit if y_upper_limit > 1 else 1.5)
        ax.set_xlim(
            min(guess_numbers) - 0.5 if guess_numbers else 0,
            max(guess_numbers) + 0.5 if guess_numbers else 5)

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


def generate_initial_text_line(game_state: GameState) -> str:
    target_line: str = game_state.get('target_line', '?????')
    return "　".join([":x:"] * 5)


def generate_final_text_line(game_state: GameState) -> str:
    target_line: str = game_state.get('target_line', '?????')
    revealed_parts = []
    for target_char in target_line:
        revealed_parts.append(f"[{target_char}](http://hi)")
    return "　".join(revealed_parts)


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

games: typing.Dict[int, GameState] = {}

channel_preferred_thresholds: typing.Dict[int, Thresholds] = {}


def get_or_initialize_game_state(
    channel_id: int,
    preferred_source: str = DEFAULT_POEMS_SOURCE,
    force_new: bool = False
) -> typing.Tuple[typing.Optional[GameState], typing.Optional[str], str]:
    # 在 global 聲明中加入 channel_last_used_source
    global games, VALID_POEM_LINES_MAP, POEM_INFO_MAP_MAP, ALL_CHARACTERS_DATA, channel_preferred_thresholds, channel_last_used_source

    state = games.get(channel_id)

    if not force_new:
        is_valid = False
        if state:
            try:
                target_line: typing.Optional[str] = state.get('target_line')
                char_histories: typing.Optional[CharHistory] = state.get(
                    'char_histories')
                guess_count: typing.Optional[int] = state.get('guess_count')
                guess_count_history: typing.Optional[
                    typing.List[int]] = state.get('guess_count_history')
                recent_lines: typing.Any = state.get('recent_lines')
                thresholds: typing.Optional[Thresholds] = state.get(
                    'thresholds')
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
                        target_char = target_line[i]
                        if target_char not in ALL_CHARACTERS_DATA:
                            chars_in_history_match = False
                            print(
                                f"Channel {channel_id} State Invalid: Target character '{target_char}' at pos {i} missing data."
                            )
                            break

                        char_history = char_histories[i]
                        if not isinstance(char_history,
                                          dict) or char_history.get(
                                              'target_char') != target_char:
                            chars_in_history_match = False
                            print(
                                f"Channel {channel_id} State Invalid: History at pos {i} structure invalid or target_char mismatch."
                            )
                            break

                        target_char_stroke_count = len(
                            ALL_CHARACTERS_DATA.get(target_char, []))
                        stroke_hists = char_history.get('stroke_histories')
                        if not isinstance(stroke_hists, list) or len(
                                stroke_hists) != target_char_stroke_count:
                            chars_in_history_match = False
                            print(
                                f"Channel {channel_id} State Invalid: Stroke history length mismatch for '{target_char}' at pos {i}. Expected {target_char_stroke_count}, got {len(stroke_hists)}"
                            )
                            break
                        if not all(
                                isinstance(sh, dict) and 'min_dist' in sh
                                and 'best_guess_char' in sh
                                and 'best_guess_stroke_index' in sh
                                for sh in stroke_hists):
                            chars_in_history_match = False
                            print(
                                f"Channel {channel_id} State Invalid: Stroke history entry structure invalid for '{target_char}' at pos {i}."
                            )
                            break

                    if chars_in_history_match:
                        is_valid = True
                        if not isinstance(recent_lines, deque):
                            state['recent_lines'] = deque(
                                recent_lines, maxlen=RECENT_LINES_LIMIT)
                        if 'thresholds' not in state or not isinstance(
                                state['thresholds'], dict):
                            state[
                                'thresholds'] = channel_preferred_thresholds.get(
                                    channel_id, {
                                        'thresh1': 10000,
                                        'thresh2': 25000
                                    })
                        if 'guess_count_history' not in state or not isinstance(
                                state['guess_count_history'], list):
                            state['guess_count_history'] = []

            except Exception as e:
                print(f"Channel {channel_id} 的遊戲狀態驗證失敗: {e}")
                traceback.print_exc()
                is_valid = False

        if is_valid:
            print(
                f"頻道 {channel_id} 找到有效遊戲狀態，目標詩句: '{state.get('target_line', '未知')}' 使用題庫: '{POEMS_SOURCES.get(state.get('current_poem_source', DEFAULT_POEMS_SOURCE), '未知')}'"
            )
            # 如果找到了有效的現有狀態，也順便確保記錄下它正在使用的來源（雖然結束時會記錄，這裡加強一下）
            if 'current_poem_source' in state:
                channel_last_used_source[channel_id] = state[
                    'current_poem_source']
            return state, None, state.get('current_poem_source',
                                          DEFAULT_POEMS_SOURCE)

    print(f"為頻道 {channel_id} 初始化或重置遊戲狀態 (force_new={force_new}).")

    # ===== 修改點 Start: 決定本次遊戲要嘗試載入的詩詞來源 =====
    source_to_attempt = preferred_source  # 預設先嘗試指令傳入的 (或函數預設的) 來源

    # 如果傳入的來源是 DEFAULT 來源，才考慮使用頻道上次成功的來源
    if preferred_source == DEFAULT_POEMS_SOURCE:
        last_source = channel_last_used_source.get(channel_id)
        # 檢查上次使用的來源是否存在於已知的 SOURCES 中，並且確實有載入到有效的詩句
        if last_source and last_source in POEMS_SOURCES and VALID_POEM_LINES_MAP.get(
                last_source):
            source_to_attempt = last_source  # 使用頻道上次成功的來源
            print(
                f"頻道 {channel_id}: 根據上次記錄，優先嘗試使用題庫 '{POEMS_SOURCES.get(source_to_attempt, source_to_attempt)}'."
            )
        else:
            print(
                f"頻道 {channel_id}: 未找到有效的上次使用記錄或記錄無效，使用預設題庫 '{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}'."
            )
    else:
        print(
            f"頻道 {channel_id}: 指令明確指定了題庫 '{POEMS_SOURCES.get(preferred_source, preferred_source)}', 優先使用此來源."
        )

    current_source = source_to_attempt  # 這是我們最終決定要檢查/使用的來源
    # ===== 修改點 End =====

    valid_lines = VALID_POEM_LINES_MAP.get(current_source)
    poem_info_map = POEM_INFO_MAP_MAP.get(current_source)

    # 現有的來源驗證和回退到 DEFAULT_POEMS_SOURCE 的邏輯
    # 如果 current_source 無效，這部分會檢查是否能回退
    if not valid_lines or poem_info_map is None:
        print(
            f"選定的題庫 ('{POEMS_SOURCES.get(current_source, current_source)}') 無效或沒有載入到有效詩句，嘗試回退到預設題庫."
        )
        # 如果當前嘗試的來源無效，無論它是傳入的還是上次使用的，都嘗試預設來源
        if current_source != DEFAULT_POEMS_SOURCE:  # 避免無限迴圈檢查同一個無效來源
            current_source = DEFAULT_POEMS_SOURCE
            valid_lines = VALID_POEM_LINES_MAP.get(current_source)
            poem_info_map = POEM_INFO_MAP_MAP.get(current_source)

        if not valid_lines or poem_info_map is None:
            error_msg = f"預設題庫 ('{POEMS_SOURCES.get(DEFAULT_POEMS_SOURCE, DEFAULT_POEMS_SOURCE)}') 也無效. 無法初始化遊戲."
            print(error_msg)
            # 注意：此處初始化失敗，不會更新 channel_last_used_source
            return None, error_msg, current_source  # 返回錯誤並指出最終嘗試的來源
        else:
            print(
                f"頻道 {channel_id}: 回退到預設題庫 '{POEMS_SOURCES.get(current_source, current_source)}' 成功."
            )

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
        if not valid_lines:  # 這裡應該不會發生，因為前面已經驗證過 valid_lines 不為空
            return None, f"內部錯誤: 無可用詩句。", current_source
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

    initial_thresholds = channel_preferred_thresholds.get(
        channel_id, {
            'thresh1': 10000,
            'thresh2': 25000
        })

    state = {
        'target_line': target_line,
        'target_poem_info': target_poem_info,
        'char_histories': char_histories,
        'recent_lines': recent_lines,
        'guess_count': 0,
        'guess_count_history': guess_count_history,
        'current_poem_source': current_source,  # 確保這裡記錄的是實際使用的來源 (可能是回退後的)
        'thresholds': initial_thresholds
    }
    games[channel_id] = state

    # ===== 修改點 Start: 成功初始化遊戲後，記錄該頻道本次使用的來源 =====
    channel_last_used_source[channel_id] = current_source
    print(
        f"頻道 {channel_id}: 成功初始化遊戲，已將使用題庫 '{POEMS_SOURCES.get(current_source, current_source)}' 記錄為頻道偏好."
    )
    # ===== 修改點 End =====

    print(
        f"頻道 {channel_id} 遊戲已初始化，目標詩句: '{state.get('target_line', '未知')}' 使用題庫: '{POEMS_SOURCES.get(state.get('current_poem_source', '未知'))}'"
    )
    return state, None, current_source


def process_guess(
    channel_id: int, guess_line: str
) -> typing.Tuple[str, typing.List[discord.File], typing.Optional[ui.View]]:
    game_state = games.get(channel_id)

    messages = []
    partial_failure = False

    if game_state is None or any(
            k not in game_state
            for k in ['target_line', 'char_histories', 'thresholds']):
        print(f"頻道 {channel_id}: 無正在進行遊戲或狀態無效，猜測 '{guess_line}' 觸發新遊戲初始化.")

        old_history: typing.List[int] = games.get(channel_id, {}).get(
            'guess_count_history', [])
        temp_history = old_history
        if channel_id in games:
            del games[channel_id]
        game_state, init_error, current_source = get_or_initialize_game_state(
            channel_id, force_new=True)

        if game_state:
            if 'guess_count_history' not in game_state or not isinstance(
                    game_state['guess_count_history'], list):
                game_state['guess_count_history'] = []
            game_state['guess_count_history'] = temp_history

        if init_error or game_state is None:
            return init_error or "無法初始化新遊戲來處理猜測.", [], None
        messages.append(
            f"此頻道沒有正在進行的遊戲，已為你開始一個新的猜詩遊戲！ (題庫: {POEMS_SOURCES.get(current_source, current_source)})"
        )
        messages.append(f"這是一句五言詩。")
        messages.append(f"> **{guess_line}**")

    else:
        messages.append(f"> **{guess_line}**")

    target_line: str = game_state['target_line']
    char_histories: CharHistory = game_state['char_histories']
    thresholds: Thresholds = game_state.get('thresholds', {
        'thresh1': 10000,
        'thresh2': 25000
    })

    if any(char not in ALL_CHARACTERS_DATA for char in target_line):
        del games[channel_id]
        return f"內部錯誤: 當前目標詩句 '{target_line}' 包含無效字元或不在資料中. 遊戲已重置，請重新開始 (`/newpoem`).", [], None

    game_state['guess_count'] = game_state.get('guess_count', 0) + 1
    current_guess_count = game_state['guess_count']

    num_curve_points_for_dtw = 7

    if len(char_histories) != 5 or any(
            char_histories[i].get('target_char') != target_line[i]
            for i in range(5)):
        messages.append(f"警告: 遊戲狀態字元歷史記錄結構異常.正在嘗試重設歷史記錄.")
        char_histories = []
        for char in target_line:
            if char in ALL_CHARACTERS_DATA:
                stroke_paths = ALL_CHARACTERS_DATA[char]
                char_histories.append({
                    'target_char':
                    char,
                    'stroke_histories': [{
                        'min_dist': float('inf'),
                        'best_guess_char': None,
                        'best_guess_stroke_index': None
                    } for _ in range(len(stroke_paths))]
                })
            else:
                messages.append(f"內部錯誤: 目標字 '{char}' 筆畫資料遺失, 無法重設歷史記錄.")
                partial_failure = True
                char_histories.append({
                    'target_char': char,
                    'stroke_histories': []
                })
        game_state['char_histories'] = char_histories

    for i in range(5):
        target_char = target_line[i]
        guess_char = guess_line[i]

        if i >= len(char_histories) or not isinstance(
                char_histories[i],
                dict) or char_histories[i].get('target_char') != target_char:
            messages.append(f"內部錯誤: 位置 {i+1} 的字元歷史記錄結構無效或與目標字元不匹配. 無法更新歷史記錄.")
            partial_failure = True
            continue

        char_history = char_histories[i]

        if target_char not in ALL_CHARACTERS_DATA:
            partial_failure = True
            continue

        target_char_stroke_count = len(ALL_CHARACTERS_DATA.get(
            target_char, []))
        if len(char_history.get('stroke_histories',
                                [])) != target_char_stroke_count:
            messages.append(
                f"警告: 位置 {i+1} 目標字 '{target_char}' 的筆畫歷史記錄長度異常.正在重設筆畫歷史記錄.")
            char_history['stroke_histories'] = [{
                'min_dist': float('inf'),
                'best_guess_char': None,
                'best_guess_stroke_index': None
            } for _ in range(target_char_stroke_count)]

        try:
            target_stroke_sequences_with_original_index = ALL_CHARACTERS_DTW_DATA.get(
                target_char, [])
            guess_stroke_sequences_with_original_index = ALL_CHARACTERS_DTW_DATA.get(
                guess_char, [])

            if not target_stroke_sequences_with_original_index or not all(
                    seq.shape[0] >= 2
                    for seq, _ in target_stroke_sequences_with_original_index
                    if seq is not None):
                partial_failure = True
                continue
            if not guess_stroke_sequences_with_original_index or not all(
                    seq.shape[0] >= 2
                    for seq, _ in guess_stroke_sequences_with_original_index
                    if seq is not None):
                partial_failure = True
                continue

            stroke_dtw_matrix = np.full(
                (len(guess_stroke_sequences_with_original_index),
                 len(target_stroke_sequences_with_original_index)), np.inf)

            for seq_g_idx, (seq_g, original_g_idx) in enumerate(
                    guess_stroke_sequences_with_original_index):
                if seq_g is None or seq_g.shape[0] < 2:
                    continue

                for seq_t_idx, (seq_t, original_t_idx) in enumerate(
                        target_stroke_sequences_with_original_index):
                    if seq_t is None or seq_t.shape[0] < 2:
                        stroke_dtw_matrix[seq_g_idx, seq_t_idx] = float('inf')
                        continue

                    try:
                        dtw_distance, path = fastdtw(seq_g,
                                                     seq_t,
                                                     dist=euclidean)
                        stroke_dtw_matrix[seq_g_idx, seq_t_idx] = dtw_distance
                    except Exception as dtw_e:
                        print(
                            f"Warning: FastDTW 計算失敗於位置 {i} (猜測 '{guess_char}' 筆畫 {original_g_idx+1} vs 目標 '{target_char}' 筆畫 {original_t_idx+1}): {dtw_e}"
                        )
                        traceback.print_exc()
                        stroke_dtw_matrix[seq_g_idx, seq_t_idx] = float('inf')

            for target_original_index in range(target_char_stroke_count):
                target_comp_index = next(
                    (idx for idx, (_, orig_idx) in enumerate(
                        target_stroke_sequences_with_original_index)
                     if orig_idx == target_original_index), None)

                if target_comp_index is not None and target_comp_index < stroke_dtw_matrix.shape[
                        1]:
                    col_distances = stroke_dtw_matrix[:, target_comp_index]

                    if col_distances.size > 0 and np.min(
                            col_distances) != np.inf:
                        min_dist_in_col = np.min(col_distances)
                        min_row_comp_index = np.argmin(col_distances)

                        best_matching_guess_original_index_for_this_target_stroke = None
                        if min_row_comp_index < len(
                                guess_stroke_sequences_with_original_index):
                            best_matching_guess_original_index_for_this_target_stroke = guess_stroke_sequences_with_original_index[
                                min_row_comp_index][1]

                        if target_original_index < len(
                                char_history.get('stroke_histories', [])):
                            hist_stroke_info = char_history[
                                'stroke_histories'][target_original_index]

                            if min_dist_in_col < hist_stroke_info.get(
                                    'min_dist', float('inf')):
                                hist_stroke_info['min_dist'] = min_dist_in_col
                                hist_stroke_info[
                                    'best_guess_char'] = guess_char
                                hist_stroke_info[
                                    'best_guess_stroke_index'] = best_matching_guess_original_index_for_this_target_stroke
                        else:
                            print(
                                f"Warning: Target stroke index {target_original_index} out of bounds for history list for char '{target_char}' at pos {i}. History length: {len(char_history.get('stroke_histories', []))}"
                            )
                            partial_failure = True
        except Exception as comp_e:
            print(
                f"錯誤: 處理位置 {i} 字元 '{target_char}' vs '{guess_char}' 的比較邏輯時發生錯誤: {comp_e}"
            )
            traceback.print_exc()
            messages.append(f"位置 {i+1} 字元比較失敗: {comp_e}")
            partial_failure = True

    is_correct_guess = (guess_line == target_line)
    is_game_over_by_guesses = current_guess_count >= 15
    is_game_over = is_correct_guess or is_game_over_by_guesses

    files = []
    view = None
    image_bytes = None

    if game_state and 'target_line' in game_state and 'char_histories' in game_state:
        try:
            image_bytes = plot_combined_character_image(
                game_state, is_game_over=is_game_over)
        except Exception as e:
            print(
                f"Error generating combined image for channel {channel_id}: {e}"
            )
            traceback.print_exc()
            partial_failure = True
            messages.append("圖片生成失敗.")

    if image_bytes:
        files.append(discord.File(io.BytesIO(image_bytes),
                                  filename='poem.png'))
    elif not partial_failure and (game_state and 'target_line' in game_state):
        messages.append("圖片生成失敗 (無數據).")

    response_parts = []
    response_parts.extend(messages)

    difficulty_display = get_difficulty_display(thresholds)

    if is_correct_guess:
        response_parts.append("恭喜你，猜對了整句詩!")
        poem_info = game_state.get('target_poem_info')
        if poem_info and isinstance(poem_info, dict):
            response_parts.append(
                f"這是出自 **《{poem_info.get('title', '未知詩名')}》**：")
            content = poem_info.get('content', [])
            if isinstance(content, list):
                formatted_content = []
                for line in content:
                    if line == target_line:
                        formatted_content.append(f"**{line}**")
                    else:
                        formatted_content.append(line)
                response_parts.extend(formatted_content)
            else:
                response_parts.append('詩詞內容無效.')

        if 'guess_count_history' not in game_state or not isinstance(
                game_state['guess_count_history'], list):
            game_state['guess_count_history'] = []
        game_state['guess_count_history'].append(current_guess_count)
        del games[channel_id]
        response_parts.append("\n使用 `/newpoem` 開始新遊戲，或使用 `/stats` 查看統計。")
        view = PoemGameView()

    elif is_game_over_by_guesses:
        response_parts.append(f"猜測次數過多 ({current_guess_count} 次), 本輪遊戲結束。")
        response_parts.append(f"正確詩句是: **{target_line}**")
        response_parts.append(generate_final_text_line(game_state))

        poem_info = game_state.get('target_poem_info')
        if poem_info and isinstance(poem_info, dict):
            response_parts.append(
                f"這是出自 **《{poem_info.get('title', '未知詩名')}》**：")
            content = poem_info.get('content', [])
            if isinstance(content, list):
                formatted_content = []
                for line in content:
                    if line == target_line:
                        formatted_content.append(f"**{line}**")
                    else:
                        formatted_content.append(line)
                response_parts.extend(formatted_content)
            else:
                response_parts.append('詩詞內容無效.')

        if 'guess_count_history' not in game_state or not isinstance(
                game_state['guess_count_history'], list):
            game_state['guess_count_history'] = []
        game_state['guess_count_history'].append(current_guess_count)
        del games[channel_id]
        response_parts.append("\n使用 `/newpoem` 開始新遊戲，或使用 `/stats` 查看統計。")
        view = PoemGameView()

    else:
        response_parts.append(
            f"第{current_guess_count}次，難度：{difficulty_display}")
        partial_reveal_parts = []
        padded_guess_line = (guess_line + '     ')[:5]
        for i in range(5):
            target_char = target_line[i]
            guess_char = padded_guess_line[i]

            if guess_char == target_char:
                partial_reveal_parts.append(target_char)
            else:
                partial_reveal_parts.append(":x:")

        response_parts.append("　".join(partial_reveal_parts))
        view = OngoingGameView()

    if partial_failure:
        response_parts.append("\n警告: 遊戲處理過程中發生部分錯誤，圖片或數據可能不完全準確。\n")

    return '\n'.join(response_parts), files, view


class GuessModal(ui.Modal, title='你的猜測'):
    guess_input = ui.TextInput(
        label='請輸入五個漢字',
        style=discord.TextStyle.short,
        placeholder='例如：花落知多少',
        required=True,
        min_length=5,
        max_length=5,
    )

    async def on_submit(self, interaction: discord.Interaction):
        guess_line = self.guess_input.value.strip()

        if len(guess_line) != 5:
            try:
                await interaction.response.send_message("無效的猜測。請輸入剛好五個有效的漢字。",
                                                        ephemeral=True)
            except discord.errors.InteractionResponded:
                print(
                    f"Warning: Modal submission interaction already responded to, couldn't send invalid length error."
                )
            return

        missing_chars = [
            char for char in guess_line if char not in ALL_CHARACTERS_DATA
        ]
        if missing_chars:
            try:
                await interaction.response.send_message(
                    f"你的猜測詩句中包含不合法的字元: {''.join(missing_chars)}. 請輸入常見漢字.",
                    ephemeral=True)
            except discord.errors.InteractionResponded:
                print(
                    f"Warning: Modal submission interaction already responded to, couldn't send invalid char error."
                )
            return

        try:
            await interaction.response.defer(ephemeral=True)
        except discord.errors.InteractionResponded:
            print(
                f"Warning: Modal submission interaction already responded to before deferring."
            )
            pass

        except Exception as e:
            print(f"Error during modal submission defer: {e}")
            traceback.print_exc()
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("處理中...",
                                                            ephemeral=True)
                else:
                    print(
                        "Warning: Cannot send placeholder message: Interaction already responded."
                    )
            except Exception as ee:
                print(
                    f"Error sending placeholder message after defer failure: {ee}"
                )

        response_text, files, view = process_guess(interaction.channel_id,
                                                   guess_line)

        try:
            if interaction.response.is_done():
                if view is not None:
                    message = await interaction.followup.send(response_text,
                                                              files=files,
                                                              view=view)
                    if isinstance(view, OngoingGameView) or isinstance(
                            view, PoemGameView):
                        view.message = message
                else:
                    await interaction.followup.send(response_text, files=files)
                print(
                    f"Channel {interaction.channel_id}: Followup sent from modal submission."
                )
            else:
                print(
                    f"Warning: Initial modal response/defer failed. Attempting to send regular message in channel {interaction.channel_id}."
                )
                channel = bot.get_channel(interaction.channel_id)
                if channel:
                    if view is not None:
                        message = await channel.send(response_text,
                                                     files=files,
                                                     view=view)
                        if isinstance(view, OngoingGameView) or isinstance(
                                view, PoemGameView):
                            view.message = message
                    else:
                        await channel.send(response_text, files=files)
                    print(
                        f"Channel {interaction.channel_id}: Sent fallback message directly to channel."
                    )
                else:
                    print(
                        f"Error: Could not get channel {interaction.channel_id} to send fallback message."
                    )

        except Exception as e:
            print(
                f"Error sending followup from modal submission or fallback message: {e}"
            )
            traceback.print_exc()
            try:
                if interaction.response.is_done():
                    await interaction.followup.send("處理你的猜測時發生錯誤，無法顯示結果。",
                                                    ephemeral=True)
                else:
                    print(
                        "Warning: Cannot send final ephemeral fallback: Interaction not done."
                    )
            except Exception as ee:
                print(
                    f"Error sending final ephemeral fallback message after followup failed: {ee}"
                )
                traceback.print_exc()

    async def on_error(self, interaction: discord.Interaction,
                       error: Exception) -> None:
        print(
            f"Error during modal submission for channel {interaction.channel_id}: {error}"
        )
        traceback.print_exc()
        try:
            if interaction.response.is_done():
                await interaction.followup.send('處理你的猜測時發生錯誤。', ephemeral=True)
            else:
                await interaction.response.send_message('處理你的猜測時發生錯誤。',
                                                        ephemeral=True)
        except Exception as e:
            print(f"Error sending error message in modal on_error: {e}")
            traceback.print_exc()


class PoemGameView(ui.View):

    def __init__(self):
        super().__init__(timeout=180)
        self.message = None

    @ui.button(label="再來一局", style=discord.ButtonStyle.primary)
    async def new_game_button(self, interaction: discord.Interaction,
                              button: ui.Button):
        for item in self.children:
            if isinstance(item, ui.Button):
                item.disabled = True
        try:
            await interaction.response.edit_message(view=self)
            print(
                f"Channel {interaction.channel_id}: Disabled button and attempted to edit original message for new_game_button click."
            )

        except (discord.errors.NotFound, discord.errors.InteractionResponded,
                discord.errors.Forbidden) as e:
            print(
                f"Warning: Cannot edit original message with disabled button on button click ({e.__class__.__name__}): {e}"
            )
            pass

        except Exception as e:
            print(
                f"Error during initial edit_message response on button click: {e}"
            )
            traceback.print_exc()
            pass

        if GAME_LOAD_ERROR:
            try:
                await interaction.followup.send(
                    f"遊戲資料載入失敗，無法開始新遊戲: {GAME_LOAD_ERROR}", ephemeral=True)
            except Exception as e:
                print(
                    f"Error sending followup message after button click (GAME_LOAD_ERROR): {e}"
                )
                traceback.print_exc()
            return

        if interaction.channel is None:
            try:
                await interaction.followup.send("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
            except Exception as e:
                print(
                    f"Error sending followup message after new game button click (channel is None): {e}"
                )
                traceback.print_exc()
            return

        channel_id = interaction.channel_id
        print(f"頻道 {channel_id}: 收到按鈕點擊開始新遊戲指令.")

        old_history: typing.List[int] = games.get(channel_id, {}).get(
            'guess_count_history', [])
        temp_history = old_history
        if channel_id in games:
            del games[channel_id]

        game_state, init_error, current_source_used = get_or_initialize_game_state(
            channel_id, preferred_source=DEFAULT_POEMS_SOURCE, force_new=True)

        if game_state:
            if 'guess_count_history' not in game_state or not isinstance(
                    game_state['guess_count_history'], list):
                game_state['guess_count_history'] = []
            game_state['guess_count_history'] = temp_history

        if init_error or game_state is None:
            try:
                await interaction.followup.send(
                    f"無法初始化新遊戲: {init_error or '狀態無效'}")
            except Exception as e:
                print(
                    f"Error sending followup message after button click (init_error): {e}"
                )
                traceback.print_exc()
            if channel_id in games:
                del games[channel_id]
            return

        messages: typing.List[str] = [
            f"新的猜詩遊戲開始了！ (題庫: {POEMS_SOURCES.get(current_source_used, current_source_used)})",
            f"這是一句五言詩。請使用 `/guess [你的猜測]` 或直接輸入五個漢字來猜測。"
        ]
        messages.append(generate_initial_text_line(game_state))

        new_game_view = OngoingGameView()

        try:
            message = await interaction.followup.send('\n'.join(messages),
                                                      view=new_game_view)
            new_game_view.message = message
            print(
                f"Channel {interaction.channel_id}: Final followup message sent for new_game_button."
            )
        except Exception as e:
            print(
                f"Error sending final followup message after new_game_button click: {e}"
            )
            traceback.print_exc()

    async def on_timeout(self):
        try:
            if hasattr(self, 'message') and self.message:
                for item in self.children:
                    if isinstance(item, ui.Button):
                        item.disabled = True
                await self.message.edit(view=self)
            else:
                pass
        except (discord.errors.NotFound, discord.errors.Forbidden) as e:
            print(
                f"Timeout: Error editing message ({e.__class__.__name__}). Message might be deleted or permissions missing."
            )
        except Exception as e:
            print(f"Timeout: Unexpected error editing message: {e}")
            traceback.print_exc()


class OngoingGameView(ui.View):

    def __init__(self):
        super().__init__(timeout=180)
        self.message = None

    @ui.button(label="猜測", style=discord.ButtonStyle.green)
    async def guess_button(self, interaction: discord.Interaction,
                           button: ui.Button):
        try:
            await interaction.response.send_modal(GuessModal())
            print(
                f"Channel {interaction.channel_id}: Sent GuessModal in response to button click."
            )
        except Exception as e:
            print(f"Error sending modal after guess button click: {e}")
            traceback.print_exc()
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(
                        "無法開啟猜測介面，請直接使用 `/guess [你的猜測]` 指令。", ephemeral=True)
                else:
                    print(
                        f"Channel {interaction.channel_id}: Could not send fallback message for failed modal: Interaction already responded."
                    )

            except Exception as ee:
                print(
                    f"Error sending fallback message after modal failed: {ee}")
                traceback.print_exc()

    async def on_timeout(self):
        try:
            for item in self.children:
                if isinstance(item, ui.Button):
                    item.disabled = True
            if hasattr(self, 'message') and self.message:
                await self.message.edit(view=self)
            else:
                print(
                    f"Timeout: OngoingGameView timed out for channel {self.message.channel.id if hasattr(self, 'message') and self.message and self.message.channel else 'Unknown'}. No message reference was stored or message deleted. Cannot disable button visually."
                )

        except (discord.errors.NotFound, discord.errors.Forbidden) as e:
            print(
                f"Timeout: Error editing OngoingGameView message ({e.__class__.__name__}) for channel {self.message.channel.id if hasattr(self, 'message') and self.message and self.message.channel else 'Unknown'}. Message might be deleted or permissions missing."
            )
        except Exception as e:
            print(
                f"Timeout: Unexpected error editing OngoingGameView message: {e}"
            )
            traceback.print_exc()


bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())
tree = bot.tree


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
        total_poem_lines = sum(
            len(VALID_POEM_LINES_MAP.get(source, []))
            for source in POEMS_SOURCES)
        print(
            f'Loaded data for {len(ALL_CHARACTERS_DATA)} characters and {total_poem_lines} poem lines across {len(POEMS_SOURCES)} sources.'
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


@tree.command(name='guess', description='猜測當前詩句。請輸入五個漢字。例如: /guess 花落知多少')
@app_commands.describe(guess_line="你的五字猜測")
async def guess(interaction: discord.Interaction, guess_line: str):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法進行猜測: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Attempting to defer interaction for /guess."
    )
    try:
        await interaction.response.defer()
        print(
            f"Channel {interaction.channel_id}: Interaction for /guess successfully deferred."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer interaction in guess for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Interaction defer failed for /guess. Aborting command processing for this interaction."
        )
        return

    guess_line = guess_line.strip()

    if len(guess_line) != 5:
        try:
            await interaction.followup.send(
                f"請輸入剛好五個漢字進行猜測 (你輸入了 {len(guess_line)} 個字).")
            print(
                f"Channel {interaction.channel_id}: Followup sent for /guess (length mismatch)."
            )
        except Exception as fe:
            print(
                f"Error sending followup after guess length check fail for /guess: {fe}"
            )
            traceback.print_exc()
        return

    missing_chars = [
        char for char in guess_line if char not in ALL_CHARACTERS_DATA
    ]
    if missing_chars:
        try:
            await interaction.followup.send(
                f"你的猜測詩句中包含不合法的字元: {''.join(missing_chars)}. 請輸入常見漢字.")
            print(
                f"Channel {interaction.channel_id}: Followup sent for /guess (invalid chars)."
            )
        except Exception as fe:
            print(
                f"Error sending followup after guess char check fail for /guess: {fe}"
            )
            traceback.print_exc()
        return

    response_text, files, view = process_guess(interaction.channel_id,
                                               guess_line)

    try:
        if view is not None:
            message = await interaction.followup.send(response_text,
                                                      files=files,
                                                      view=view)
            if isinstance(view, OngoingGameView) or isinstance(
                    view, PoemGameView):
                view.message = message
        else:
            await interaction.followup.send(response_text, files=files)
        print(
            f"Channel {interaction.channel_id}: Final followup message sent for /guess."
        )
    except Exception as fe:
        print(f"Error sending final followup message in guess: {fe}")
        traceback.print_exc()


@tree.command(name='newpoem', description=f'開始一個新的詩詞遊戲。可以指定題庫。')
@app_commands.describe(
    source=
    f'選擇題庫來源: {"、".join([f"{POEMS_SOURCES[k]} ({k})" for k in POEMS_SOURCES] + [f"數字別名({k})" for k in POEMS_SOURCE_MAP if k not in POEMS_SOURCES and POEMS_SOURCE_MAP[k] in POEMS_SOURCES])} (預設: {POEMS_SOURCES[DEFAULT_POEMS_SOURCE]})'
)
async def newpoem(interaction: discord.Interaction,
                  source: str = DEFAULT_POEMS_SOURCE):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法開始新遊戲: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    resolved_source_key = POEMS_SOURCE_MAP.get(source, source)

    if resolved_source_key not in POEMS_SOURCES:
        available_sources_display = "、".join(
            [f"{POEMS_SOURCES[k]} ({k})" for k in POEMS_SOURCES] + [
                f"數字別名({k})" for k in POEMS_SOURCE_MAP if
                k not in POEMS_SOURCES and POEMS_SOURCE_MAP[k] in POEMS_SOURCES
            ])
        await interaction.response.send_message(
            f"無效的題庫來源 '{source}'. 可用的題庫: {available_sources_display}. 使用預設題庫 `{DEFAULT_POEMS_SOURCE}`.",
            ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Checks passed for /newpoem with source '{source}'. Attempting to defer interaction."
    )
    try:
        await interaction.response.defer()
        print(
            f"Channel {interaction.channel_id}: Interaction for /newpoem successfully deferred."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer interaction in newpoem for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Interaction defer failed for /newpoem. Aborting command processing for this interaction."
        )
        return

    channel_id = interaction.channel_id
    print(f"頻道 {channel_id}: 收到開始新遊戲指令，來源: {resolved_source_key}")

    old_history: typing.List[int] = games.get(channel_id,
                                              {}).get('guess_count_history',
                                                      [])
    temp_history = old_history
    if channel_id in games:
        del games[channel_id]

    game_state, init_error, current_source_used = get_or_initialize_game_state(
        channel_id, preferred_source=resolved_source_key, force_new=True)

    if game_state:
        if 'guess_count_history' not in game_state or not isinstance(
                game_state['guess_count_history'], list):
            game_state['guess_count_history'] = []
        game_state['guess_count_history'] = temp_history

    if init_error or game_state is None:
        try:
            await interaction.followup.send(f"無法初始化新遊戲: {init_error or '狀態無效'}"
                                            )
            print(
                f"Channel {interaction.channel_id}: Followup sent for /newpoem (init error)."
            )
        except Exception as fe:
            print(f"Error sending followup after newpoem init fail: {fe}")
            traceback.print_exc()
        if channel_id in games:
            del games[channel_id]
        return

    messages: typing.List[str] = [
        f"新的猜詩遊戲開始了！ (題庫: {POEMS_SOURCES.get(current_source_used, current_source_used)})",
        f"這是一句五言詩。請使用 `/guess [你的猜測]` 或直接輸入五個漢字來猜測。"
    ]
    messages.append(generate_initial_text_line(game_state))

    view = OngoingGameView()

    try:
        message = await interaction.followup.send('\n'.join(messages),
                                                  view=view)
        view.message = message
        print(
            f"Channel {interaction.channel_id}: Final followup message sent for /newpoem."
        )
    except Exception as fe:
        print(f"Error sending final followup message in newpoem: {fe}")
        traceback.print_exc()


@tree.command(name='stats', description='顯示你的猜測次數統計圖')
async def stats(interaction: discord.Interaction):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法顯示統計: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Attempting to defer interaction for /stats."
    )
    try:
        await interaction.response.defer()
        print(
            f"Channel {interaction.channel_id}: Interaction for /stats successfully deferred."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer interaction in stats for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Interaction defer failed for /stats. Aborting command processing for this interaction."
        )
        return

    game_state = games.get(interaction.channel_id)
    guess_count_history = game_state.get(
        'guess_count_history', []) if game_state and isinstance(
            game_state.get('guess_count_history'), list) else []

    if len(guess_count_history) < 3:
        try:
            await interaction.followup.send(
                "目前完成的詩詞數量不足 (少於 3 首)，無法生成統計圖表。完成更多詩詞後再試試！")
            print(
                f"Channel {interaction.channel_id}: Followup sent for /stats (not enough history)."
            )
        except Exception as fe:
            print(
                f"Error sending followup after stats not enough history: {fe}")
            traceback.print_exc()
        return

    try:
        image_bytes = generate_stats_plot_buffer(guess_count_history)
        if image_bytes:
            try:
                await interaction.followup.send("你的猜測次數統計圖：",
                                                file=discord.File(
                                                    io.BytesIO(image_bytes),
                                                    filename='stats.png'))
                print(
                    f"Channel {interaction.channel_id}: Followup sent for /stats (with image)."
                )
            except Exception as fe:
                print(f"Error sending followup with stats image: {fe}")
                traceback.print_exc()
        else:
            try:
                await interaction.followup.send("無法生成統計圖表。")
                print(
                    f"Channel {interaction.channel_id}: Followup sent for /stats (plot fail)."
                )
            except Exception as fe:
                print(f"Error sending followup after stats plot fail: {fe}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error during stats plot generation or sending followup: {e}")
        traceback.print_exc()
        try:
            await interaction.followup.send("生成統計圖表時發生錯誤。")
        except Exception as fe:
            print(f"Error sending followup after stats generate error: {fe}")
            traceback.print_exc()


@tree.command(name='difficulty', description='設定遊戲難度 (1-6)，1最難')
@app_commands.describe(level=f'難度等級 (1-6)')
async def difficulty(interaction: discord.Interaction, level: int):
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法設定難度: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Attempting to defer ephemeral interaction for /difficulty."
    )
    try:
        await interaction.response.defer(ephemeral=True)
        print(
            f"Channel {interaction.channel_id}: Interaction for /difficulty successfully deferred ephemeral."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer ephemeral interaction in difficulty for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Ephemeral Interaction defer failed for /difficulty. Aborting command processing for this interaction."
        )
        return

    if level not in DIFFICULTY_THRESHOLDS:
        try:
            await interaction.followup.send(f"無效的難度等級。請輸入 1 到 6 之間的數字。",
                                            ephemeral=True)
            print(
                f"Channel {interaction.channel_id}: Followup sent for /difficulty (invalid level)."
            )
        except Exception as fe:
            print(
                f"Error sending followup after difficulty invalid level: {fe}")
            traceback.print_exc()
        return

    channel_id = interaction.channel_id
    game_state = games.get(channel_id)

    thresh1, thresh2 = DIFFICULTY_THRESHOLDS[level]
    new_thresholds = {'thresh1': float(thresh1), 'thresh2': float(thresh2)}

    channel_preferred_thresholds[channel_id] = new_thresholds
    print(
        f"Channel {channel_id}: Updated preferred thresholds to {new_thresholds}."
    )

    if game_state:
        game_state['thresholds'] = new_thresholds
        print(f"Channel {channel_id}: Updated current game thresholds.")
    else:
        print(
            f"Channel {channel_id}: No active game to update thresholds for.")

    try:
        await interaction.followup.send(
            f"已將遊戲難度設定為等級 {level} (閾值: 黑 < {thresh1}, 灰 < {thresh2}). 這將應用於下一輪或當前遊戲 (如果存在).",
            ephemeral=True)
        print(
            f"Channel {interaction.channel_id}: Final ephemeral followup message sent for /difficulty."
        )
    except Exception as fe:
        print(
            f"Error sending final ephemeral followup message in difficulty: {fe}"
        )
        traceback.print_exc()


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

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Attempting to defer ephemeral interaction for /set_thresholds."
    )
    try:
        await interaction.response.defer(ephemeral=True)
        print(
            f"Channel {interaction.channel_id}: Interaction for /set_thresholds successfully deferred ephemeral."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer ephemeral interaction in set_thresholds for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Ephemeral Interaction defer failed for /set_thresholds. Aborting command processing for this interaction."
        )
        return

    if thresh1 < 0 or thresh2 < 0 or thresh1 > thresh2:
        try:
            await interaction.followup.send(
                f"警告: 建議閾值應非負且黑閾值不高於灰閾值 (0 ≤ 黑閾值 ≤ 灰閾值)。您輸入的值為 黑={thresh1}, 灰={thresh2}.",
                ephemeral=True)
        except Exception as fe:
            print(
                f"Error sending followup warning for set_thresholds invalid values: {fe}"
            )
            traceback.print_exc()

    channel_id = interaction.channel_id
    game_state = games.get(channel_id)

    new_thresholds = {'thresh1': thresh1, 'thresh2': thresh2}

    channel_preferred_thresholds[channel_id] = new_thresholds
    print(
        f"Channel {channel_id}: Updated preferred thresholds to {new_thresholds}."
    )

    if game_state:
        game_state['thresholds'] = new_thresholds
        print(f"Channel {channel_id}: Updated current game thresholds.")
    else:
        print(
            f"Channel {channel_id}: No active game to update thresholds for.")

    try:
        await interaction.followup.send(
            f"已將閾值設定為：黑 < {thresh1}, 灰 < {thresh2}. 這將應用於下一輪或當前遊戲 (如果存在).",
            ephemeral=True)
        print(
            f"Channel {interaction.channel_id}: Final ephemeral followup message sent for /set_thresholds."
        )
    except Exception as fe:
        print(
            f"Error sending final ephemeral followup message in set_thresholds: {fe}"
        )
        traceback.print_exc()


@set_thresholds_cmd.error
async def set_thresholds_cmd_error(interaction: discord.Interaction,
                                   error: app_commands.AppCommandError):
    try:
        if interaction.response.is_done():
            print(
                f"Warning: Interaction already responded to in set_thresholds_cmd_error."
            )
            try:
                await interaction.followup.send(f"設定閾值時發生錯誤: {error}",
                                                ephemeral=True)
                print(
                    f"Channel {interaction.channel_id}: Sent ephemeral followup for set_thresholds_cmd_error (after response done)."
                )
            except discord.errors.InteractionResponded:
                print(
                    f"Warning: Interaction already fully responded to. Cannot send followup."
                )
            except Exception as fe:
                print(
                    f"Error sending followup in set_thresholds_cmd_error after response done: {fe}"
                )
                traceback.print_exc()
            return

        print(
            f"Channel {interaction.channel_id}: Handling set_thresholds command error."
        )
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("你沒有權限設定閾值。",
                                                    ephemeral=True)
            print(
                f"Channel {interaction.channel_id}: Sent ephemeral error for MissingPermissions."
            )
        elif isinstance(error, app_commands.BadArgument):
            await interaction.response.send_message(
                "無效的閾值格式。請輸入兩個數字。例如: `/set_thresholds 10000 25000`",
                ephemeral=True)
            print(
                f"Channel {interaction.channel_id}: Sent ephemeral error for BadArgument."
            )
        else:
            print(f"Unexpected error in set_thresholds_cmd: {error}")
            traceback.print_exc()
            await interaction.response.send_message("設定閾值時發生未知錯誤。",
                                                    ephemeral=True)
            print(
                f"Channel {interaction.channel_id}: Sent ephemeral error for unexpected error."
            )
    except Exception as e:
        print(f"Error responding in set_thresholds_cmd_error: {e}")
        traceback.print_exc()


@tree.command(name='giveup', description='放棄當前遊戲，顯示答案和原詩句')
async def giveup(interaction: discord.Interaction):
    global games, channel_last_used_source
    if GAME_LOAD_ERROR:
        await interaction.response.send_message(
            f"遊戲資料載入失敗，無法放棄遊戲: {GAME_LOAD_ERROR}", ephemeral=True)
        return

    if interaction.channel is None:
        await interaction.response.send_message("此指令只能在伺服器頻道中使用。",
                                                ephemeral=True)
        return

    print(
        f"Channel {interaction.channel_id}: Attempting to defer interaction for /giveup."
    )
    try:
        await interaction.response.defer()
        print(
            f"Channel {interaction.channel_id}: Interaction for /giveup successfully deferred."
        )
    except discord.errors.HTTPException as e:
        print(
            f"Failed to defer interaction in giveup for channel {interaction.channel_id}: {e}"
        )
        traceback.print_exc()
        print(
            f"Channel {interaction.channel_id}: Interaction defer failed for /giveup. Aborting command processing for this interaction."
        )
        return

    channel_id = interaction.channel_id
    game_state = games.get(channel_id)

    if game_state is None:
        try:
            await interaction.followup.send("此頻道沒有正在進行的遊戲。")
            print(
                f"Channel {channel_id}: Sent followup for /giveup (no game).")
        except Exception as fe:
            print(f"Error sending followup after giveup no game: {fe}")
            traceback.print_exc()
        return

    print(f"Channel {channel_id}: User used /giveup.")

    target_line = game_state['target_line']
    poem_info = game_state.get('target_poem_info')
    guess_count = game_state.get('guess_count', 0)

    if 'guess_count_history' not in game_state or not isinstance(
            game_state['guess_count_history'], list):
        game_state['guess_count_history'] = []
    game_state['guess_count_history'].append(guess_count)

    messages = [
        f"蛤！你這樣就放棄了？", f"答案: **{target_line}**", f"(猜測次數: {guess_count} 次)",
        generate_final_text_line(game_state)
    ]

    if poem_info and isinstance(poem_info, dict):
        messages.append(f"這是出自 **《{poem_info.get('title', '未知詩名')}》**：")
        content = poem_info.get('content', [])
        if isinstance(content, list):
            formatted_content = []
            for line in content:
                if line == target_line:
                    formatted_content.append(f"**{line}**")
                else:
                    formatted_content.append(line)
            messages.extend(formatted_content)
        else:
            messages.append('詩詞內容無效.')

    files = []
    image_bytes = None
    try:
        image_bytes = plot_combined_character_image(game_state,
                                                    is_game_over=True)
        if image_bytes:
            files.append(
                discord.File(io.BytesIO(image_bytes), filename='poem.png'))
        else:
            messages.append("圖片生成失敗 (無數據).")

    except Exception as e:
        print(
            f"Error generating combined image for giveup in channel {channel_id}: {e}"
        )
        traceback.print_exc()
        messages.append("圖片生成失敗.")
    if game_state and 'current_poem_source' in game_state:
        channel_last_used_source[channel_id] = game_state[
            'current_poem_source']

    del games[channel_id]

    view = PoemGameView()

    try:
        message = await interaction.followup.send('\n'.join(messages),
                                                  files=files,
                                                  view=view)
        view.message = message
        print(f"Channel {channel_id}: Sent followup for /giveup.")
    except Exception as fe:
        print(f"Error sending final followup message in giveup: {fe}")
        traceback.print_exc()


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    if message.content.startswith(bot.command_prefix):
        return

    if message.channel is None:
        return

    guess_line = message.content.strip()

    if len(guess_line) == 5:
        missing_chars = [
            char for char in guess_line if char not in ALL_CHARACTERS_DATA
        ]
        if missing_chars:
            return

        channel_id = message.channel.id
        print(f"頻道 {channel_id}: 收到非指令猜測訊息: '{guess_line}'")

        if GAME_LOAD_ERROR:
            try:
                await message.channel.send(
                    f"遊戲資料載入失敗，無法進行猜測: {GAME_LOAD_ERROR}")
            except Exception as e:
                print(f"Error sending error message in on_message: {e}")
            return

        response_text, files, view = process_guess(channel_id, guess_line)

        try:
            if view is not None:
                message_response = await message.channel.send(response_text,
                                                              files=files,
                                                              view=view)
                if isinstance(view, OngoingGameView) or isinstance(
                        view, PoemGameView):
                    view.message = message_response
            else:
                await message.channel.send(response_text, files=files)
            print(f"Channel {channel_id}: Sent response for on_message guess.")
        except discord.errors.Forbidden:
            print(
                f"Error: Missing permissions to send message in channel {channel_id}"
            )
        except Exception as e:
            print(
                f"Error sending message in on_message for channel {channel_id}: {e}"
            )
            traceback.print_exc()


BOT_TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
if not BOT_TOKEN:
    print("錯誤: 未找到環境變數 'DISCORD_BOT_TOKEN'. 請在 Replit Secrets 或環境變數中設定.")
else:
    if GAME_LOAD_ERROR:
        print("由於資料載入錯誤，機器人將不啟動.")
    else:
        try:
            bot.run(BOT_TOKEN)
        except Exception as e:
            print(f"運行 Discord 機器人時發生錯誤: {e}")
            traceback.print_exc()
