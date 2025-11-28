from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Protocol, Sequence, cast

import pypdfium2 as pdfium
from google import genai
from google.genai import types as genai_types
from PIL import Image

DEFAULT_PRESET_PROMPT = (
    "あなたはグラフリテラシーに長けたアナリストです。"
    " 以下のフォーマットで簡潔にコメントを返してください:"
    " 1) トレンド 2) 読み取れる含意 3) 注意点。"
)

OUTPUT_FORMAT_INSTRUCTION = dedent(
    """
    以下のJSON配列形式で出力してください。

    ```json
    [
      {"graph_name": "...", "item_name": "...", "comment": "..."},
      {"graph_name": "...", "item_name": "...", "comment": "..."}
    ]
    ```

    ■ グラフタイプ別の出力ルール【厳守】:

    【T1+T2: 月別電力推移グラフ(折れ線+棒)】→ 必ず2つのオブジェクトを出力
    ```json
    [
      {"graph_name": "直近1年間の電力使用状況", "item_name": "最大電力[kW]", "comment": "..."},
      {"graph_name": "直近1年間の電力使用状況", "item_name": "電力使用量[kWh]", "comment": "..."}
    ]
    ```

    【T3: 月間電力使用量の内訳(ドーナツ+表)】→ 必ず2つのオブジェクトを出力
    ```json
    [
      {"graph_name": "月間電力使用量の内訳", "item_name": "回路別内訳",
       "comment": "(当月の構成比のみ)"},
      {"graph_name": "月間電力使用量の内訳", "item_name": "前年比較",
       "comment": "(前年同月との差分)"}
    ]
    ```
    ※「回路別内訳」には前年比較を含めない。「前年比較」は別オブジェクトとして出力。

    【T4: 日別の回路別電力使用量】→ 1つのオブジェクトを出力
    ```json
    [
      {"graph_name": "日別の回路別電力使用量...", "item_name": "", "comment": "..."}
    ]
    ```

    【T5: 最大デマンド関連】→ 1〜2つのオブジェクトを出力
    ```json
    [
      {"graph_name": "今月の最大電力内訳", "item_name": "最大デマンド発生時内訳", "comment": "..."}
    ]
    ```

    ■ 各項目のコメント内容:

    「回路別内訳」: 当月の回路別構成比を説明
    - 例: 上位3回路「1F事務所SR_電灯」(22.6%)、「3F事務所_電灯」(22.2%)...で全体の57.9%
    - 前年との比較は含めない

    「前年比較」: 前年同月との差分・増減を説明
    - 例: 計測回路合計は10,588kWhで前年同月比+468kWh(+4.1%)増加
    - どの回路が増減したかを具体的に記載

    ■ 気温データとの相関【重要】:
    【気温データ】が提供されている場合は、必ず以下のように気温と電力消費の関係をコメントに含める:
    - 最大電力[kW]: 気温が高い/低い日にピークが発生した可能性を言及
    - 電力使用量[kWh]: 気温差(前年比○℃)による冷暖房負荷の増減を言及
    - 前年比較: 気温差が電力使用量の増減に与えた影響を必ず記載
      例: 「前年同月比で最高気温+3.6℃、平均気温+2.1℃となった気温差による冷房設備の
           負荷増大が電力使用量増加の主要因である可能性があります」
    - 日別推移: 特定日の気温と電力ピークの相関を言及

    ■ 前年比較の計算:
    - 【前年同月データ】の数値を使って前年比を計算
    - 【気温データ】があれば気温差との相関を必ずコメントに含める
    """
)

PRESET_PROMPT = dedent(
    """
    あなたは電力管理の上級アナリストです。入力は【グラフ画像】【グラフ画像の元データ】です。
    任意で【日別気温CSV】です。

    画像内の数値・凡例・表・注記だけを根拠として、各グラフごとに
    1. 可能な限り前年同月データを参照し、前年比較をして
    2. 事実(具体的な数値)、短い仮説、一言アクションを内容に含め
    1〜3文で簡潔に記述してください。

    (計測カバー率に対するコメントは不要)

    右側や下部の説明文は読まない。推測で数値を作らない。読めない値は「不明」とする。

    ========================
    ■ グラフタイプの自動判定(複数該当可)
    T1: 月別 最大電力[kW](折れ線/マーカー)
    T2: 月別 電力量[kWh](棒)
    T3: 回路別内訳(上位5+その他+未計測; ドーナツ/表)
    T4: 日別 電力量の積み上げ+日別 最大電力(折れ線)
    T5: 最大デマンド発生時(30分区間・回路別kW/%)

    ■ 共通X軸(上段目盛なし→下段と共有)の扱い(必須)
    - 同一パネル内で上下(または左右)に並び、枠幅/縦グリッド/点数が一致する場合は
      **X軸共有**とみなす。
    - **上段に目盛が無い場合**は、**下段のX軸ラベルを左→右で上段に対応付け**
      て解釈する。
    - X軸ラベルが「期間(例:(8月〜5月))」など**集約カテゴリ**の場合は、その区間を
      **月次比較(先月比/前年比)の計算対象から除外**し、文章内で「比較不可(集約)」
      と明記。
    - 共有が成立しない/判定不明は「X軸共有: 不明」とし、上段は数値ラベルのみで記述。

    ■ 数値抽出と表記
    - 優先度: データラベル>注記/表>目盛近似(±1目盛)
    - 単位は必ず付与(kW/kWh/%)。丸め: kW=小数1桁、kWh=整数(3桁区切り)、%=小数1桁。
    - 合計/比率の整合は±2%を許容。超えるときは「整合に注意」と一言添える。

    ■ 比較と計算(読めるときのみ)
    - 前年同月比[%]=(今月-前年同月)/前年同月x100
    - 先月比[%]=(今月-先月)/先月x100
    - 上位3/5回路の合計比率[%]、計測カバー率[%]=計測回路合計/受電電力量x100
    - 集約カテゴリは**比較に使わない**(「比較不可(集約)」と記載)。

    ========================
    ■ 出力(Markdown。各グラフ=見出し+1〜3文。長文禁止)
    - 見出しは画像内タイトルをそのまま使う。無ければ「グラフ1」「グラフ2」等。
    - 文章構成: 事実(数値)→短い仮説→一言アクションの3要素を含めること。
    - 文体: **各要素を接続詞で滑らかに繋ぎ、自然な日本語の文章にする**こと。
      「(事実)」「(仮説)」といったラベル表記は禁止。
    - 共通X軸を用いた場合、先頭文の末尾に(上段は下段と同一X軸)と添える。
    - 例: 「比較不可(集約)」「読取不可」「整合に注意」などは文脈に組み込むか、
      括弧で短く注記。

    【出力フォーマット例】
    ### 例タイトル(画像内タイトル)
    - 例: 最大電力[kW](上段は下段と同一X軸)
      今月は **86.9 kW** で前年7月 **90.9 kW** を下回り、12か月内のピーク
      **100.5 kW**(集約区間)より低水準で推移しました。夏期ピークの同時起動が抑えられた
      可能性があるため、引き続きピーク帯の段階投入と需要監視アラート90%の設定を推奨します。
    """
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
CSV_EXTENSIONS = {".csv"}
DEFAULT_MODEL_NAME = "gemini-3-pro-preview"


@dataclass
class MonthlyReportData:
    """前年同月の月報データ"""

    month_label: str  # 例: "2023年10月"
    max_power_daily: list[float] = field(default_factory=list)  # 日別最大電力[kW]
    circuits: dict[str, list[float]] = field(default_factory=dict)  # 回路名 -> 日別値[kWh]
    total_power_daily: list[float] = field(default_factory=list)  # 日別受電電力[kWh]

    @property
    def max_power_monthly(self) -> float:
        """月間最大電力 = 日別最大値の最大"""
        if not self.max_power_daily:
            return 0.0
        return max(self.max_power_daily)

    @property
    def total_power_monthly(self) -> float:
        """月間電力使用量 = 日別合計のSUM"""
        return sum(self.total_power_daily)

    def circuit_monthly_total(self, circuit_name: str) -> float:
        """回路別月間合計"""
        return sum(self.circuits.get(circuit_name, []))


@dataclass
class MonthlyTemperatureSummary:
    """月別気温サマリー"""

    year_month: str  # 例: "2023-10", "2024-10"
    max_temp: float  # 月間最高気温
    min_temp: float  # 月間最低気温
    avg_temp: float  # 月間平均気温


def parse_monthly_report_csv(path: Path) -> MonthlyReportData:
    """
    月報CSVをパースして構造化データに変換。

    CSVフォーマット:
    - 行1: ヘッダー説明
    - 行2: 日付列 (10/1(日), 10/2(月), ...)
    - 行3: 最大電力[kW]
    - 行4以降: 回路名と日別値
    - 最終行: 受電電力
    """
    rows = _read_csv_rows(path)
    if len(rows) < 3:
        msg = f"月報CSV {path.name} のフォーマットが不正です(行数不足)"
        raise ValueError(msg)

    # ファイル名から年月を抽出 (例: 月報202310.csv -> 2023年10月)
    match = re.search(r"(\d{4})(\d{2})", path.name)
    if match:
        year, month = match.groups()
        month_label = f"{year}年{int(month)}月"
    else:
        month_label = "不明"

    max_power_daily: list[float] = []
    circuits: dict[str, list[float]] = defaultdict(list)
    total_power_daily: list[float] = []

    for row in rows[2:]:  # 行3以降をパース
        if not row or not row[0].strip():
            continue

        row_name = row[0].strip()
        values: list[float] = []
        for cell in row[1:]:
            try:
                values.append(float(cell))
            except ValueError:
                continue

        if not values:
            continue

        if row_name == "最大電力[kW]":
            max_power_daily = values
        elif row_name == "受電電力":
            total_power_daily = values
        else:
            circuits[row_name] = values

    return MonthlyReportData(
        month_label=month_label,
        max_power_daily=max_power_daily,
        circuits=dict(circuits),
        total_power_daily=total_power_daily,
    )


def parse_temperature_csv_for_comparison(
    path: Path,
) -> tuple[MonthlyTemperatureSummary, MonthlyTemperatureSummary]:
    """
    気温CSVをパースして前年・当年の月別サマリーを返す。

    CSVフォーマット:
    - 時間別データ (年月日時刻, 気温, ...)
    - 前年と当年のデータが含まれる
    """
    rows = _read_csv_rows(path)

    # 年月ごとに気温を集計
    temps_by_year_month: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if len(row) < 2:
            continue

        date_str = row[0].strip()
        # 日付パターンを検出 (2023/10/1 or 2023-10-01)
        date_match = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", date_str)
        if not date_match:
            continue

        year, month, _ = date_match.groups()
        year_month = f"{year}-{int(month):02d}"

        try:
            temp = float(row[1])
            temps_by_year_month[year_month].append(temp)
        except ValueError:
            continue

    if len(temps_by_year_month) < 2:
        msg = f"気温CSV {path.name} に前年・当年のデータが見つかりません"
        raise ValueError(msg)

    # 同じ月で前年・当年を比較
    # データ件数が最も多い年月をメインとし、その1年前/後を探す
    sorted_by_count = sorted(
        temps_by_year_month.keys(),
        key=lambda ym: len(temps_by_year_month[ym]),
        reverse=True,
    )

    # 最もデータが多い月を基準にする
    main_month = sorted_by_count[0]
    main_year, main_mm = main_month.split("-")
    main_year_int = int(main_year)

    # 同じ月の前年・当年を探す
    prev_year_month = f"{main_year_int - 1}-{main_mm}"
    curr_year_month = f"{main_year_int}-{main_mm}"

    # 前年データがない場合は逆(当年がメインで翌年を探す)
    if prev_year_month not in temps_by_year_month:
        next_year_month = f"{main_year_int + 1}-{main_mm}"
        if next_year_month in temps_by_year_month:
            prev_year_month = main_month
            curr_year_month = next_year_month
        else:
            msg = f"気温CSV {path.name} に同じ月の前年・当年データが見つかりません"
            raise ValueError(msg)

    prev_temps = temps_by_year_month[prev_year_month]
    curr_temps = temps_by_year_month[curr_year_month]

    prev_summary = MonthlyTemperatureSummary(
        year_month=prev_year_month,
        max_temp=max(prev_temps),
        min_temp=min(prev_temps),
        avg_temp=sum(prev_temps) / len(prev_temps),
    )

    curr_summary = MonthlyTemperatureSummary(
        year_month=curr_year_month,
        max_temp=max(curr_temps),
        min_temp=min(curr_temps),
        avg_temp=sum(curr_temps) / len(curr_temps),
    )

    return prev_summary, curr_summary


def build_supplementary_context(
    monthly_report: MonthlyReportData | None,
    temperature: tuple[MonthlyTemperatureSummary, MonthlyTemperatureSummary] | None,
) -> str:
    """月報・気温データをプロンプト用のコンテキスト文字列に変換。"""
    parts: list[str] = []

    if monthly_report:
        parts.append(f"【前年同月データ({monthly_report.month_label})】")
        parts.append(f"- 月間最大電力: {monthly_report.max_power_monthly:.1f} kW")
        parts.append(f"- 月間電力使用量: {monthly_report.total_power_monthly:,.0f} kWh")

        if monthly_report.circuits:
            parts.append("- 回路別内訳:")
            # 電力使用量の多い順にソート
            sorted_circuits = sorted(
                monthly_report.circuits.items(),
                key=lambda x: sum(x[1]),
                reverse=True,
            )
            for circuit_name, daily_values in sorted_circuits[:10]:  # 上位10回路
                total = sum(daily_values)
                parts.append(f"  - {circuit_name}: {total:,.0f} kWh")

    if temperature:
        prev, curr = temperature
        parts.append("")
        parts.append("【気温データ】")

        # 年月を読みやすい形式に変換
        prev_label = prev.year_month.replace("-", "年") + "月"
        curr_label = curr.year_month.replace("-", "年") + "月"

        parts.append(
            f"- {prev_label}: 最高{prev.max_temp:.1f}℃, "
            f"最低{prev.min_temp:.1f}℃, 平均{prev.avg_temp:.1f}℃"
        )

        max_diff = curr.max_temp - prev.max_temp
        avg_diff = curr.avg_temp - prev.avg_temp
        parts.append(
            f"- {curr_label}: 最高{curr.max_temp:.1f}℃, "
            f"最低{curr.min_temp:.1f}℃, 平均{curr.avg_temp:.1f}℃ "
            f"(前年比 最高{max_diff:+.1f}℃, 平均{avg_diff:+.1f}℃)"
        )

    return "\n".join(parts)


@dataclass
class GraphEntry:
    display_label: str
    image: Image.Image | None = None
    text: str | None = None


def collect_graph_entries(paths: Iterable[Path]) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(path) as img:
                entries.append(GraphEntry(display_label=path.name, image=img.convert("RGB")))
            continue

        if suffix == ".pdf":
            for page_index, page_image in pdf_to_images(path):
                label = f"{path.name}#{page_index + 1}"
                entries.append(GraphEntry(display_label=label, image=page_image))
            continue

        if suffix in CSV_EXTENSIONS:
            entries.append(csv_to_graph_entry(path))

    return entries


def pdf_to_images(path: Path) -> list[tuple[int, Image.Image]]:
    document = pdfium.PdfDocument(path)
    images: list[tuple[int, Image.Image]] = []
    for page_index in range(len(document)):
        page = document.get_page(page_index)
        try:
            pil_image = page.render(scale=2).to_pil().convert("RGB")
            images.append((page_index, pil_image))
        finally:
            page.close()
    document.close()
    return images


class GraphLanguageModel(Protocol):
    def comment_on_graph(self, image: Image.Image, prompt: str) -> str: ...

    def comment_on_text(self, text: str, prompt: str) -> str: ...


class GeminiGraphLanguageModel:
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL_NAME):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @classmethod
    def from_env(cls) -> "GeminiGraphLanguageModel":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY is not set"
            raise RuntimeError(msg)
        return cls(api_key=api_key)

    def comment_on_graph(self, image: Image.Image, prompt: str) -> str:
        image_part = self._image_part(image)
        contents = cast(genai_types.ContentListUnionDict, [prompt, image_part])
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text or ""

    def comment_on_text(self, text: str, prompt: str) -> str:
        contents = cast(genai_types.ContentListUnionDict, [prompt, text])
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text or ""

    @staticmethod
    def _image_part(image: Image.Image) -> genai_types.Part:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return genai_types.Part.from_bytes(
            mime_type="image/png",
            data=buffer.getvalue(),
        )


def csv_to_graph_entry(path: Path) -> GraphEntry:
    series = parse_temperature_csv(path)
    text = format_temperature_series(series)
    return GraphEntry(display_label=path.name, text=text)


def _read_csv_rows(path: Path) -> list[list[str]]:
    errors: list[Exception] = []
    for encoding in ("utf-8", "cp932"):  # Shift_JIS (Windows) fallback
        try:
            with path.open(encoding=encoding, newline="") as fp:
                return list(csv.reader(fp))
        except UnicodeDecodeError as exc:
            errors.append(exc)

    error_details = "; ".join(str(error) for error in errors)
    msg = f"CSV {path.name} を UTF-8/Shift_JIS として読み取れませんでした: {error_details}"
    raise ValueError(msg)


def parse_temperature_csv(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for row in _read_csv_rows(path):
        if len(row) < 2:
            continue
        label = row[0].strip()
        try:
            temperature = float(row[1])
        except ValueError:
            continue
        if not label:
            continue
        rows.append((label, temperature))

    if not rows:
        msg = f"CSV {path.name} に有効な気温データがありません"
        raise ValueError(msg)
    return rows


def format_temperature_series(series: Sequence[tuple[str, float]]) -> str:
    lines = ["日付,気温(°C)"]
    lines.extend(f"{label},{temperature}" for label, temperature in series)
    return "\n".join(lines)


def analyze_text(
    text: str,
    prompt: str = PRESET_PROMPT,
    llm: GraphLanguageModel | None = None,
) -> str:
    try:
        model = llm or GeminiGraphLanguageModel.from_env()
    except Exception as exc:
        return f"Gemini呼び出しに失敗しました: {exc}"

    try:
        return model.comment_on_text(text, prompt)
    except Exception as exc:
        return f"Gemini呼び出しに失敗しました: {exc}"


def analyze_image(
    image: Image.Image,
    prompt: str = PRESET_PROMPT,
    llm: GraphLanguageModel | None = None,
) -> str:
    try:
        model = llm or GeminiGraphLanguageModel.from_env()
    except Exception as exc:
        return f"Gemini呼び出しに失敗しました: {exc}"

    try:
        return model.comment_on_graph(image, prompt)
    except Exception as exc:
        return f"Gemini呼び出しに失敗しました: {exc}"
