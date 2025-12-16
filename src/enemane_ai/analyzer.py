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
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
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

CALENDAR_ANALYSIS_PROMPT = dedent(
    """
    あなたは電力管理の上級アナリストです。
    入力は【電力カレンダーPDF画像】と【30分間隔電力使用量データ】です。

    ■ 分析対象
    - PDFのグラフ部分のみを分析対象とする
    - 右側や下部の説明文・コメント欄は無視する
    - カレンダー内の日別グラフ(30分刻み電力使用量推移)を読み取る

    ■ 分析観点
    以下の観点で分析し、事実(グラフ・CSVから読み取れる数値)と
    仮説(電力使用要因の推測)を組み合わせて記述してください。

    1. 全体傾向: 月全体の電力使用パターン、特徴的な傾向
    2. 最大電力使用量日・最大需要電力日の確認: それぞれの日と要因推測
       - 最大電力使用量日: 1日の合計電力使用量(kWh)が最大の日
       - 最大需要電力日: 30分間隔のピーク値(kW)が最大の日
    3. 平日・休日差: 稼働日と非稼働日の消費パターンの違い
    4. 時間帯別パターン: ピーク時間帯、ベースロード
    5. 省エネ改善の示唆: 削減余地のある時間帯や日の特定
    """
)

CALENDAR_OUTPUT_FORMAT = dedent(
    """
    以下のJSON配列形式で出力してください。

    ```json
    [
      {"item": "全体傾向", "analysis": "事実+仮説の記述"},
      {"item": "最大電力使用量日・最大需要電力日の確認", "analysis": "事実+仮説の記述"},
      {"item": "平日・休日差", "analysis": "事実+仮説の記述"},
      {"item": "時間帯別パターン", "analysis": "事実+仮説の記述"},
      {"item": "省エネ改善の示唆", "analysis": "事実+仮説の記述"}
    ]
    ```

    ■ 記述ルール
    - 各項目は2-4文で簡潔に
    - 具体的な数値(日付、kWh、時刻)を必ず含める
    - 事実と仮説を自然な文章で接続する(ラベル表記禁止)
    - 推測で数値を作らない。読めない値は「不明」とする
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


@dataclass
class PowerUsage30min:
    """30分間隔の電力使用量データ"""

    datetime: str  # "2024-10-01 00:00"
    kwh: float  # 4.29


@dataclass
class DailyPowerSummary:
    """日別電力サマリー"""

    date: str  # "2024-10-01"
    day_of_week: str  # "火"
    total_kwh: float  # 日別合計
    max_kwh: float  # 日別最大(30分値)
    max_time: str  # 最大発生時刻 "14:00"


@dataclass
class MonthlyPowerCalendarData:
    """月別電力カレンダーデータ"""

    year_month: str  # "2024年10月"
    daily_summaries: list[DailyPowerSummary] = field(default_factory=list)
    total_monthly_kwh: float = 0.0
    # 最大電力使用量(1日の合計kWhが最大)
    max_usage_day: str = ""  # "22日(火)"
    max_usage_kwh: float = 0.0
    # 最大需要電力(30分ピークが最大)
    max_demand_day: str = ""  # "15日(金)"
    max_demand_kw: float = 0.0
    weekday_avg_kwh: float = 0.0
    weekend_avg_kwh: float = 0.0


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


# 曜日変換用の定数
_WEEKDAY_NAMES = ["月", "火", "水", "木", "金", "土", "日"]


def parse_power_30min_csv(path: Path) -> MonthlyPowerCalendarData:
    """
    30分間隔電力CSVをパースし、月別電力カレンダーデータに集計して返す。

    CSVフォーマット (Shift_JIS):
    - 行1-4: ヘッダー情報(スキップ)
    - 行5以降: "2024-10-01 00:00,4.29,,,,,,,,"
    """
    from datetime import datetime

    rows = _read_csv_rows(path)

    # 30分データを収集
    power_data: list[PowerUsage30min] = []
    datetime_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})")

    for row in rows:
        if len(row) < 2:
            continue

        date_str = row[0].strip()
        match = datetime_pattern.match(date_str)
        if not match:
            continue

        try:
            kwh = float(row[1])
            power_data.append(PowerUsage30min(datetime=date_str, kwh=kwh))
        except ValueError:
            continue

    if not power_data:
        msg = f"CSV {path.name} に有効な30分電力データがありません"
        raise ValueError(msg)

    # 日別に集計
    daily_data: dict[str, list[PowerUsage30min]] = defaultdict(list)
    for item in power_data:
        date_part = item.datetime.split()[0]
        daily_data[date_part].append(item)

    # DailyPowerSummaryのリストを構築
    daily_summaries: list[DailyPowerSummary] = []
    for date_str in sorted(daily_data.keys()):
        items = daily_data[date_str]
        total_kwh = sum(item.kwh for item in items)

        # 最大値とその時刻を特定
        max_item = max(items, key=lambda x: x.kwh)
        max_kwh = max_item.kwh
        max_time = max_item.datetime.split()[1] if " " in max_item.datetime else ""

        # 曜日を取得
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = _WEEKDAY_NAMES[dt.weekday()]

        daily_summaries.append(
            DailyPowerSummary(
                date=date_str,
                day_of_week=day_of_week,
                total_kwh=total_kwh,
                max_kwh=max_kwh,
                max_time=max_time,
            )
        )

    # 月間統計を計算
    total_monthly_kwh = sum(s.total_kwh for s in daily_summaries)

    # 最大電力使用量日を特定(1日の合計が最大)
    max_usage_summary = max(daily_summaries, key=lambda s: s.total_kwh)
    dt_usage = datetime.strptime(max_usage_summary.date, "%Y-%m-%d")
    max_usage_day = f"{dt_usage.day}日({max_usage_summary.day_of_week})"
    max_usage_kwh = max_usage_summary.total_kwh

    # 最大需要電力日を特定(30分ピークが最大)
    max_demand_summary = max(daily_summaries, key=lambda s: s.max_kwh)
    dt_demand = datetime.strptime(max_demand_summary.date, "%Y-%m-%d")
    max_demand_day = f"{dt_demand.day}日({max_demand_summary.day_of_week})"
    max_demand_kw = max_demand_summary.max_kwh

    # 平日/休日平均を計算 (土日を休日とみなす)
    weekday_totals = [s.total_kwh for s in daily_summaries if s.day_of_week not in ("土", "日")]
    weekend_totals = [s.total_kwh for s in daily_summaries if s.day_of_week in ("土", "日")]

    weekday_avg_kwh = sum(weekday_totals) / len(weekday_totals) if weekday_totals else 0.0
    weekend_avg_kwh = sum(weekend_totals) / len(weekend_totals) if weekend_totals else 0.0

    # 年月ラベルを生成
    first_date = daily_summaries[0].date
    year, month, _ = first_date.split("-")
    year_month = f"{year}年{int(month)}月"

    return MonthlyPowerCalendarData(
        year_month=year_month,
        daily_summaries=daily_summaries,
        total_monthly_kwh=total_monthly_kwh,
        max_usage_day=max_usage_day,
        max_usage_kwh=max_usage_kwh,
        max_demand_day=max_demand_day,
        max_demand_kw=max_demand_kw,
        weekday_avg_kwh=weekday_avg_kwh,
        weekend_avg_kwh=weekend_avg_kwh,
    )


def build_power_calendar_context(data: MonthlyPowerCalendarData) -> str:
    """電力カレンダー分析用のコンテキスト文字列を構築。"""
    parts: list[str] = []

    parts.append(f"【30分間隔電力データ({data.year_month})】")
    parts.append(f"- 月間電力使用量: {data.total_monthly_kwh:,.1f} kWh")
    parts.append(f"- 最大電力使用量日: {data.max_usage_day} ({data.max_usage_kwh:,.1f} kWh)")
    parts.append(f"- 最大需要電力日: {data.max_demand_day} ({data.max_demand_kw:.1f} kW)")
    parts.append(f"- 平日平均: {data.weekday_avg_kwh:,.1f} kWh/日")
    parts.append(f"- 休日平均: {data.weekend_avg_kwh:,.1f} kWh/日")

    if data.weekday_avg_kwh > 0 and data.weekend_avg_kwh > 0:
        ratio = data.weekday_avg_kwh / data.weekend_avg_kwh
        parts.append(f"- 平日/休日比: {ratio:.2f}倍")

    # 上位5日の詳細
    parts.append("")
    parts.append("【電力使用量 上位5日】")
    sorted_days = sorted(data.daily_summaries, key=lambda s: s.total_kwh, reverse=True)[:5]
    for s in sorted_days:
        dt_obj = __import__("datetime").datetime.strptime(s.date, "%Y-%m-%d")
        day_label = f"{dt_obj.day}日({s.day_of_week})"
        parts.append(
            f"- {day_label}: {s.total_kwh:,.1f} kWh (最大 {s.max_kwh:.1f} kWh @ {s.max_time})"
        )

    return "\n".join(parts)


# =============================================================================
# 記事検索・要約機能
# =============================================================================


@dataclass
class ArticleFetchResult:
    """ページ取得結果"""

    title: str
    content: str
    og_image: str
    link: str
    og_type: str  # "article", "website" など


ARTICLE_SUMMARIZATION_PROMPT = dedent(
    """
    あなたは、公開ブログ向けに合法性へ配慮して要約するライターです。
    下記の文章を、以下の制約を満たすよう要約してください。

    #制約

    表現の独自化: 事実・主張・根拠を高い抽象度で再記述し、
    原文の決まり文句や比喩・見出し・段落構成を踏襲しない。連続7語以上の一致を禁止。

    再構成: 要点ごとに並べ替え可。原文特有の具体例や固有のリストは一般化する。

    自己チェック: 生成後に語句の類似が強い箇所をさらに抽象化して言い換える。

    #出力形式

    要約本文: 約300-600字、敬体、独自表現。
    """
)


def is_likely_article_url(url: str) -> bool:
    """
    URLパターンから記事ページの可能性を判定する。

    Args:
        url: 判定対象のURL

    Returns:
        bool: 記事ページの可能性が高い場合True
    """
    # 優先パターン (記事ページ) - 先にチェック
    article_patterns = [
        r"/article/",
        r"/column/",
        r"/blog/",
        r"/news/",
        r"/post/",
        r"/entry/",
        r"/\d{4}/\d{2}/",  # 日付パターン (例: /2024/01/)
    ]

    # 優先パターンに一致したら True (除外パターンより優先)
    for pattern in article_patterns:
        if re.search(pattern, url):
            return True

    # 除外パターン (トップページ、カテゴリページなど)
    exclude_patterns = [
        r"^https?://[^/]+/?$",  # トップページのみ (ドメイン直下)
        r"/category/",  # カテゴリページ
        r"/tag/",  # タグページ
        r"/archive/",  # アーカイブページ
        r"/search",  # 検索結果
        r"/page/\d+",  # ページネーション
        r"/author/",  # 著者一覧
    ]

    # 除外パターンに一致したら False
    for pattern in exclude_patterns:
        if re.search(pattern, url):
            return False

    # どちらにも該当しない場合は True (フィルタしすぎを防ぐ)
    return True


def fetch_page_content(url: str, timeout: int = 10) -> ArticleFetchResult:
    """
    URLからページ本文とog:imageを取得する。

    Args:
        url: 取得対象のURL
        timeout: タイムアウト秒数

    Returns:
        ArticleFetchResult: 取得結果

    Raises:
        requests.RequestException: ページ取得に失敗した場合
    """
    # 一般的なブラウザのUser-Agentを使用 (ボットブロック回避)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    # エンコーディング処理
    response.encoding = response.apparent_encoding or "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")

    # タイトル取得
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # og:image取得 (property属性とname属性の両方を試行)
    og_image = ""
    og_image_tag = soup.find("meta", property="og:image")
    if not og_image_tag:
        og_image_tag = soup.find("meta", attrs={"name": "og:image"})
    if not og_image_tag:
        # Twitter Cardのimage
        og_image_tag = soup.find("meta", attrs={"name": "twitter:image"})
    if og_image_tag and hasattr(og_image_tag, "get"):
        img_content = og_image_tag.get("content")
        if img_content:
            og_image = str(img_content)

    # og:type取得 (記事判定に使用)
    og_type = ""
    og_type_tag = soup.find("meta", property="og:type")
    if og_type_tag and hasattr(og_type_tag, "get"):
        type_content = og_type_tag.get("content")
        if type_content:
            og_type = str(type_content)

    # 本文取得 (複数の方法を試行)
    content = ""

    # 方法1: article タグ
    article = soup.find("article")
    if article:
        content = article.get_text(separator="\n", strip=True)

    # 方法2: main タグ
    if not content:
        main = soup.find("main")
        if main:
            content = main.get_text(separator="\n", strip=True)

    # 方法3: body全体からscript/styleを除去
    if not content:
        body = soup.find("body")
        if body and hasattr(body, "find_all"):
            for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = body.get_text(separator="\n", strip=True)

    # 長すぎる場合は切り詰め (Geminiの入力制限を考慮)
    max_content_length = 10000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    return ArticleFetchResult(
        title=title,
        content=content,
        og_image=og_image,
        link=url,
        og_type=og_type,
    )


def summarize_article(content: str, llm: GraphLanguageModel) -> str:
    """
    記事本文をGeminiで要約する。

    Args:
        content: 記事本文
        llm: Geminiクライアント

    Returns:
        str: 要約文
    """
    prompt = f"{ARTICLE_SUMMARIZATION_PROMPT}\n\n#入力内容\n\n入力本文:\n{content}"
    return llm.comment_on_text(content, prompt)


@dataclass
class DuckDuckGoSearchResult:
    """DuckDuckGo検索結果"""

    title: str
    url: str
    body: str  # スニペット


# コラムテーマ種別
ARTICLE_THEME_LAW = "法令改正"
ARTICLE_THEME_TREND = "社会トレンド"
ARTICLE_THEME_CASE = "他社事例"
ARTICLE_THEME_ADVANCED = "先進事例"
ARTICLE_THEME_BEHAVIOR = "従業員行動改善"

# テーマ別の検索クエリテンプレート
THEME_SEARCH_QUERIES: dict[str, list[str]] = {
    ARTICLE_THEME_LAW: [
        "省エネ法 改正 2025 解説 資源エネルギー庁",
        "建築物省エネ法 改正 解説 国交省",
        "法令概要 企業向け 資料",
    ],
    ARTICLE_THEME_TREND: [
        "ESG 情報開示 日本 最新ガイドライン",
        "脱炭素 企業 動向",
        "再エネ 導入 トレンド 企業",
        "サステナビリティ開示 義務化 日本",
    ],
    ARTICLE_THEME_CASE: [
        "省エネ 事例 製造業 導入例",
        "企業 省エネ 導入 成功事例 LED 空調 太陽光",
        "EMS 導入 企業 事例 日本",
        "環境省 補助金 活用事例",
    ],
    ARTICLE_THEME_ADVANCED: [
        "省エネ 先進事例 企業",
        "再エネ 先進的 取り組み 企業",
        "カーボンニュートラル 先進企業 事例",
        "ZEB ZEH 先進事例",
    ],
    ARTICLE_THEME_BEHAVIOR: [
        "従業員 省エネ 行動 改善",
        "オフィス 省エネ 従業員 意識",
        "企業 省エネ 社員教育 事例",
        "エコアクション 従業員参加",
    ],
}

# 利用可能なテーマ一覧
AVAILABLE_ARTICLE_THEMES = list(THEME_SEARCH_QUERIES.keys())


def search_with_duckduckgo(
    theme: str,
    target: str | None = None,
    max_results: int = 10,
) -> list[DuckDuckGoSearchResult]:
    """
    DuckDuckGoで記事を検索する。

    Args:
        theme: 検索テーマ (法令改正/社会トレンド/他社事例/先進事例/従業員行動改善)
        target: 送付先の属性 (検索キーワードに追加)
        max_results: 最大取得件数

    Returns:
        list[DuckDuckGoSearchResult]: 検索結果のリスト
    """
    ddgs = DDGS()
    all_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    # テーマに対応する検索クエリを取得
    search_queries = THEME_SEARCH_QUERIES.get(theme, [f"{theme} コラム 記事"])

    # 各クエリで検索し、結果をマージ
    results_per_query = max(max_results // len(search_queries), 3)

    for query in search_queries:
        # ターゲットが指定されている場合はクエリに追加
        if target and target.strip():
            query = f"{target} {query}"

        try:
            results = ddgs.text(
                query,
                region="jp-jp",
                max_results=results_per_query,
            )
            for r in results:
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        except Exception:
            # 個別クエリの失敗は無視して続行
            continue

        # 十分な結果が得られたら終了
        if len(all_results) >= max_results:
            break

    # 最大件数に制限
    all_results = all_results[:max_results]

    return [
        DuckDuckGoSearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            body=r.get("body", ""),
        )
        for r in all_results
    ]
