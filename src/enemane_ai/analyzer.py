from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterable, Protocol, Sequence, cast

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
    ※【当年最大デマンド発生日データ】と【前年最大デマンド発生日データ】が提供されている場合:
    - 前年との最大デマンド値の比較 (差分とパーセント) を必ず含める
    - ピーク発生時刻の比較 (同一か変化したか) を言及
    - 回路別の増減要因の分析を含める (どの回路が増加/減少に寄与したか)
    - 前年に比べて負荷が大きく増加/減少した回路を特定し、その要因を推測

    ■ 各項目のコメント内容:

    「回路別内訳」: 当月の回路別構成比を説明
    - 例: 上位3回路「1F事務所SR_電灯」(22.6%)、「3F事務所_電灯」(22.2%)...で全体の57.9%
    - 前年との比較は含めない

    「前年比較」: 前年同月との差分・増減を説明
    - 例: 計測回路合計は前年同月の10,120 kWhから10,588 kWhへ+468 kWh(+4.6%)増加
    - 比較値を述べる際は必ず比較元の値を併記
    - どの回路が増減したかを具体的に記載

    ■ kW/kWh混在禁止ルール:
    - 各コメント項目は「kW(最大電力/デマンド)」または「kWh(電力使用量)」のどちらか一方に絞って記述
    - 1文中でkWとkWhを切り替えない

    ■ 気温データとの相関【重要】(kW/kWh別):
    【気温データ】が提供されている場合は、必ず以下のように気温と電力消費の関係をコメントに含める:
    - 電力使用量[kWh]: 気温の影響を受けやすい(冷暖房稼働時間に直結)
      - 気温差(前年比○℃)による冷暖房負荷の増減を言及
    - 最大電力[kW]: 気温だけでなく同時稼働タイミングに依存し、気温との直接的因果は限定的
      - kWの前年差については気温のみで説明しない(同時稼働要因を考慮)
    - 前年比較: 気温差が電力使用量の増減に与えた影響を必ず記載
      例: 「前年同月比で最高気温+3.6℃、平均気温+2.1℃となった気温差による冷房設備の
           負荷増大が電力使用量増加の主要因である可能性があります」
    - 日別推移: 特定日の気温と電力ピークの相関を言及
    - 気温と電力消費の相関が明確でない場合は、無理に結びつけない
    - 中間期(春秋)は冷暖房負荷が小さいため、気温影響は限定的

    ■ 気温指標の使い分け:
    - 平均気温: 月全体のkWh増減の主因分析
    - 最高気温: 冷房ピーク(夏期kWピーク)の分析
    - 最低気温: 暖房起動(冬期の底冷え、早朝暖房立ち上げ)の分析

    ■ 各項目の独立性【重要】
    - 各出力項目は独立した観点で記述し、他の項目と内容が重複しないようにする
    - 同一テーマに触れる必要がある場合は、補足・別の視点からの分析として記述

    ■ 前年比較の計算:
    - 【前年同月データ】の数値を使って前年比を計算
    - 【気温データ】があれば気温差との相関を必ずコメントに含める
    """
)

CALENDAR_ANALYSIS_PROMPT = dedent(
    """
    あなたは電力管理の上級アナリストです。
    入力は【電力カレンダーPDF画像】と【30分間隔電力使用量データ】です。
    任意で【前年同月の電力データ】と【気温データ】が含まれます。

    ■ 分析対象
    - PDFのグラフ部分のみを分析対象とする
    - 右側や下部の説明文・コメント欄は無視する
    - カレンダー内の日別グラフ(30分刻み電力使用量推移)を読み取る

    ■ 分析観点
    以下の観点で分析し、事実(グラフ・CSVから読み取れる数値)と
    仮説(電力使用要因の推測)を組み合わせて記述してください。

    1. 全体傾向: 月全体の電力使用パターン、特徴的な傾向
       - 前年データがある場合: 前年同月比の増減と主要因を推測
    2. 最大電力使用量日・最大需要電力日の確認: それぞれの日と要因推測
       - 最大電力使用量日: 1日の合計電力使用量(kWh)が最大の日
       - 最大需要電力日: 30分間隔のピーク値(kW)が最大の日
       - 前年データがある場合: 前年同日との比較
    3. 稼働日・非稼働日パターン: 稼働日と非稼働日の消費パターンの違い
       - 【施設属性情報】がある場合は営業日カレンダーに基づいて判断
       - 施設属性がない場合は平日/休日(土日)で分析
       - 曜日ごとの特徴的パターンを分析(月曜の立ち上げ負荷、土曜営業の有無等)
       - 前年データがある場合: 前年との平日/休日平均の比較
    4. 時間帯別パターン: ピーク時間帯、ベースロード
    5. 省エネ改善の示唆: 削減余地のある時間帯や日の特定
       - 柔らかいトーンで記述(「〜をご検討ください」「〜の見直し余地があります」)
       - 前年比で増加している場合はその要因分析を含める

    ■ 表現ルール【厳守】
    - 強い断定表現は禁止。推測・因果関係には以下の表現に統一:
      - 「〜可能性があります」「〜と考えられます」「〜かもしれません」「〜の傾向が見られます」
    - 使用禁止: 「〜が原因です」「〜に違いありません」(推測を事実として述べる表現)
    - 数値の読み取り事実のみ断定可。因果関係や要因推測は必ず推測語を使用

    ■ 施設属性に基づく分析【重要】
    【施設属性情報】が提供されている場合:
    - 各フロア/施設の営業形態に応じた電力パターンを考慮
    - 営業日情報をもとに「稼働日/非稼働日」を判断(単純な平日/休日ではない)

    ■ 気温データとの相関【重要】
    【気温データ】が提供されている場合は、必ず以下のように気温と電力消費の関係をコメントに含める:
    - 気温が高い/低い日にピークが発生した可能性を言及
    - 気温差(前年比○℃)による冷暖房負荷の増減を言及
    - 電力使用量の増減が気温要因か、その他要因かを推測
      例: 「前年同月比で平均気温+2.1℃となった気温差による冷房設備の
           負荷増大が電力使用量増加の主要因である可能性があります」
    - 気温と電力消費の相関が明確でない場合は、無理に結びつけない
    - 中間期(春秋)は冷暖房負荷が小さいため、気温影響は限定的
    """
)

CALENDAR_OUTPUT_FORMAT = dedent(
    """
    以下のJSON配列形式で出力してください。

    ```json
    [
      {"item": "全体傾向", "analysis": "事実+仮説の記述 (前年比較・気温相関を含む)"},
      {"item": "最大電力使用量日・最大需要電力日の確認", "analysis": "事実+仮説の記述"},
      {"item": "平日・休日差", "analysis": "事実+仮説の記述 (前年比較を含む)"},
      {"item": "時間帯別パターン", "analysis": "事実+仮説の記述"},
      {"item": "省エネ改善の示唆", "analysis": "事実+仮説の記述 (気温要因の考慮を含む)"}
    ]
    ```

    ■ 記述ルール
    - 各項目は2-4文で簡潔に
    - 具体的な数値(日付、kWh、時刻、気温差)を必ず含める
    - 前年データがある場合は前年比(%)を必ず含める
    - 気温データがある場合は気温との相関を必ず含める
    - 事実と仮説を自然な文章で接続する(ラベル表記禁止)
    - 推測で数値を作らない。読めない値は「不明」とする

    ■ 表現ルール【厳守】
    - 強い断定表現は禁止。推測・因果関係には推測語を使用
    - 改善提案は柔らかいトーンで記述(「〜をご検討ください」「〜の見直し余地があります」)

    ■ 各項目の独立性【重要】
    - 各出力項目は独立した観点で記述し、他の項目と内容が重複しないようにする
    - 同一テーマに触れる必要がある場合は、補足・別の視点からの分析として記述
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

    ■ 表現ルール【厳守】
    - 強い断定表現は禁止。推測・因果関係には以下の表現に統一:
      - 「〜可能性があります」「〜と考えられます」「〜かもしれません」「〜の傾向が見られます」
    - 使用禁止: 「〜が原因です」「〜に違いありません」(推測を事実として述べる表現)
    - 数値の読み取り事実のみ断定可。因果関係や要因推測は必ず推測語を使用

    ■ 改善提案の表現トーン【厳守】
    - 運用改善案・省エネ提案は柔らかいトーンに統一:
      - 「〜をご検討ください」「〜の見直し余地があります」「〜の余地があると考えられます」
    - 使用禁止: 「〜すべきです」「〜する必要があります」「〜しなければなりません」
    - 「設定最適化」「運用改善」等の曖昧なワードは、具体化できない場合は使用しない
      - 良い例: 「空調の設定温度を1℃緩和することをご検討ください」
      - 悪い例: 「設定最適化をご検討ください」

    ■ 前月比・前年比の記述ルール【厳守】
    - 比較値(差分やパーセント)を述べる際は、必ず比較元の値を併記
      - 良い例: 「前月の12,500 kWhから+596 kWh(+4.8%)増加し、13,096 kWhとなりました」
      - 悪い例: 「+596 kWh増加しました」(比較元が不明)

    ■ kW/kWh の混在禁止【厳守】
    - 各コメント項目は「kW(最大電力/デマンド)」または「kWh(電力使用量)」のどちらか一方に絞って記述
    - T1+T2: 「最大電力[kW]」のコメントにkWh記述を含めない。逆も同様
    - T3回路別内訳: kWh(使用量)ベースの構成比のみ
    - T5最大デマンド: kW(需要電力)ベースのみ
    - 各文は「kW」または「kWh」のどちらか一方のみを扱い、1文中で切り替えない

    ■ T5 最大デマンド分析【厳守】
    - 前時限→ピーク時限→次時限の3点を必ず比較
    - ピークの形状(急上昇型/緩やか型/台形型)を分析
    - 回路別にも3点の推移を記述し、ピーク形成に寄与した回路を特定

    ■ 施設属性に基づく分析【重要】
    【施設属性情報】が提供されている場合:
    - 各フロア/施設の営業形態に応じた電力パターンを考慮
    - 営業日情報をもとに「稼働日/非稼働日」を判断(単純な平日/休日ではない)

    ■ 曜日別分析【重要】
    - 曜日ごとの特徴的パターンがあれば言及:
      - 月曜日: 週明け立ち上げ負荷(空調冷やし込み等)
      - 土日: 施設属性によっては稼働日。一律に「休日=低い」としない
    - 【施設属性情報】がある場合は営業日カレンダーに基づいて分析

    ■ 増加要因の推測ルール【重要】
    - 詳細な要因推測(稼働時間の変化、負荷種類の差等)は、30分間隔データ等の裏付けがある場合にのみ記述
    - データがない場合は「詳細な要因分析には30分間隔データの照合が必要です」と補足

    ■ 回路別変化の時系列記述【重要】
    - 回路別の増減はピーク時点の値のみでなく、時系列での変化パターンを記述
    - ベースロードとの比較を通じて、常時増加 vs ピーク集中を分析

    ■ 補助データの提案
    - 分析精度向上に有効な補助データがある場合は、1文で簡潔に言及
      例: 「来場者数データと照合することでショールームの電力変動要因をより正確に分析できます」

    ■ 回路名称の解釈
    【回路名称→用途】が提供されている場合は、用途名称を活用して分かりやすく説明する
      例: 「1F事務所SR_電灯(ショールーム照明)が構成比22.6%で最大」
    """
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
CSV_EXTENSIONS = {".csv"}
DEFAULT_MODEL_NAME = "gemini-3-pro-preview"
RECEIVED_POWER_CIRCUIT = "受電電力"


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


@dataclass
class PeakDayCircuitData:
    """回路別30分間隔データ"""

    circuit_name: str  # "受電電力", "1F事務所SR_電灯", etc.
    values: list[float] = field(default_factory=list)  # 48 values for 30-min intervals


@dataclass
class PeakDayPowerData:
    """最大デマンド発生日の詳細データ (グラフ用データ(4))"""

    peak_date: str  # "2024/10/02"
    time_slots: list[str] = field(default_factory=list)  # ["0:00", "0:30", ..., "23:30"]
    circuits: list[PeakDayCircuitData] = field(default_factory=list)

    def _find_peak_index(self) -> int | None:
        """受電電力の最大値インデックスを返す。受電電力がない/空の場合はNone。"""
        for circuit in self.circuits:
            if circuit.circuit_name == RECEIVED_POWER_CIRCUIT:
                if circuit.values:
                    return max(range(len(circuit.values)), key=lambda i: circuit.values[i])
                return None
        return None

    @property
    def peak_time(self) -> str:
        """最大デマンド発生時刻を返す (受電電力の最大値の時刻)"""
        peak_idx = self._find_peak_index()
        if peak_idx is None:
            return ""
        return self.time_slots[peak_idx] if peak_idx < len(self.time_slots) else ""

    @property
    def peak_power_kw(self) -> float:
        """最大デマンド値 (受電電力の最大値 * 2 for kW)"""
        peak_idx = self._find_peak_index()
        if peak_idx is None:
            return 0.0
        for circuit in self.circuits:
            if circuit.circuit_name == RECEIVED_POWER_CIRCUIT:
                return circuit.values[peak_idx] * 2  # kWh (30min) -> kW
        return 0.0

    def get_circuit_power_at_peak(self) -> dict[str, float]:
        """ピーク時刻における各回路の電力値を返す"""
        peak_idx = self._find_peak_index()
        if peak_idx is None:
            return {}

        return {
            circuit.circuit_name: circuit.values[peak_idx]
            for circuit in self.circuits
            if peak_idx < len(circuit.values)
        }

    def get_circuit_power_3point(self) -> dict[str, tuple[float | None, float, float | None]]:
        """前時限→ピーク→次時限の3点データを返す。

        Returns:
            回路名 -> (前時限値, ピーク値, 次時限値) のdict。
            前時限/次時限が存在しない場合(先頭/末尾スロット)はNone。
        """
        peak_idx = self._find_peak_index()
        if peak_idx is None:
            return {}

        result: dict[str, tuple[float | None, float, float | None]] = {}
        for circuit in self.circuits:
            if peak_idx >= len(circuit.values):
                continue
            peak_val = circuit.values[peak_idx]
            prev_val = circuit.values[peak_idx - 1] if peak_idx > 0 else None
            next_val = circuit.values[peak_idx + 1] if peak_idx + 1 < len(circuit.values) else None
            result[circuit.circuit_name] = (prev_val, peak_val, next_val)

        return result


@dataclass
class FacilityProfile:
    """施設属性データ"""

    facility_type: str  # "ショールーム", "オフィス", "工場" 等
    floor_label: str  # "1F", "3F" 等
    business_days: tuple[str, ...]  # 正規化済み: ("月","火","水","木","金","土")
    notes: str = ""


_VALID_WEEKDAYS = frozenset("月火水木金土日")


def parse_business_days(s: str) -> tuple[str, ...]:
    """曜日文字列をバリデーションしてタプルに変換。"""
    days: list[str] = []
    for ch in s.replace(",", "").replace("、", "").replace(" ", "").replace("　", ""):
        if ch in _VALID_WEEKDAYS and ch not in days:
            days.append(ch)
    return tuple(days)


def parse_facility_profiles(text: str) -> tuple[list[FacilityProfile], list[str]]:
    """
    施設属性テキストをパースする。

    入力形式: 1行1エントリ、カンマ区切り
    「フロア,施設タイプ,営業日,備考」

    Returns:
        (有効なプロファイルリスト, 警告メッセージリスト)
    """
    profiles: list[FacilityProfile] = []
    warnings: list[str] = []

    if not text.strip():
        return profiles, warnings

    # 全角カンマのみ半角カンマに正規化。
    # 読点は営業日や備考の内部で使われるため変換しない。
    text = text.replace("\uff0c", ",")

    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            warnings.append(f"行{i}: フォーマット不正 (フロア,施設タイプ,営業日 が必要): {line}")
            continue

        floor_label = parts[0]
        facility_type = parts[1]

        # parts[2:]から曜日フィールドと備考を分離。
        # 曜日文字のみで構成されるフィールドは営業日の一部とみなす。
        _day_separators = " \u3000\u3001"  # space, fullwidth space, toten
        day_parts: list[str] = []
        notes_start = len(parts)
        for j, p in enumerate(parts[2:], start=2):
            stripped = "".join(ch for ch in p if ch not in _day_separators)
            if stripped and all(ch in _VALID_WEEKDAYS for ch in stripped):
                day_parts.append(p)
            else:
                notes_start = j
                break

        business_days = parse_business_days("".join(day_parts))

        if not business_days:
            warnings.append(f"行{i}: 有効な曜日がありません: {parts[2]}")
            continue

        notes = ",".join(parts[notes_start:]) if notes_start < len(parts) else ""
        profiles.append(
            FacilityProfile(
                facility_type=facility_type,
                floor_label=floor_label,
                business_days=business_days,
                notes=notes,
            )
        )

    return profiles, warnings


def build_facility_context(profiles: list[FacilityProfile]) -> str:
    """施設属性情報のコンテキスト文字列を構築。"""
    if not profiles:
        return ""

    parts: list[str] = ["【施設属性情報】"]
    for p in profiles:
        days_str = ",".join(p.business_days)
        line = f"- {p.floor_label} ({p.facility_type}): 営業日={days_str}"
        if p.notes:
            line += f" / {p.notes}"
        parts.append(line)

    return "\n".join(parts)


def parse_circuit_mapping(text: str) -> tuple[dict[str, str], list[str]]:
    """
    回路名称→用途マッピングテキストをパースする。

    入力形式: 1行1エントリ、「回路名:用途」または「回路名→用途」

    Returns:
        (有効なマッピング, 警告メッセージリスト)
    """
    mapping: dict[str, str] = {}
    warnings: list[str] = []

    if not text.strip():
        return mapping, warnings

    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        # 区切り文字を試行
        for sep in ("\u2192", ":", "\uff1a"):  # arrow, colon, fullwidth colon
            if sep in line:
                parts = line.split(sep, 1)
                circuit_name = parts[0].strip()
                usage = parts[1].strip()
                if circuit_name and usage:
                    mapping[circuit_name] = usage
                else:
                    warnings.append(f"行{i}: 回路名または用途が空です: {line}")
                break
        else:
            warnings.append(f"行{i}: 区切り文字が見つかりません (回路名:用途 の形式): {line}")

    return mapping, warnings


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
        elif row_name == RECEIVED_POWER_CIRCUIT:
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
    facility_profiles: list[FacilityProfile] | None = None,
    circuit_mapping: dict[str, str] | None = None,
) -> str:
    """月報・気温データをプロンプト用のコンテキスト文字列に変換。"""
    parts: list[str] = []

    if monthly_report:
        parts.append(f"【前年同月データ({monthly_report.month_label})】")
        parts.append(f"- 月間最大電力: {monthly_report.max_power_monthly:.1f} kW")
        parts.append(f"- 月間電力使用量: {monthly_report.total_power_monthly:,.0f} kWh")

        if monthly_report.circuits:
            parts.append("- 回路別内訳:")
            sorted_circuits = sorted(
                monthly_report.circuits.items(),
                key=lambda x: sum(x[1]),
                reverse=True,
            )
            for circuit_name, daily_values in sorted_circuits[:10]:
                total = sum(daily_values)
                display_name = _format_circuit_name(circuit_name, circuit_mapping)
                parts.append(f"  - {display_name}: {total:,.0f} kWh")

    if temperature:
        prev, curr = temperature
        parts.append("")
        parts.append("【気温データ】")

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

    # 施設属性情報
    parts.extend(_build_facility_context_section(facility_profiles))

    # 回路名称→用途マッピング
    parts.extend(_build_circuit_mapping_section(circuit_mapping))

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
    max_demand_kw = max_demand_summary.max_kwh * 2  # 30分kWh→kW変換

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


def parse_peak_day_power_csv(path: Path) -> PeakDayPowerData:
    """
    グラフ用データ(4) CSVをパースして構造化データに変換。

    CSVフォーマット:
    - 行1: ヘッダー説明 "(4) 詳細データ..."
    - 行2: 日付と48時間スロット "2024/10/02,0:00,0:30,1:00,...,23:30"
    - 行3+: 回路名と48値 "受電電力,4.18,4.18,4.03,..."

    Args:
        path: CSVファイルパス

    Returns:
        PeakDayPowerData: パースされたピーク日電力データ

    Raises:
        ValueError: CSVフォーマットが不正な場合
    """
    rows = _read_csv_rows(path)

    if len(rows) < 3:
        msg = f"グラフ用データ(4) CSV {path.name} のフォーマットが不正です(行数不足)"
        raise ValueError(msg)

    # 行2: 日付と時間スロット
    date_time_row = rows[1]
    if len(date_time_row) < 2:
        msg = f"グラフ用データ(4) CSV {path.name} の時間スロットが不足しています"
        raise ValueError(msg)

    peak_date = date_time_row[0].strip()
    # 時間スロットは最大48個まで (0:00〜23:30)
    max_slots = min(len(date_time_row) - 1, 48)
    time_slots = [cell.strip() for cell in date_time_row[1 : max_slots + 1]]

    # 日付フォーマット検証
    date_match = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", peak_date)
    if not date_match:
        msg = f"グラフ用データ(4) CSV {path.name} の日付フォーマットが不正です: {peak_date}"
        raise ValueError(msg)

    # 行3以降: 回路データ
    circuits: list[PeakDayCircuitData] = []

    for row in rows[2:]:
        if not row or not row[0].strip():
            continue

        circuit_name = row[0].strip()
        values: list[float] = []

        # 時間スロット数に合わせて値を取得
        for cell in row[1 : len(time_slots) + 1]:
            try:
                values.append(float(cell))
            except ValueError:
                values.append(0.0)

        # 時間スロット数に満たない場合は0で埋める
        while len(values) < len(time_slots):
            values.append(0.0)

        circuits.append(
            PeakDayCircuitData(
                circuit_name=circuit_name,
                values=values,
            )
        )

    if not circuits:
        msg = f"グラフ用データ(4) CSV {path.name} に有効な回路データがありません"
        raise ValueError(msg)

    return PeakDayPowerData(
        peak_date=peak_date,
        time_slots=time_slots,
        circuits=circuits,
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
            f"- {day_label}: {s.total_kwh:,.1f} kWh (最大 {s.max_kwh * 2:.1f} kW @ {s.max_time})"
        )

    return "\n".join(parts)


def _compute_operating_day_averages(
    daily_summaries: list[DailyPowerSummary],
    facility_profiles: list[FacilityProfile],
) -> tuple[float | None, float | None]:
    """施設属性に基づく稼働日/非稼働日平均を計算。

    稼働日 = いずれかのフロアの営業日に該当する日

    Returns:
        (稼働日平均kWh or None, 非稼働日平均kWh or None)
        該当日がない場合はNoneを返す。
    """
    # 全フロアの営業日をOR結合
    all_business_days: set[str] = set()
    for p in facility_profiles:
        all_business_days.update(p.business_days)

    operating_totals: list[float] = []
    non_operating_totals: list[float] = []

    for s in daily_summaries:
        if s.day_of_week in all_business_days:
            operating_totals.append(s.total_kwh)
        else:
            non_operating_totals.append(s.total_kwh)

    op_avg = sum(operating_totals) / len(operating_totals) if operating_totals else None
    non_op_avg = (
        sum(non_operating_totals) / len(non_operating_totals) if non_operating_totals else None
    )
    return op_avg, non_op_avg


def build_power_calendar_extended_context(
    curr_power: MonthlyPowerCalendarData,
    prev_power: MonthlyPowerCalendarData | None = None,
    temperature: tuple[MonthlyTemperatureSummary, MonthlyTemperatureSummary] | None = None,
    facility_profiles: list[FacilityProfile] | None = None,
) -> str:
    """
    電力カレンダー分析用の拡張コンテキスト文字列を構築。

    前年データおよび気温データを含めた包括的なコンテキストを生成する。

    Args:
        curr_power: 当年の電力データ
        prev_power: 前年の電力データ (オプション)
        temperature: 気温データ (前年, 当年) のタプル (オプション)
        facility_profiles: 施設属性データ (オプション)

    Returns:
        str: プロンプト用のコンテキスト文字列
    """
    from datetime import datetime

    parts: list[str] = []

    # 当年電力データ
    parts.append(f"【当年30分間隔電力データ({curr_power.year_month})】")
    parts.append(f"- 月間電力使用量: {curr_power.total_monthly_kwh:,.1f} kWh")
    parts.append(
        f"- 最大電力使用量日: {curr_power.max_usage_day} ({curr_power.max_usage_kwh:,.1f} kWh)"
    )
    parts.append(
        f"- 最大需要電力日: {curr_power.max_demand_day} ({curr_power.max_demand_kw:.1f} kW)"
    )
    parts.append(f"- 平日平均: {curr_power.weekday_avg_kwh:,.1f} kWh/日")
    parts.append(f"- 休日平均: {curr_power.weekend_avg_kwh:,.1f} kWh/日")

    if curr_power.weekday_avg_kwh > 0 and curr_power.weekend_avg_kwh > 0:
        ratio = curr_power.weekday_avg_kwh / curr_power.weekend_avg_kwh
        parts.append(f"- 平日/休日比: {ratio:.2f}倍")

    # 稼働日/非稼働日平均 (施設属性がある場合)
    if facility_profiles and curr_power.daily_summaries:
        op_avg, non_op_avg = _compute_operating_day_averages(
            curr_power.daily_summaries, facility_profiles
        )
        if op_avg is not None:
            parts.append(f"- 稼働日平均: {op_avg:,.1f} kWh/日")
        if non_op_avg is not None:
            parts.append(f"- 非稼働日平均: {non_op_avg:,.1f} kWh/日")

    # 当年電力使用量 上位5日
    parts.append("")
    parts.append("【当年電力使用量 上位5日】")
    sorted_days = sorted(curr_power.daily_summaries, key=lambda s: s.total_kwh, reverse=True)[:5]
    for s in sorted_days:
        dt_obj = datetime.strptime(s.date, "%Y-%m-%d")
        day_label = f"{dt_obj.day}日({s.day_of_week})"
        parts.append(
            f"- {day_label}: {s.total_kwh:,.1f} kWh (最大 {s.max_kwh * 2:.1f} kW @ {s.max_time})"
        )

    # 施設属性情報
    parts.extend(_build_facility_context_section(facility_profiles))

    # 前年電力データ (存在する場合)
    if prev_power:
        parts.append("")
        parts.append(f"【前年30分間隔電力データ({prev_power.year_month})】")
        parts.append(f"- 月間電力使用量: {prev_power.total_monthly_kwh:,.1f} kWh")
        parts.append(
            f"- 最大電力使用量日: {prev_power.max_usage_day} ({prev_power.max_usage_kwh:,.1f} kWh)"
        )
        parts.append(
            f"- 最大需要電力日: {prev_power.max_demand_day} ({prev_power.max_demand_kw:.1f} kW)"
        )
        parts.append(f"- 平日平均: {prev_power.weekday_avg_kwh:,.1f} kWh/日")
        parts.append(f"- 休日平均: {prev_power.weekend_avg_kwh:,.1f} kWh/日")

        # 前年比較
        parts.append("")
        parts.append("【前年比較】")
        power_diff = curr_power.total_monthly_kwh - prev_power.total_monthly_kwh
        power_pct = (
            (power_diff / prev_power.total_monthly_kwh * 100)
            if prev_power.total_monthly_kwh > 0
            else 0
        )
        parts.append(f"- 月間電力使用量: {power_diff:+,.1f} kWh ({power_pct:+.1f}%)")

        weekday_diff = curr_power.weekday_avg_kwh - prev_power.weekday_avg_kwh
        weekend_diff = curr_power.weekend_avg_kwh - prev_power.weekend_avg_kwh
        parts.append(f"- 平日平均差: {weekday_diff:+,.1f} kWh/日")
        parts.append(f"- 休日平均差: {weekend_diff:+,.1f} kWh/日")

        # 施設属性がある場合は稼働日/非稼働日ベースの前年比較も出力
        if facility_profiles and prev_power.daily_summaries:
            prev_op_avg, prev_non_op_avg = _compute_operating_day_averages(
                prev_power.daily_summaries, facility_profiles
            )
            # 当年の稼働日平均は上で計算済み (curr の daily_summaries がある場合)
            if curr_power.daily_summaries:
                curr_op_avg, curr_non_op_avg = _compute_operating_day_averages(
                    curr_power.daily_summaries, facility_profiles
                )
                if curr_op_avg is not None and prev_op_avg is not None:
                    parts.append(f"- 稼働日平均差: {curr_op_avg - prev_op_avg:+,.1f} kWh/日")
                if curr_non_op_avg is not None and prev_non_op_avg is not None:
                    parts.append(
                        f"- 非稼働日平均差: {curr_non_op_avg - prev_non_op_avg:+,.1f} kWh/日"
                    )

    # 気温データ
    if temperature:
        prev_temp, curr_temp = temperature
        parts.append("")
        parts.append("【気温データ】")

        prev_label = prev_temp.year_month.replace("-", "年") + "月"
        parts.append(
            f"- {prev_label}: 最高{prev_temp.max_temp:.1f}℃, "
            f"最低{prev_temp.min_temp:.1f}℃, 平均{prev_temp.avg_temp:.1f}℃"
        )

        curr_label = curr_temp.year_month.replace("-", "年") + "月"
        max_diff = curr_temp.max_temp - prev_temp.max_temp
        avg_diff = curr_temp.avg_temp - prev_temp.avg_temp
        parts.append(
            f"- {curr_label}: 最高{curr_temp.max_temp:.1f}℃, "
            f"最低{curr_temp.min_temp:.1f}℃, 平均{curr_temp.avg_temp:.1f}℃ "
            f"(前年比 最高{max_diff:+.1f}℃, 平均{avg_diff:+.1f}℃)"
        )

    return "\n".join(parts)


def _format_circuit_name(name: str, circuit_mapping: dict[str, str] | None) -> str:
    """回路名に用途名を付記する。"""
    if circuit_mapping and name in circuit_mapping:
        return f"{name}({circuit_mapping[name]})"
    return name


def _build_circuit_mapping_section(circuit_mapping: dict[str, str] | None) -> list[str]:
    """回路名称→用途マッピングのコンテキスト行を返す。"""
    if not circuit_mapping:
        return []
    parts: list[str] = ["", "【回路名称→用途】"]
    for name, usage in circuit_mapping.items():
        parts.append(f"- {name}: {usage}")
    return parts


def _build_facility_context_section(
    facility_profiles: list[FacilityProfile] | None,
) -> list[str]:
    """施設属性情報のコンテキスト行を返す。"""
    if not facility_profiles:
        return []
    facility_ctx = build_facility_context(facility_profiles)
    if not facility_ctx:
        return []
    return ["", facility_ctx]


def _build_peak_circuit_breakdown(
    peak_circuits: dict[str, float],
    circuit_mapping: dict[str, str] | None = None,
    max_circuits: int = 10,
) -> list[str]:
    """ピーク時刻の回路別内訳行を返す。"""
    if not peak_circuits:
        return []
    parts: list[str] = ["- ピーク時刻の回路別内訳 (kWh/30分):"]
    sorted_circuits = sorted(
        [(name, val) for name, val in peak_circuits.items() if name != RECEIVED_POWER_CIRCUIT],
        key=lambda x: x[1],
        reverse=True,
    )
    for circuit_name, value in sorted_circuits[:max_circuits]:
        kw_value = value * 2
        display_name = _format_circuit_name(circuit_name, circuit_mapping)
        parts.append(f"  - {display_name}: {kw_value:.1f} kW ({value:.2f} kWh)")
    return parts


def _build_3point_section(
    peak_data: PeakDayPowerData,
    label: str,
    circuit_mapping: dict[str, str] | None = None,
) -> list[str]:
    """3点比較 (前時限→ピーク→次時限) セクションを構築。"""
    parts: list[str] = []
    three_point = peak_data.get_circuit_power_3point()
    if not three_point:
        return parts

    # 受電電力のピーク値と前後をチェック
    recv = three_point.get(RECEIVED_POWER_CIRCUIT)
    if recv:
        prev_val, peak_val, next_val = recv
        # ガード: ピーク値が正で前後の両方が0.0の場合のみデータ不足と判断
        # (片側だけ0.0は有効な計測値の可能性がある)
        if (
            peak_val > 0
            and prev_val is not None
            and prev_val == 0.0
            and next_val is not None
            and next_val == 0.0
        ):
            parts.append(f"- {label}3点比較: (データ不足のため3点比較なし)")
            return parts

    parts.append(f"- {label}3点比較 (前時限→ピーク→次時限):")
    for circuit_name, (prev_v, peak_v, next_v) in three_point.items():
        display_name = _format_circuit_name(circuit_name, circuit_mapping)
        prev_kw = f"{prev_v * 2:.1f}" if prev_v is not None else "-"
        peak_kw = f"{peak_v * 2:.1f}"
        next_kw = f"{next_v * 2:.1f}" if next_v is not None else "-"
        parts.append(f"  - {display_name}: {prev_kw} → {peak_kw} → {next_kw} kW")

    return parts


def build_peak_day_comparison_context(
    curr_peak: PeakDayPowerData | None,
    prev_peak: PeakDayPowerData | None = None,
    circuit_mapping: dict[str, str] | None = None,
) -> str:
    """
    最大デマンド発生日の前年比較コンテキスト文字列を構築。

    T5グラフ (最大デマンド発生時) の分析用コンテキストを生成する。

    Args:
        curr_peak: 当年の最大デマンド発生日データ
        prev_peak: 前年の最大デマンド発生日データ (オプション)
        circuit_mapping: 回路名→用途名のマッピング (オプション)

    Returns:
        str: プロンプト用のコンテキスト文字列
    """
    if curr_peak is None:
        return ""

    parts: list[str] = []

    # 当年データ
    parts.append(f"【当年最大デマンド発生日データ ({curr_peak.peak_date})】")
    parts.append(f"- 最大デマンド発生時刻: {curr_peak.peak_time}")
    parts.append(f"- 最大デマンド値: {curr_peak.peak_power_kw:.1f} kW")

    # 当年ピーク時の回路別内訳
    curr_peak_circuits = curr_peak.get_circuit_power_at_peak()
    parts.extend(_build_peak_circuit_breakdown(curr_peak_circuits, circuit_mapping))

    # 当年3点比較
    parts.extend(_build_3point_section(curr_peak, "当年", circuit_mapping))

    # 回路マッピング情報
    parts.extend(_build_circuit_mapping_section(circuit_mapping))

    # 前年データ (存在する場合)
    if prev_peak:
        parts.append("")
        parts.append(f"【前年最大デマンド発生日データ ({prev_peak.peak_date})】")
        parts.append(f"- 最大デマンド発生時刻: {prev_peak.peak_time}")
        parts.append(f"- 最大デマンド値: {prev_peak.peak_power_kw:.1f} kW")

        prev_peak_circuits = prev_peak.get_circuit_power_at_peak()
        parts.extend(_build_peak_circuit_breakdown(prev_peak_circuits, circuit_mapping))

        # 前年3点比較
        parts.extend(_build_3point_section(prev_peak, "前年", circuit_mapping))

        # 前年比較
        parts.append("")
        parts.append("【前年比較】")
        power_diff = curr_peak.peak_power_kw - prev_peak.peak_power_kw
        if prev_peak.peak_power_kw > 0:
            power_pct = (power_diff / prev_peak.peak_power_kw) * 100
            parts.append(f"- 最大デマンド変化: {power_diff:+.1f} kW ({power_pct:+.1f}%)")
        else:
            parts.append(f"- 最大デマンド変化: {power_diff:+.1f} kW")

        if curr_peak.peak_time and prev_peak.peak_time:
            if curr_peak.peak_time == prev_peak.peak_time:
                parts.append(f"- 発生時刻: 同一 ({curr_peak.peak_time})")
            else:
                parts.append(f"- 発生時刻: {prev_peak.peak_time} → {curr_peak.peak_time}")

        if curr_peak_circuits and prev_peak_circuits:
            parts.append("- 回路別変化 (主要回路):")
            common_circuits = set(curr_peak_circuits.keys()) & set(prev_peak_circuits.keys())
            changes: list[tuple[str, float, float, float]] = []
            for circuit_name in common_circuits:
                if circuit_name == RECEIVED_POWER_CIRCUIT:
                    continue
                curr_val = curr_peak_circuits[circuit_name] * 2
                prev_val = prev_peak_circuits[circuit_name] * 2
                diff = curr_val - prev_val
                if abs(diff) >= 0.5:
                    changes.append((circuit_name, curr_val, prev_val, diff))

            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            for circuit_name, curr_val, prev_val, diff in changes[:5]:
                display_name = _format_circuit_name(circuit_name, circuit_mapping)
                parts.append(
                    f"  - {display_name}: {prev_val:.1f} kW → {curr_val:.1f} kW ({diff:+.1f} kW)"
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

    要約本文: 約[300-400]字、敬体、独自表現。

    出典
    - タイトル: [元記事タイトル]
    - URL: [元記事URL]
    - 著者: [著者名。不明な場合は省略]
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


def summarize_article(
    content: str,
    llm: GraphLanguageModel,
    *,
    title: str = "",
    url: str = "",
    author: str = "",
) -> str:
    """
    記事本文をGeminiで要約する。

    Args:
        content: 記事本文
        llm: Geminiクライアント
        title: 元記事タイトル
        url: 元記事URL
        author: 著者名 (不明な場合は空文字)

    Returns:
        str: 要約文
    """
    input_parts = [
        "#入力内容",
        f"元記事タイトル: {title}" if title else "",
        f"元記事URL: {url}" if url else "",
        f"著者: {author}" if author else "",
        "",
        "入力本文:",
        content,
    ]
    input_section = "\n".join(part for part in input_parts if part or part == "")
    prompt = f"{ARTICLE_SUMMARIZATION_PROMPT}\n\n{input_section}"
    return llm.comment_on_text(content, prompt)


SUMMARY_QUALITY_EVALUATION_PROMPT = dedent(
    """
    あなたは省エネ・環境対策に関する記事要約の品質を評価する専門家です。
    以下の要約リストを評価し、各要約に1〜10のスコアを付けてください。

    ## 評価基準
    1. 情報の具体性・実用性 (省エネ施策として役立つ具体的な情報があるか)
    2. 内容の新規性・独自性 (一般論ではなく独自の知見があるか)
    3. 要約の読みやすさ・まとまり (文章として整っているか)
    4. 建物タイプへの関連性 (対象建物で実践可能な内容か)

    ## 出力形式
    必ず以下のJSON配列形式で出力してください。他のテキストは含めないでください。

    ```json
    [
      {"index": 0, "score": 8, "reason": "評価理由 (50文字以内)"},
      {"index": 1, "score": 6, "reason": "評価理由 (50文字以内)"},
      ...
    ]
    ```

    ## 要約リスト
    """
)


@dataclass
class SummaryQualityScore:
    """要約の品質スコア"""

    index: int
    score: int
    reason: str


def evaluate_summary_quality(
    summaries: list[dict[str, str]],
    llm: GraphLanguageModel,
    top_n: int = 3,
) -> list[int]:
    """
    複数の要約の品質を評価し、上位N件のインデックスを返す。

    Args:
        summaries: 要約のリスト。各要素は {"theme": str, "title": str, "content": str} を含む
        llm: Geminiクライアント
        top_n: 返す上位件数

    Returns:
        list[int]: 品質スコア上位N件の要約のインデックス (元のリストにおける位置)
    """
    if len(summaries) <= top_n:
        return list(range(len(summaries)))

    # 要約リストをテキスト化
    summary_texts = []
    for i, s in enumerate(summaries):
        summary_texts.append(
            f"[{i}] テーマ: {s.get('theme', '')}\n"
            f"タイトル: {s.get('title', '')}\n"
            f"要約: {s.get('content', '')}"
        )

    input_text = "\n\n---\n\n".join(summary_texts)
    prompt = f"{SUMMARY_QUALITY_EVALUATION_PROMPT}\n{input_text}"

    try:
        response = llm.comment_on_text(input_text, prompt)

        # JSONパース
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            scores_data = json.loads(json_match.group())
            # スコアでソートして上位N件のインデックスを取得
            scores_data.sort(key=lambda x: x.get("score", 0), reverse=True)
            return [item["index"] for item in scores_data[:top_n]]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # パース失敗時は最初のN件を返す
    return list(range(top_n))


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

# 建物タイプ一覧
BUILDING_TYPES = ["オフィス", "自治体施設", "工場", "介護福祉施設"]

# 記事適切性判定用モデル (高速・安価)
FLASH_MODEL_NAME = "gemini-2.5-flash"

# 適切性判定プロンプト
ARTICLE_RELEVANCE_PROMPT = dedent(
    """
    あなたは省エネ・環境関連の記事の適切性を判定するエキスパートです。

    以下の記事が、指定された建物タイプの担当者にとって
    有益で適切な内容かどうかを判定してください。

    #建物タイプ
    {building_type}

    #判定基準
    1. 建物タイプに関連する省エネ・環境対策の情報が含まれている
    2. 実用的なアクションや知見が得られる
    3. 広告・宣伝目的ではなく、情報提供を目的としている
    4. 日本国内で適用可能な内容である

    #出力形式
    以下のJSON形式のみで出力してください(他の文章は不要):
    {{"is_relevant": true または false, "reason": "判定理由を50文字以内で"}}

    #入力記事
    タイトル: {title}
    URL: {url}
    本文:
    {content}
    """
)


@dataclass
class ArticleRelevanceResult:
    """記事適切性判定結果"""

    is_relevant: bool
    reason: str
    url: str
    title: str


@dataclass
class ArticleCollectionResult:
    """記事収集結果"""

    articles: list[ArticleFetchResult]
    relevance_results: list[ArticleRelevanceResult]
    total_searched: int
    total_judged: int
    stopped_reason: str  # "target_reached" / "max_attempts" / "no_more_results"


def judge_article_relevance(
    content: str,
    title: str,
    url: str,
    building_type: str,
    llm: GraphLanguageModel,
) -> ArticleRelevanceResult:
    """
    記事が対象建物タイプにとって適切かを判定する。

    Args:
        content: 記事本文
        title: 記事タイトル
        url: 記事URL
        building_type: 建物タイプ (オフィス/自治体施設/工場/介護福祉施設)
        llm: Gemini Flash クライアント

    Returns:
        ArticleRelevanceResult: 判定結果
    """
    # コンテンツを適切な長さに切り詰め (判定にはフル本文不要、コスト削減)
    truncated_content = content[:3000] if len(content) > 3000 else content

    prompt = ARTICLE_RELEVANCE_PROMPT.format(
        building_type=building_type,
        title=title,
        url=url,
        content=truncated_content,
    )

    # プロンプトに記事内容が既に含まれているため、textは空文字列を渡す
    # (二重送信によるトークン使用量の増加を防ぐ)
    response = llm.comment_on_text("", prompt)

    # JSON解析
    try:
        # コードフェンスを除去
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # 最初と最後の```行を除去
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)
        return ArticleRelevanceResult(
            is_relevant=bool(data.get("is_relevant", False)),
            reason=str(data.get("reason", "")),
            url=url,
            title=title,
        )
    except (json.JSONDecodeError, KeyError):
        # パース失敗時はフォールスルー (適切と判定して収集を継続)
        return ArticleRelevanceResult(
            is_relevant=True,
            reason="判定不能のため通過",
            url=url,
            title=title,
        )


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


@dataclass
class ArticleProgressInfo:
    """記事収集の進捗情報"""

    event: str  # "query_start" / "article_found" / "article_judged"
    query: str  # 現在の検索クエリ
    title: str  # 記事タイトル
    url: str  # 記事URL
    is_relevant: bool | None  # 適切性判定結果 (None=未判定)
    reason: str  # 判定理由
    total_searched: int  # 検索した記事数
    total_judged: int  # 判定した記事数
    total_collected: int  # 収集した記事数
    target_count: int  # 目標記事数


def collect_relevant_articles(
    theme: str,
    building_type: str,
    flash_llm: GraphLanguageModel,
    target_count: int = 20,
    max_search_attempts: int = 10,
    progress_callback: Callable[[ArticleProgressInfo], None] | None = None,
) -> ArticleCollectionResult:
    """
    適切な記事を指定件数集まるまで収集する。

    Args:
        theme: 検索テーマ
        building_type: 建物タイプ
        flash_llm: Gemini Flash クライアント (判定用)
        target_count: 目標記事数 (デフォルト20)
        max_search_attempts: 最大検索回数 (無限ループ防止)
        progress_callback: 進捗コールバック (ArticleProgressInfo)

    Returns:
        ArticleCollectionResult: 収集結果
    """
    collected_articles: list[ArticleFetchResult] = []
    all_relevance_results: list[ArticleRelevanceResult] = []
    seen_urls: set[str] = set()
    total_searched = 0
    total_judged = 0

    # 建物タイプをコンテキストとして使用
    building_context = building_type

    # テーマに対応する検索クエリを取得
    search_queries = THEME_SEARCH_QUERIES.get(theme, [f"{theme} コラム 記事"])

    # 各クエリで順番に検索
    for attempt, query in enumerate(search_queries):
        if attempt >= max_search_attempts:
            break

        if len(collected_articles) >= target_count:
            break

        # ターゲット (建物タイプ) をクエリに追加
        full_query = f"{building_context} {query}"

        # クエリ開始を通知
        if progress_callback:
            progress_callback(
                ArticleProgressInfo(
                    event="query_start",
                    query=full_query,
                    title="",
                    url="",
                    is_relevant=None,
                    reason="",
                    total_searched=total_searched,
                    total_judged=total_judged,
                    total_collected=len(collected_articles),
                    target_count=target_count,
                )
            )

        try:
            ddgs = DDGS()
            results = ddgs.text(
                full_query,
                region="jp-jp",
                max_results=20,
            )
        except Exception:
            continue

        for r in results:
            url = r.get("href", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            total_searched += 1
            title = r.get("title", "")

            # 記事発見を通知
            if progress_callback:
                progress_callback(
                    ArticleProgressInfo(
                        event="article_found",
                        query=full_query,
                        title=title,
                        url=url,
                        is_relevant=None,
                        reason="取得中...",
                        total_searched=total_searched,
                        total_judged=total_judged,
                        total_collected=len(collected_articles),
                        target_count=target_count,
                    )
                )

            # URLパターンフィルタリング
            if not is_likely_article_url(url):
                continue

            # ページ取得
            try:
                fetch_result = fetch_page_content(url, timeout=10)
            except Exception:
                continue

            # コンテンツ長チェック (最小300文字)
            if len(fetch_result.content) < 300:
                continue

            total_judged += 1

            # 適切性判定
            relevance = judge_article_relevance(
                content=fetch_result.content,
                title=fetch_result.title or title,
                url=url,
                building_type=building_context,
                llm=flash_llm,
            )
            all_relevance_results.append(relevance)

            # 判定結果を通知
            if progress_callback:
                progress_callback(
                    ArticleProgressInfo(
                        event="article_judged",
                        query=full_query,
                        title=fetch_result.title or title,
                        url=url,
                        is_relevant=relevance.is_relevant,
                        reason=relevance.reason,
                        total_searched=total_searched,
                        total_judged=total_judged,
                        total_collected=(
                            len(collected_articles) + 1
                            if relevance.is_relevant
                            else len(collected_articles)
                        ),
                        target_count=target_count,
                    )
                )

            if relevance.is_relevant:
                collected_articles.append(fetch_result)

                if len(collected_articles) >= target_count:
                    return ArticleCollectionResult(
                        articles=collected_articles,
                        relevance_results=all_relevance_results,
                        total_searched=total_searched,
                        total_judged=total_judged,
                        stopped_reason="target_reached",
                    )

    # ループ終了 (目標未達またはクエリ枯渇)
    stopped_reason = "max_attempts" if total_searched > 0 else "no_more_results"
    return ArticleCollectionResult(
        articles=collected_articles,
        relevance_results=all_relevance_results,
        total_searched=total_searched,
        total_judged=total_judged,
        stopped_reason=stopped_reason,
    )
