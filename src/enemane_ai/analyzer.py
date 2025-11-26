from __future__ import annotations

import csv
import os
from dataclasses import dataclass
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

PRESET_PROMPT = dedent(
    """
    あなたは電力管理の上級アナリストです。入力は【グラフ画像】【グラフ画像の元データ】です。
    任意で【日別気温CSV】、【前年同月の月報CSV】(ファイル名例: 月報202410.csv)です。

    画像内の数値・凡例・表・注記、およびCSVデータを根拠として、各グラフごとに
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
    ■ 出力構成ルール(重要: 必ず守ること)

    **1. T1(最大電力)とT2(電力量)の完全分離**
      - 画像内でこれらが隣接していたり、同じグラフエリアに描画されていても、
        **必ず別々の見出し(###)を立てて、個別にコメントを生成**する。
      - 1つの項目内で「最大電力は○○、電力量は○○」とまとめて記述することを禁ずる。

    **2. T3(回路別内訳)の2段階記述(CSV連携)**
      - **現状(画像)**: 画像内のデータに基づき、シェアや順位、特徴を記述。
      - **前年比較(CSV)**: 入力に「月報yyyyMM.csv」がある場合は以下を行う。
        1. CSV各行の日付列を合計し、回路の**月間合計値(kWh)**を算出。
        2. 画像(T3)上位回路名とCSV回路名を照合し、一致する回路の前年同月比を算出。
        3. 「前年比 +○○%」や「前年同月 ○○ kWh から減少」と明確に記述。

    ========================
    ■ 出力(Markdown。各グラフ=見出し+1〜3文。長文禁止)
    - 見出しは画像内タイトルをそのまま使う。無ければ「グラフ1」「グラフ2」等。
    - 文章構成: 事実(数値)→短い仮説→一言アクションの3要素を含めること。
    - 文体: **各要素を接続詞で滑らかに繋ぎ、自然な日本語の文章にする**こと。
      「(事実)」「(仮説)」といったラベル表記は**禁止**。
    - 共通X軸を用いた場合、先頭文の末尾に(上段は下段と同一X軸)と添える。
    - 例: 「比較不可(集約)」「読取不可」「整合に注意」などは文脈に組み込むか、
      括弧で短く注記。

    【出力フォーマット例】
    ### 最大電力[kW](上段は下段と同一X軸)
      今月は **86.9 kW** で前年7月 **90.9 kW** を下回り、12か月内のピーク
      **100.5 kW**(集約区間)より低水準で推移しました。夏期ピークの同時起動が抑えられた
      可能性があるため、引き続きピーク帯の段階投入と需要監視アラート90%の設定を推奨します。

    ### 電力量[kWh]
      今月は **18,513 kWh** となり、先月 **14,104 kWh** 比 **+31.3%**、前年7月
      **19,034 kWh** 比 **-2.7%** で着地しました。冷房需要は高いものの前年より抑制傾向にあると
      考えられるため、上位回路の運用点検と昼休みの不要負荷停止を徹底してください。

    ### 回路別内訳(CSV連携あり)
      - **現状**: 上位5回路で **85%** を占め、最大は「3F事務所・電灯」の
        **2,606 kWh(22.2%)** でした。常時負荷が高くピーク・ベース双方に寄与している
        可能性があるため、スケジュール・在室連動の適用をご検討ください。
      - **前年比較**: 前年同月(CSV算出)と比較すると、「3F事務所・電灯」は
        **2,800 kWh** から **-6.9%** 減少しました。照明のLED化または消灯励行の効果が出ている可能性が
        あるため、継続的な周知をお願いします。

    (以降、検出した各グラフについて同様に簡潔な1〜3文で記述)

    ========================
    ■ 禁則
    - 画像外の推測数値や固有情報(気象・契約電力等)は書かない。
    - kWとkWhを混同しない。根拠が曖昧な比較は行わない(「比較不可」と明記)。
    """
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
CSV_EXTENSIONS = {".csv"}
DEFAULT_MODEL_NAME = "gemini-3-pro-preview"


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
