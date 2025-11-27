from __future__ import annotations

import csv
import hmac
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterable

import streamlit as st

from enemane_ai.analyzer import (
    OUTPUT_FORMAT_INSTRUCTION,
    PRESET_PROMPT,
    GeminiGraphLanguageModel,
    GraphLanguageModel,
    MonthlyReportData,
    MonthlyTemperatureSummary,
    analyze_image,
    build_supplementary_context,
    collect_graph_entries,
    parse_monthly_report_csv,
    parse_temperature_csv_for_comparison,
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass
class AnalyzedGraph:
    label: str
    comment: str
    image_title: str | None = None
    item_name: str | None = None
    image_data: bytes | None = None
    text: str | None = None


@dataclass
class ResultRow:
    image_title: str
    item_name: str
    comment: str


@dataclass
class OutputRow:
    """target_data.csv形式の1行"""

    graph_name: str  # 対応するグラフ名
    item_name: str  # 項目名
    ai_comment: str  # 生成するAIコメント


def save_uploads_to_temp(files: Iterable["UploadedFile"], tmpdir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    for file in files:
        destination = tmpdir / file.name
        destination.write_bytes(file.getvalue())
        saved_paths.append(destination)
    return saved_paths


def analyze_files(
    file_paths: list[Path],
    prompt: str,
    llm: GraphLanguageModel | None = None,
) -> list[AnalyzedGraph]:
    entries = collect_graph_entries(file_paths)
    analyzed: list[AnalyzedGraph] = []
    for entry in entries:
        if entry.image is not None:
            comment_prompt = build_image_prompt(prompt)
            raw_comment = analyze_image(entry.image, prompt=comment_prompt, llm=llm)
            image_title, item_name, comment = parse_structured_comment(
                raw_comment, fallback_label=entry.display_label
            )
            buffer = BytesIO()
            entry.image.save(buffer, format="PNG")
            analyzed.append(
                AnalyzedGraph(
                    label=entry.display_label,
                    comment=comment,
                    image_title=image_title,
                    item_name=item_name,
                    image_data=buffer.getvalue(),
                )
            )
            continue

        if entry.text is not None:
            analyzed.append(
                AnalyzedGraph(
                    label=entry.display_label,
                    comment="",
                    image_title=entry.display_label,
                    item_name="テキスト/CSV",
                    text=entry.text,
                )
            )
    return analyzed


def resolve_gemini_client() -> GeminiGraphLanguageModel | None:
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY が見つかりません。"
            " .streamlit/secrets.toml に設定してください (環境変数 GEMINI_API_KEY でも可)。"
        )
        return None

    try:
        return GeminiGraphLanguageModel(api_key=api_key)
    except Exception as exc:
        st.error(f"Gemini クライアント生成に失敗しました: {exc}")
        return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return text


def strip_markdown(text: str) -> str:
    # Bold: **text** or __text__
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    # Italic: *text* or _text_
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Inline code: `text` -> text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Headers: # text -> text (remove leading # and space)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # List markers: - text, * text -> text (remove leading marker and space)
    text = re.sub(r"^[\*\-]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_structured_comment(raw_comment: str, fallback_label: str) -> tuple[str, str, str]:
    cleaned = strip_code_fence(raw_comment)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback_label, fallback_label, raw_comment

    if not isinstance(data, dict):
        return fallback_label, fallback_label, raw_comment

    image_title = str(data.get("image_title") or fallback_label)
    item_name = str(data.get("item_name") or fallback_label)
    comment = str(data.get("comment") or raw_comment)
    return image_title, item_name, comment


def parse_multi_item_response(
    raw_response: str,
    fallback_graph_name: str,
) -> list[tuple[str, str, str]]:
    """
    JSON配列レスポンスを(graph_name, item_name, comment)のリストに変換。

    レスポンスがJSON配列の場合は複数項目を返し、
    単一オブジェクトまたはパースエラーの場合はフォールバック。
    """
    cleaned = strip_code_fence(raw_response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # JSONパースに失敗した場合、単一項目として返す
        return [(fallback_graph_name, "", raw_response)]

    if isinstance(data, list):
        results: list[tuple[str, str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            graph_name = str(item.get("graph_name") or fallback_graph_name)
            item_name = str(item.get("item_name") or "")
            comment = str(item.get("comment") or "")
            results.append((graph_name, item_name, comment))
        if results:
            return results
        # 空の配列の場合はフォールバック
        return [(fallback_graph_name, "", raw_response)]

    if isinstance(data, dict):
        # 単一オブジェクトの場合(後方互換性)
        graph_name = str(data.get("graph_name") or data.get("image_title") or fallback_graph_name)
        item_name = str(data.get("item_name") or "")
        comment = str(data.get("comment") or "")
        return [(graph_name, item_name, comment)]

    return [(fallback_graph_name, "", raw_response)]


def build_image_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "以下の3つの値を含む JSON 文字列だけを返してください。\n"
        '{ "image_title": "画像上部のタイトル", "item_name": "グラフ上の名称", "comment": "1) トレンド 2) 含意 3) 注意点" }\n'  # noqa: E501
        "日本語で簡潔に。"
    )


def build_result_rows(analyzed: list[AnalyzedGraph]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for item in analyzed:
        if item.image_data is None:
            continue  # CSV/テキストは表から除外
        image_title = item.image_title or item.label
        item_name = item.item_name
        if item_name is None:
            item_name = "グラフ画像" if item.image_data is not None else "テキスト/CSV"
        comment = item.comment
        if not comment and item.text is not None:
            comment = "CSV/テキストはコメント生成を省略しています。"

        # テーブル出力用にMarkdownを除去してプレーンテキスト化
        rows.append(
            ResultRow(
                image_title=strip_markdown(image_title),
                item_name=strip_markdown(item_name),
                comment=strip_markdown(comment),
            )
        )
    return rows


def export_table_csv(rows: list[ResultRow]) -> bytes:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["画像内タイトル", "項目名", "AIで生成したコメント"])
    for row in rows:
        writer.writerow([row.image_title, row.item_name, row.comment])
    return buffer.getvalue().encode("utf-8")


def export_target_format_csv(results: list[OutputRow]) -> bytes:
    """target_data.csv形式でエクスポート(BOM付きUTF-8)。"""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["対応するグラフ名", "項目名", "生成するAIコメント"])
    for row in results:
        writer.writerow([row.graph_name, row.item_name, row.ai_comment])
    # BOM付きUTF-8でExcel互換性を確保
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def analyze_graphs_with_context(
    graph_paths: list[Path],
    monthly_report: MonthlyReportData | None,
    temperature: tuple[MonthlyTemperatureSummary, MonthlyTemperatureSummary] | None,
    base_prompt: str,
    llm: GraphLanguageModel,
) -> list[OutputRow]:
    """
    グラフ画像を補助データのコンテキスト付きで分析し、OutputRowのリストを返す。
    """
    # 補助データコンテキストを構築
    context = build_supplementary_context(monthly_report, temperature)

    # プロンプトを構築
    full_prompt = base_prompt
    if context:
        full_prompt = f"{base_prompt}\n\n{context}"
    full_prompt = f"{full_prompt}\n\n{OUTPUT_FORMAT_INSTRUCTION}"

    all_results: list[OutputRow] = []
    entries = collect_graph_entries(graph_paths)

    for entry in entries:
        if entry.image is None:
            continue

        raw_response = analyze_image(entry.image, prompt=full_prompt, llm=llm)
        items = parse_multi_item_response(raw_response, entry.display_label)

        for graph_name, item_name, comment in items:
            all_results.append(
                OutputRow(
                    graph_name=strip_markdown(graph_name),
                    item_name=strip_markdown(item_name),
                    ai_comment=strip_markdown(comment),
                )
            )

    return all_results


def check_password() -> bool:
    """Basic認証を行い、認証成功ならTrueを返す。"""

    def password_entered() -> None:
        """パスワード入力時のコールバック。"""
        if hmac.compare_digest(
            st.session_state["username"], st.secrets.auth.username
        ) and hmac.compare_digest(st.session_state["password"], st.secrets.auth.password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("ユーザー名", key="username")
    st.text_input("パスワード", type="password", key="password")
    st.button("ログイン", on_click=password_entered)

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("ユーザー名またはパスワードが正しくありません")

    return False


def main() -> None:
    st.set_page_config(page_title="Graph Insight Uploader", layout="wide")

    if not check_password():
        return

    st.title("グラフ分析ダッシュボード")
    st.caption(
        "グラフ画像をアップロードし、月報CSV・気温データと組み合わせて" "AIコメントを生成します。"
    )

    # ファイルアップローダーを3つに分離
    st.subheader("1. グラフ画像")
    graph_files = st.file_uploader(
        "分析したいグラフ画像をアップロードしてください",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
        key="graph_images",
    )

    st.subheader("2. 補助データ (オプション)")
    col1, col2 = st.columns(2)

    with col1:
        monthly_report_file = st.file_uploader(
            "月報CSV (前年同月データ)",
            type=["csv"],
            accept_multiple_files=False,
            key="monthly_report",
            help="前年同月の電力使用量データ。前年比較に使用します。",
        )

    with col2:
        temperature_file = st.file_uploader(
            "気温データCSV (前年・当年)",
            type=["csv"],
            accept_multiple_files=False,
            key="temperature",
            help="前年と当年の気温データ。気温との相関分析に使用します。",
        )

    st.subheader("3. 追加指示 (オプション)")
    additional_instructions = st.text_area(
        "追加の指示",
        placeholder="例: 重要なトレンドのみを箇条書きでまとめてください。",
        height=80,
    )

    prompt = PRESET_PROMPT
    if additional_instructions.strip():
        prompt = f"{PRESET_PROMPT}\n\n{additional_instructions.strip()}"

    if not graph_files:
        st.info("グラフ画像を選択してください。")
        return

    if st.button("分析を実行", type="primary"):
        llm = resolve_gemini_client()
        if llm is None:
            return

        with st.status("ファイルを準備しています...", expanded=False) as status:
            with TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)

                # グラフ画像を保存
                graph_paths = save_uploads_to_temp(graph_files, tmpdir)

                # 月報CSVをパース
                monthly_report: MonthlyReportData | None = None
                if monthly_report_file:
                    report_path = tmpdir / monthly_report_file.name
                    report_path.write_bytes(monthly_report_file.getvalue())
                    try:
                        monthly_report = parse_monthly_report_csv(report_path)
                        st.info(
                            f"月報データ読み込み: {monthly_report.month_label}, "
                            f"月間電力使用量: {monthly_report.total_power_monthly:,.0f} kWh"
                        )
                    except Exception as exc:
                        st.warning(f"月報CSVの読み込みに失敗しました: {exc}")

                # 気温データをパース
                temperature: tuple[MonthlyTemperatureSummary, MonthlyTemperatureSummary] | None = (
                    None
                )
                if temperature_file:
                    temp_path = tmpdir / temperature_file.name
                    temp_path.write_bytes(temperature_file.getvalue())
                    try:
                        temperature = parse_temperature_csv_for_comparison(temp_path)
                        prev, curr = temperature
                        st.info(
                            f"気温データ読み込み: {prev.year_month} → {curr.year_month}, "
                            f"平均気温差: {curr.avg_temp - prev.avg_temp:+.1f}℃"
                        )
                    except Exception as exc:
                        st.warning(f"気温CSVの読み込みに失敗しました: {exc}")

                status.update(label="分析中...", state="running")
                try:
                    results = analyze_graphs_with_context(
                        graph_paths=graph_paths,
                        monthly_report=monthly_report,
                        temperature=temperature,
                        base_prompt=prompt,
                        llm=llm,
                    )
                except Exception as exc:
                    status.update(label="失敗", state="error")
                    st.error(f"分析に失敗しました: {exc}")
                    return
            status.update(label="完了", state="complete")

        st.subheader("結果")

        if not results:
            st.warning("分析結果がありません。")
            return

        # テーブル表示
        st.markdown("#### テーブル出力")
        table_data = [
            {
                "対応するグラフ名": row.graph_name,
                "項目名": row.item_name,
                "生成するAIコメント": row.ai_comment,
            }
            for row in results
        ]
        st.table(table_data)

        # CSVダウンロード
        st.download_button(
            "CSVをダウンロード",
            data=export_target_format_csv(results),
            file_name="analysis_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
