from __future__ import annotations

import csv
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
    PRESET_PROMPT,
    GeminiGraphLanguageModel,
    GraphLanguageModel,
    analyze_image,
    build_comparison_context,
    collect_graph_entries,
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
    comparison_context: str | None = None,
) -> list[AnalyzedGraph]:
    entries = collect_graph_entries(file_paths)
    analyzed: list[AnalyzedGraph] = []
    for entry in entries:
        if entry.image is not None:
            comment_prompt = build_image_prompt(prompt, comparison_context=comparison_context)
            raw_comment = analyze_image(entry.image, prompt=comment_prompt, llm=llm)
            image_title, item_name, comment = parse_structured_comment(
                raw_comment, fallback_label=entry.display_label
            )
            buffer = BytesIO()
            entry.image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            # LLMが返したimage_titleのみで判定 (ファイル名では判定しない)
            monthly_chart = _is_monthly_energy_chart(image_title)
            power_usage_status = _is_power_usage_status_chart(image_title)

            if power_usage_status:
                # 「電力使用状況」のグラフ: 最大電力[kW]と電力使用量[kWh]の2つを生成
                for item_info in [
                    ("最大電力[kW]", "最大電力[kW](前年同月比較・ピーク値の推移)"),
                    ("電力使用量[kWh]", "電力使用量[kWh](前年同月比較・消費量の推移)"),
                ]:
                    forced_item, focus_desc = item_info
                    item_prompt = build_image_prompt(
                        prompt,
                        comparison_context=comparison_context,
                        forced_item_name=forced_item,
                        focus=focus_desc,
                    )
                    item_raw = analyze_image(entry.image, prompt=item_prompt, llm=llm)
                    it_title, it_item, it_comment = parse_structured_comment(
                        item_raw, fallback_label=image_title
                    )
                    analyzed.append(
                        AnalyzedGraph(
                            label=entry.display_label,
                            comment=it_comment,
                            image_title=it_title or image_title,
                            item_name=forced_item,
                            image_data=image_bytes,
                        )
                    )
            elif monthly_chart:
                # 「月間電力使用量」のグラフ: 前年比較と回路別内訳の2つを生成
                # 前年比較: 専用プロンプトで前年同月との比較にフォーカス
                yoy_prompt = build_image_prompt(
                    prompt,
                    comparison_context=comparison_context,
                    forced_item_name="前年比較",
                    focus="前年同月との電力使用量[kWh]の比較(増減率・要因分析)",
                )
                yoy_raw = analyze_image(entry.image, prompt=yoy_prompt, llm=llm)
                yoy_title, yoy_item, yoy_comment = parse_structured_comment(
                    yoy_raw, fallback_label=image_title
                )
                analyzed.append(
                    AnalyzedGraph(
                        label=entry.display_label,
                        comment=yoy_comment,
                        image_title=yoy_title or image_title,
                        item_name="前年比較",
                        image_data=image_bytes,
                    )
                )
                # 回路別内訳: 上位回路の構成比にフォーカス (前年比較は上で行うのでここでは不要)
                breakdown_prompt = build_image_prompt(
                    prompt,
                    comparison_context=comparison_context,
                    forced_item_name="回路別内訳",
                    focus="回路別内訳(上位回路のシェア・その他構成・カバー率のみ、前年比較は不要)",
                )
                breakdown_raw = analyze_image(entry.image, prompt=breakdown_prompt, llm=llm)
                br_title, br_item, br_comment = parse_structured_comment(
                    breakdown_raw, fallback_label=image_title
                )
                analyzed.append(
                    AnalyzedGraph(
                        label=entry.display_label,
                        comment=br_comment,
                        image_title=br_title or image_title,
                        item_name="回路別内訳",
                        image_data=image_bytes,
                    )
                )
            else:
                # 通常のグラフ
                analyzed.append(
                    AnalyzedGraph(
                        label=entry.display_label,
                        comment=comment,
                        image_title=image_title,
                        item_name=item_name,
                        image_data=image_bytes,
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


def _extract_json_dict(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for candidate in (strip_code_fence(text), text):
        for match in re.finditer(r"{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _strip_first_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            _, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        return (text[: match.start()] + text[end:]).strip()
    return text.strip()


def parse_structured_comment(raw_comment: str, fallback_label: str) -> tuple[str, str, str]:
    data = _extract_json_dict(raw_comment)
    image_title = fallback_label
    item_name = fallback_label
    if isinstance(data, dict):
        image_title = str(data.get("image_title") or fallback_label)
        item_name = str(data.get("item_name") or fallback_label)
        comment_value = data.get("comment")
        if comment_value:
            return image_title, item_name, str(comment_value)

    cleaned = strip_code_fence(raw_comment)
    cleaned = _strip_first_json_object(cleaned)
    return image_title, item_name, cleaned or fallback_label


TARGET_MONTHLY_ENERGY_TITLE = "電力使用量 [kWh] について <月間電力使用量>"
MONTHLY_ENERGY_KEYWORDS = ["月間電力使用量"]
POWER_USAGE_STATUS_KEYWORDS = ["電力使用状況"]


def _is_monthly_energy_chart(title: str | None) -> bool:
    """月間電力使用量グラフかどうかを判定する。

    「月間電力使用量」というキーワードを含む場合にTrueを返す。
    「電力使用状況」など他のグラフとは区別される。
    """
    if not title:
        return False
    return any(keyword in title for keyword in MONTHLY_ENERGY_KEYWORDS)


def _is_power_usage_status_chart(title: str | None) -> bool:
    """電力使用状況グラフかどうかを判定する。

    「電力使用状況」というキーワードを含む場合にTrueを返す。
    """
    if not title:
        return False
    return any(keyword in title for keyword in POWER_USAGE_STATUS_KEYWORDS)


def build_image_prompt(
    base_prompt: str,
    comparison_context: str | None = None,
    forced_item_name: str | None = None,
    focus: str | None = None,
) -> str:
    comparison_note = ""
    if comparison_context:
        comparison_note = f"\n\n【前年比較用データ】\n{comparison_context}"

    special_notes = [
        f"- 画像タイトルが「{TARGET_MONTHLY_ENERGY_TITLE}」の場合、"
        "必ず前年同月との比較を含めてください。",
        '- その場合、JSON 出力の "item_name" は必ず「前年比較」に設定してください。',
        "- 前年データは「月報YYYYMM.csv」などのファイル名で渡されます。"
        "YYYYMMから対象月と前年同月を推定して比較してください。",
    ]
    if forced_item_name:
        special_notes.append(
            f'- この出力では JSON の "item_name" を「{forced_item_name}」に固定してください。'
        )
    if focus:
        special_notes.append(f"- コメントは{focus}の観点で簡潔にまとめてください。")

    return (
        f"{base_prompt}"
        "\n\n【特記事項】\n"
        f"{chr(10).join(special_notes)}\n"
        f"{comparison_note}\n\n"
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
    # Shift_JIS (CP932) でエンコード (Windows Excelで確実に開ける)
    return buffer.getvalue().encode("cp932", errors="replace")


def main() -> None:
    st.set_page_config(page_title="Graph Insight Uploader", layout="wide")
    st.title("グラフ/PDF/気温CSVの分析ダッシュボード")
    st.caption(
        "複数のグラフ画像やPDF、気温データのCSVをまとめてアップロードし、"
        "あらかじめ決めたプロンプトで分析します。"
    )

    st.caption("分析プロンプトは固定です。追加で伝えたいことがあれば下記に記入してください。")
    additional_instructions = st.text_area(
        "追加の指示 (任意)",
        placeholder="例: 重要なトレンドのみを箇条書きでまとめてください。",
        height=120,
    )
    prompt = PRESET_PROMPT
    if additional_instructions.strip():
        prompt = f"{PRESET_PROMPT}\n\n{additional_instructions.strip()}"
    uploaded_files = st.file_uploader(
        "分析したいファイルをまとめてアップロードしてください",
        type=["png", "jpg", "jpeg", "bmp", "gif", "tiff", "pdf", "csv"],
        accept_multiple_files=True,
    )
    st.caption("CSV は「日付,気温」の2列を想定しています (例: 2024-01-01,12.3)。")

    comparison_files = st.file_uploader(
        "前年比較用のデータをアップロードしてください (任意・CSV)",
        type=["csv"],
        accept_multiple_files=True,
        key="comparison_files",
    )
    st.caption(
        "前年同月などの比較用CSVを追加すると、コメント生成時に参照します。"
        " 例: 月別電力量や最大電力の履歴。"
    )

    if not uploaded_files:
        st.info("画像、PDF、または気温CSVファイルを選択してください。")
        return

    if st.button("分析を実行", type="primary"):
        llm = resolve_gemini_client()
        if llm is None:
            return

        with st.status("ファイルを準備しています...", expanded=False) as status:
            with TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                stored_paths = save_uploads_to_temp(uploaded_files, tmpdir)
                comparison_paths = save_uploads_to_temp(comparison_files or [], tmpdir)
                status.update(label="分析中...", state="running")
                try:
                    comparison_context = None
                    if comparison_paths:
                        comparison_context = build_comparison_context(comparison_paths)
                    analyzed = analyze_files(
                        stored_paths,
                        prompt=prompt,
                        llm=llm,
                        comparison_context=comparison_context,
                    )
                except Exception as exc:
                    status.update(label="失敗", state="error")
                    st.error(f"ファイルの処理に失敗しました: {exc}")
                    return
            status.update(label="完了", state="complete")

        st.subheader("結果")
        result_rows = build_result_rows(analyzed)
        st.markdown("#### テーブル出力")

        # st.table 用のデータを作成(テキストを全て表示するため)
        table_data = [
            {
                "画像内タイトル": row.image_title,
                "項目名": row.item_name,
                "AIで生成したコメント": row.comment,
            }
            for row in result_rows
        ]
        st.table(table_data)

        st.download_button(
            "CSVをダウンロード",
            data=export_table_csv(result_rows),
            file_name="analysis_results.csv",
            mime="text/csv",
        )

        for item in analyzed:
            if item.image_data is not None:
                col_image, col_comment = st.columns([1, 2], gap="large")
                with col_image:
                    st.image(
                        item.image_data, caption=item.image_title or item.label, width="stretch"
                    )
                with col_comment:
                    display_title = item.image_title or item.label
                    st.markdown(f"**{display_title}**")
                    if item.item_name:
                        st.caption(f"項目名: {item.item_name}")
                    st.markdown(item.comment)
            else:
                st.markdown(f"**{item.image_title or item.label}**")
                if item.text is not None:
                    st.caption("CSV はコメントやデータ表示を省略しています。")
                elif item.comment:
                    st.markdown(item.comment)


if __name__ == "__main__":
    main()
