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
                status.update(label="分析中...", state="running")
                try:
                    analyzed = analyze_files(stored_paths, prompt=prompt, llm=llm)
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
