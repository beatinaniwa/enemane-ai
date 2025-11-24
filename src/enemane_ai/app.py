from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterable

import streamlit as st

from enemane_ai.analyzer import (
    PRESET_PROMPT,
    GeminiGraphLanguageModel,
    GraphLanguageModel,
    analyze_image,
    analyze_text,
    collect_graph_entries,
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass
class AnalyzedGraph:
    label: str
    comment: str
    image_data: bytes | None = None
    text: str | None = None


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
            comment = analyze_image(entry.image, prompt=prompt, llm=llm)
            buffer = BytesIO()
            entry.image.save(buffer, format="PNG")
            analyzed.append(
                AnalyzedGraph(
                    label=entry.display_label, comment=comment, image_data=buffer.getvalue()
                )
            )
            continue

        if entry.text is not None:
            comment = analyze_text(entry.text, prompt=prompt, llm=llm)
            analyzed.append(
                AnalyzedGraph(label=entry.display_label, comment=comment, text=entry.text)
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


def main() -> None:
    st.set_page_config(page_title="Graph Insight Uploader", layout="wide")
    st.title("グラフ/PDF/気温CSVの分析ダッシュボード")
    st.caption(
        "複数のグラフ画像やPDF、気温データのCSVをまとめてアップロードし、"
        "あらかじめ決めたプロンプトで分析します。"
    )

    prompt = st.text_area("分析プロンプト (必要に応じて編集)", PRESET_PROMPT, height=120)
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
        for item in analyzed:
            if item.image_data is not None:
                col_image, col_comment = st.columns([1, 2], gap="large")
                with col_image:
                    st.image(item.image_data, caption=item.label, use_container_width=True)
                with col_comment:
                    st.markdown(f"**{item.label}**")
                    st.markdown(item.comment)
            else:
                st.markdown(f"**{item.label}**")
                if item.text:
                    st.code(item.text, language="text")
                st.markdown(item.comment)


if __name__ == "__main__":
    main()
