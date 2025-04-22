import streamlit as st
import whisper
import tempfile
import os
import re

# UIタイトル
st.title("音声文字起こしアプリ")

# Whisperモデル選択
model_size = st.selectbox("モデルを選択：", ["tiny", "base", "small", "medium", "large"])

# 言語選択（オプション）
language = st.selectbox("音声の言語を選択（自動検出の場合は空欄）：", ["", "英語", "中国語", "日本語"])

# 言語コードの変換
language_dict = {
    "": None,
    "英語": "en",
    "中国語": "zh",
    "日本語": "ja"
}

# 音声ファイルアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（mp3, wav, m4a）", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Whisperモデル読み込み
    st.info("モデルを読み込み中...")
    model = whisper.load_model(model_size)

    # 文字起こし処理
    st.info("音声を文字起こし中です...")
    transcribe_options = {}
    if language_dict[language] is not None:
        transcribe_options["language"] = language_dict[language]

    result = model.transcribe(tmp_path, **transcribe_options)
    sentences = re.split(r'(?<=[。．.!?！？])\s*', result["text"].strip())
    formatted_text = "\n".join(sentences)

    # 文字起こし結果表示
    st.subheader("📝 文字起こし結果")
    st.text_area("文単位での文字起こし", formatted_text, height=300)

    # テキストファイルとしてダウンロード
    st.download_button(
        label="📥 テキストをダウンロード",
        data=formatted_text,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # 言語検出結果の表示（自動検出の場合）
    if language == "":
        st.info(f"検出された言語: {result['language']} (信頼度: {result['language_probability']:.1%})")

    os.remove(tmp_path)
