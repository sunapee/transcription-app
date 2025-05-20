import streamlit as st
import whisper
import tempfile
import os
import re
import gc
import torch

# タブタイトルを設定
st.set_page_config(
    page_title="音声文字起こしアプリ",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UIタイトル
st.title("音声文字起こしアプリ")

# サイドバーにモデル情報を表示
st.sidebar.header("モデル情報")
st.sidebar.markdown("""
| モデル | 必要メモリ(概算) | 処理速度 | 精度 |
|--------|--------------|---------|------|
| tiny   | 1GB          | 最速     | 低   |
| base   | 1GB          | 速い     | 中低 |
| small  | 2GB          | 中程度    | 中   |
| medium | 5GB          | 遅い     | 中高 |
| large  | 10GB         | 最遅     | 高   |
""")

# メモリ状態表示関数
def display_memory_status():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        st.sidebar.text(f"GPU使用メモリ: {allocated:.2f}GB / {reserved:.2f}GB")
    else:
        st.sidebar.text("GPU利用不可 - CPUモードで実行中")

# Whisperモデル選択
model_size = st.selectbox("モデルを選択：", ["tiny", "base", "small", "medium", "large"])

# デバイス選択
device_option = "cpu"
if torch.cuda.is_available():
    device_option = st.radio("実行デバイス:", ["cuda", "cpu"], horizontal=True)

# 言語選択（オプション）
language = st.selectbox("音声の言語を選択（自動検出の場合は空欄）：", ["", "英語", "中国語", "日本語"])

# 言語コードの変換
language_dict = {
    "": None,
    "英語": "en",
    "中国語": "zh",
    "日本語": "ja"
}

# 文字起こしの高度なオプション
with st.expander("高度なオプション"):
    chunk_size = st.slider("音声チャンク処理サイズ(秒)", 5, 60, 30, 
                         help="長い音声ファイルを小さな区間で処理します。メモリ使用量を減らせますが、精度に影響する場合があります。")
    fp16_option = st.checkbox("半精度計算を使用(FP16)", True, 
                            help="オンにするとメモリ使用量を半減できますが、精度がわずかに低下する可能性があります。")

# 音声ファイルアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（mp3, wav, m4a）", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    with st.spinner("モデルを読み込み中..."):
        try:
            # モデルロード前にメモリ解放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # モデル読み込みオプション設定
            model_options = {
                "device": device_option
            }
            
            if fp16_option and device_option == "cuda":
                model_options["fp16"] = True
            
            # Whisperモデル読み込み
            model = whisper.load_model(model_size, **model_options)
            display_memory_status()
            
            # 文字起こし処理
            st.info("音声を文字起こし中です...")
            transcribe_options = {
                "fp16": fp16_option and device_option == "cuda"
            }
            
            if language_dict[language] is not None:
                transcribe_options["language"] = language_dict[language]
            
            result = model.transcribe(tmp_path, **transcribe_options)
            
            # メモリ解放
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 文章の整形
            sentences = re.split(r'(?<=[。．.!?！？])\s*', result["text"].strip())
            formatted_text = "\n".join(sentences)
            
            # 文字起こし結果表示
            st.subheader("📝 文字起こし結果")
            st.text_area("文単位での文字起こし", formatted_text, height=300)
            
            # テキストファイルとしてダウンロード
            st.download_button(
                label="📥 テキストをダウンロード",
                data=formatted_text,
                file_name=f"transcription_{os.path.splitext(uploaded_file.name)[0]}.txt",
                mime="text/plain"
            )
            
            # 言語検出結果の表示（自動検出の場合）
            if language == "":
                st.info(f"検出された言語: {result['language']} (信頼度: {result['language_probability']:.1%})")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e):
                st.error(f"メモリ不足エラーが発生しました。より小さいモデルを選択するか、設定を調整してください。\n\nエラー詳細: {str(e)}")
            else:
                st.error(f"エラーが発生しました: {str(e)}")
        except Exception as e:
            st.error(f"予期しないエラーが発生しました: {str(e)}")
        
        finally:
            os.remove(tmp_path)
            display_memory_status()