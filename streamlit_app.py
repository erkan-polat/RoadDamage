# streamlit_app.py
# -------------------------------------------------
# YOLOv8 Road Damage Detector — Streamlit Web UI
# - Tek/çoklu resim yükle
# - Eşik ayarları (conf, IoU, imgsz)
# - Sonuç görselleştirme + indir
# - İsteğe bağlı: data.yaml ile validation
# -------------------------------------------------
import os, io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detector (YOLOv8)", layout="wide")
os.environ["WANDB_MODE"] = "disabled"  # W&B kapat

# -------------------- Yardımcılar --------------------
@st.cache_resource
def load_model(weights_path: str):
    """Ağırlıkları dosyadan bir kez yükle ve cache'le."""
    return YOLO(weights_path)

def result_to_df(result, names):
    """Ultralytics Result -> pandas DataFrame (bbox + sınıf + skor)."""
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["class_id","class_name","conf","xmin","ymin","xmax","ymax"])
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    df = pd.DataFrame({
        "class_id": cls,
        "class_name": [names[int(i)] for i in cls],
        "conf": np.round(conf, 3),
        "xmin": xyxy[:,0], "ymin": xyxy[:,1], "xmax": xyxy[:,2], "ymax": xyxy[:,3]
    })
    return df

def bgr_to_rgb(img_np):
    """Ultralytics .plot() BGR döndürebilir; RGB'ye çevir."""
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        return img_np[:, :, ::-1]
    return img_np

# -------------------- Sidebar --------------------
st.sidebar.title("Ayarlar")

weights_path = st.sidebar.text_input("Model ağırlığı (.pt)", value="best_model.pt")
imgsz       = st.sidebar.selectbox("Image size (imgsz)", [320, 512, 640, 960, 1280], index=2)
conf        = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou         = st.sidebar.slider("IoU",        0.0, 1.0, 0.45, 0.01)
max_det     = st.sidebar.number_input("Max detections", min_value=10, max_value=1000, value=300, step=10)

with st.sidebar.expander("İsteğe bağlı: Validation (data.yaml)"):
    data_yaml = st.text_input("data.yaml yolu", value="")
    run_val = st.button("Validation çalıştır")

# -------------------- Model yükle --------------------
if not os.path.exists(weights_path):
    st.error(f"Ağırlık dosyası bulunamadı: {weights_path}")
    st.stop()

model = load_model(weights_path)
names = model.model.names if hasattr(model, "model") else model.names
st.success(f"Model yüklendi. Sınıflar: {len(names)}")

# -------------------- Sekmeler --------------------
tab_img, tab_batch = st.tabs(["🖼️ Tek Resim", "🗂️ Çoklu Resim"])

with tab_img:
    st.subheader("Tek Resim Yükle")
    up = st.file_uploader("Resim seç (.jpg/.jpeg/.png)", type=["jpg","jpeg","png"], accept_multiple_files=False)
    if up is not None:
        img = Image.open(up).convert("RGB")
        colL, colR = st.columns([3,2], gap="large")

        with colL:
            st.image(img, caption="Orijinal", use_container_width=True)

        # Inference
        res = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

        # Görselleştirme
        plotted = bgr_to_rgb(res.plot())
        with colR:
            st.image(plotted, caption="Tespitler", use_container_width=True)

            # İndir butonu
            buf = io.BytesIO()
            Image.fromarray(plotted).save(buf, format="PNG")
            st.download_button("⤓ Çıktıyı indir (PNG)", data=buf.getvalue(),
                               file_name=f"detections_{os.path.splitext(up.name)[0]}.png",
                               mime="image/png")

        # Tablo + sınıf dağılımı
        df = result_to_df(res, names)
        st.markdown("**Tespit Tablosu**")
        st.dataframe(df, use_container_width=True)

        if len(df):
            st.markdown("**Sınıf Bazında Sayım**")
            st.dataframe(df["class_name"].value_counts().rename_axis("class").reset_index(name="count"),
                         use_container_width=True)

with tab_batch:
    st.subheader("Çoklu Resim Yükle")
    files = st.file_uploader("Birden fazla resim seç", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        for f in files:
            st.write("---")
            st.write(f"**{f.name}**")
            img = Image.open(f).convert("RGB")
            res = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
            plotted = bgr_to_rgb(res.plot())
            st.image(plotted, use_column_width=True)
            df = result_to_df(res, names)
            with st.expander("Detay tablosu"):
                st.dataframe(df, use_container_width=True)

# -------------------- İsteğe bağlı: Validation --------------------
if run_val and data_yaml.strip():
    st.info("Validation çalışıyor... (dataset: val)")
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    st.success("Validation tamamlandı.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("mAP@0.50", f"{metrics.box.map50:.4f}")
    with col2:
        st.metric("mAP@0.50:0.95", f"{metrics.box.map:.4f}")
