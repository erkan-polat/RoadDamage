# RDD2020 → YOLOv8 (PyTorch → TFLite)

Yol çatlak/hasar tespiti için **Road Damage Dataset 2020 (RDD2020)** ile YOLOv8 modeli eğitme, doğrulama, Streamlit ile etkileşimli arayüz ve **TFLite (FP16 + NMS)** dışa aktarma rehberi.

---

## 🌟 Özellikler

* VOC **XML → YOLO** etikete otomatik dönüşüm
* **Train/Val** ayrımı ve **data.yaml** üretimi
* **YOLOv8** eğitim, doğrulama ve metrikler (mAP\@0.50, mAP\@0.50:0.95)
* **Streamlit Web UI**: tek/çoklu resim yükleme, eşik ayarları (conf/IoU/imgsz), çıktı indirme
* **TFLite (FP16)** export (grafikte **NMS** dahil)

image.png
indir.png

---

## 1) Proje Özeti

* **Veri kümesi**: RDD2020 (*Czech*, *Japan*, *India* alt klasörleri)
* **Amaç**: VOC kutularını YOLO formatına çevirip YOLOv8 ile eğitmek
* **Çıktılar**:

  * Dataset: `/working/rdd2020_subset_yolo/` → `images/{train,val}`, `labels/{train,val}`, `data.yaml`
  * Eğitim: `runs/detect/train/weights/{best.pt,last.pt}`
  * Dağıtım: `best_float16.tflite`

---

## 2) Dizin Yapısı (Beklenen)

```bash
archive/
└─ train/
   ├─ Czech/
   │  ├─ images/*.jpg
   │  └─ annotations/xmls/*.xml
   ├─ India/
   │  ├─ images/*.jpg
   │  └─ annotations/xmls/*.xml
   └─ Japan/
      ├─ images/*.jpg
      └─ annotations/xmls/*.xml

/working/rdd2020_subset_yolo/
├─ images/{train,val}/*.jpg
├─ labels/{train,val}/*.txt
└─ data.yaml
```

---

## 3) Kurulum

```bash
# YOLOv8 + yardımcı kütüphaneler
pip install ultralytics==8.* tqdm pyyaml pillow numpy pandas
# (GPU kullanacaksan) uygun Torch + CUDA sürümünü ayrıca kur
```

> **Windows ipucu**: Yol ayracı (`\`) ve Türkçe/boşluk içeren klasör adları sorun çıkarabilir; mümkünse ASCII klasör adları kullanın.

---

## 4) Veri Hazırlama (VOC XML → YOLO)

Kodun yaptığı adımlar:

1. Ülkeleri gezer (`Czech, Japan, India`)
2. XML’leri toplar ve `subset_ratio` ile altküme seçer
3. `split_ratio` ile **train/val** böler
4. VOC kutularını YOLO formatına (class x\_center y\_center w h, normalize) çevirir
5. `data.yaml` dosyasını üretir

### Örnek Çalıştırma (script)

```bash
python prepare_rdd2020_yolo.py \
  --base_path archive/train \
  --out /working/rdd2020_subset_yolo \
  --countries Czech Japan India \
  --subset_ratio 1.0 \
  --split_ratio 0.8
```

**data.yaml** (otomatik):

```yaml
train: /working/rdd2020_subset_yolo/images/train
val:   /working/rdd2020_subset_yolo/images/val
nc: <sınıf_sayısı>
names: [<sınıf_0>, <sınıf_1>, ...]
```

> **Not**: Sınıf kimlikleri veri hazırlama sırasında ilk karşılaşılan sıraya göre oluşturulur. Tekrar üretilebilirlik için sabit bir sınıf listesi tutmanız önerilir.

---

## 5) YOLOv8 Eğitimi & Doğrulama

```python
from ultralytics import YOLO
import os
os.environ['WANDB_MODE'] = 'disabled'

model = YOLO('yolov8s.pt')
model.train(
    data='/working/rdd2020_subset_yolo/data.yaml',
    epochs=20,
    imgsz=640,
    device=0  # CUDA:0, CPU için -1
)

metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

**Çıktılar**: `runs/detect/train/weights/best.pt`, `results.png`, `confusion_matrix.png` vb.

---

## 6) Modeli Kaydetme

Ultralytics eğitim sonunda `best.pt` üretir. İstersen ayrıca:

```python
model.save('best_model.pt')
```

---

## 7) TFLite’a Dışa Aktarma (FP16 + NMS)

```python
from ultralytics import YOLO
m = YOLO('runs/detect/train/weights/best.pt')  # veya 'best_model.pt'
m.export(
    format='tflite',
    imgsz=640,
    half=True,  # FP16
    nms=True    # grafiğe NMS ekler
)
# => best_float16.tflite
```

> `imgsz` eğitimdeki ile aynı olmalı. Bazı ortamlarda TensorFlow/FlatBuffers gerekebilir.

---

## 8) Hızlı Çıkarım (Inference)

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model('sample.jpg')
results[0].save('out/')
```

---

## 9) Streamlit Web UI

`streamlit_app.py` dosyası ile interaktif arayüz:

**Kurulum**

```bash
pip install streamlit ultralytics pillow pandas numpy
```

**Çalıştırma**

```bash
streamlit run streamlit_app.py
```

**Kullanım**

* Sidebar → **Model ağırlığı (.pt)**: `runs/detect/train/weights/best.pt`
* **Image size / Conf / IoU / Max detections** ayarlarını yap
* Tek/çoklu resim yükle, **çıktıyı indir (PNG)**
* (Opsiyonel) `data.yaml` gir → **Validation çalıştır**

**Notlar**

* `@st.cache_resource` ile model tek sefer yüklenir
* Yükleme boyutu sınırı için `~/.streamlit/config.toml`:

  ```toml
  [server]
  maxUploadSize = 512
  ```
* Çoklu sekmesinde CSV/PNG indirme butonları eklenebilir (örnek kodda hazır şablonlar var)

---

## 10) Karşılaşılabilecek Hatalar & Çözümler

* **`Image file ... does not exist. Skipping.`** → XML `filename` ile gerçek dosya adı uyuşmuyor olabilir. Uzantı farklarını (jpg/JPG/png) normalize edin.
* **`Size missing in annotation ...`** → `<size>` yoksa örnek atlanır; hatalı XML’i düzeltin.
* **`ET.ParseError`** → Bozuk XML; problemli dosyaları loglayıp temizleyin.
* **CUDA görünmüyor** → `device=-1` ile CPU’da deneyin; PyTorch–CUDA sürüm eşleşmesini kontrol edin.
* **Sınıf sırası değişken** → Sınıf–ID haritasını sabitleyin.

---

## 11) İyileştirme İpuçları

* **Model boyutu**: `yolov8n/s/m/l` karşılaştırın (hız/doğruluk)
* **Epoch/Batch/LR**: 20 yerine 50–100 denenebilir; erken durdurma (patience)
* **imgsz**: 640 → 768/960 doğruluğu artırabilir (maliyet artar)
* **Augmentasyon**: mosaic, mixup, hsv, degrees vb.
* **Dengesiz sınıflar**: oversampling/weighted loss stratejileri

---

## 12) Hızlı Komut Özeti

```bash
  data=/working/rdd2020_subset_yolo/data.yaml imgsz=640 device=0

#  Export (TFLite FP16)
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 half=True nms=True
```

---

## 13) Lisans & Atıf

* RDD2020 veri kümesi (https://www.kaggle.com/datasets/ziedkelboussi/rdd2020-dataset/data)
* Ultralytics YOLOv8 kullanım https://github.com/ultralytics/ultralytics
https://github.com/sekilab/RoadDamageDetector

---

## Ek — Argparse’lı Veri Hazırlama İskeleti

*(İsterseniz script olarak kullanın.)*

```python
# prepare_rdd2020_yolo.py (iskelet)
import os, random, shutil, yaml
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--countries', nargs='+', default=['Czech','Japan','India'])
parser.add_argument('--subset_ratio', type=float, default=1.0)
parser.add_argument('--split_ratio', type=float, default=0.8)
args = parser.parse_args()

base_path = args.base_path
output_base_path = args.out
countries = args.countries
subset_ratio = args.subset_ratio
split_ratio = args.split_ratio

for split in ['train','val']:
    os.makedirs(os.path.join(output_base_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'labels', split), exist_ok=True)

class_name_to_id = {}

for country in countries:
    annotations_xmls_path = os.path.join(base_path, country, 'annotations', 'xmls')
    images_path = os.path.join(base_path, country, 'images')

    xml_files = [f for f in os.listdir(annotations_xmls_path) if f.lower().endswith('.xml')]
    num_subset_files = max(1, int(len(xml_files) * subset_ratio))
    subset_xml_files = random.sample(xml_files, num_subset_files)

    annotated_images = []
    for xml_file in subset_xml_files:
        ann_path = os.path.join(annotations_xmls_path, xml_file)
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            filename = root.findtext('filename')
            if not filename:
                filename = os.path.splitext(xml_file)[0] + '.jpg'
            annotated_images.append((filename, xml_file))
        except ET.ParseError as e:
            print(f"Error parsing {ann_path}: {e}")
            continue

    random.shuffle(annotated_images)
    split_index = int(len(annotated_images) * split_ratio)
    train_images = annotated_images[:split_index]
    val_images = annotated_images[split_index:]

    for split, image_list in zip(['train', 'val'], [train_images, val_images]):
        for image_file, xml_file in tqdm(image_list, desc=f"Processing {country} {split} set"):
            image_src_path = os.path.join(images_path, image_file)
            image_dst_path = os.path.join(output_base_path, 'images', split, image_file)
            label_dst_path = os.path.join(output_base_path, 'labels', split, f"{os.path.splitext(image_file)[0]}.txt")
            ann_path = os.path.join(annotations_xmls_path, xml_file)

            if not os.path.exists(image_src_path):
                print(f"Missing: {image_src_path}")
                continue

            shutil.copy(image_src_path, image_dst_path)

            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"Error parsing {ann_path}: {e}")
                continue

            size = root.find('size')
            if size is None:
                print(f"Size missing in {ann_path}")
                continue
            width = int(size.findtext('width'))
            height = int(size.findtext('height'))

            with open(label_dst_path, 'w') as label_file:
                for obj in root.findall('object'):
                    class_name = obj.findtext('name')
                    if not class_name:
                        continue
                    if class_name not in class_name_to_id:
                        class_name_to_id[class_name] = len(class_name_to_id)
                    class_id = class_name_to_id[class_name]

                    b = obj.find('bndbox')
                    if b is None:
                        continue
                    xmin = float(b.findtext('xmin'))
                    ymin = float(b.findtext('ymin'))
                    xmax = float(b.findtext('xmax'))
                    ymax = float(b.findtext('ymax'))

                    x_center = ((xmin + xmax) / 2) / width
                    y_center = ((ymin + ymax) / 2) / height
                    bw = (xmax - xmin) / width
                    bh = (ymax - ymin) / height

                    label_file.write(f"{class_id} {x_center} {y_center} {bw} {bh}
")

# data.yaml
data = {
    'train': os.path.join(output_base_path, 'images', 'train'),
    'val': os.path.join(output_base_path, 'images', 'val'),
    'nc': len(class_name_to_id),
    'names': list(class_name_to_id.keys()),
}
with open(os.path.join(output_base_path, 'data.yaml'), 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print('Done. Classes:')
for k, v in class_name_to_id.items():
    print(v, k)
```
