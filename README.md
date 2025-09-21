# RDD2020 → YOLOv8 (PyTorch→TFLite)

Yol çatlak/hasar tespiti için **Road Damage Dataset 2020 (RDD2020)** verisiyle YOLOv8 modeli eğitip, modeli **TFLite (FP16)**’a dışa aktarmak için uçtan uca bir rehber.

> Bu README, paylaştığınız Python kodunu temel alır: VOC XML → YOLO etikete dönüştürme, eğitim/validasyon ayrımı, YOLOv8 ile eğitim/validasyon ve **TFLite** export.

---

## 1) Proje Özeti

* **Veri kümesi**: RDD2020 (örnek ülkeler: *Czech*, *Japan*, *India*).
* **Amaç**: VOC XML etiketlerini YOLO formatına çevirip küçük bir altküme üzerinde (isteğe göre tüm veri) YOLOv8 eğitmek.
* **Çıktılar**: `rdd2020_subset_yolo/` (YOLO dataset yapısı), `runs/detect/train/` (eğitim artefaktları), `best.pt`, `best_float16.tflite`.

---

## 2) Dizin Yapısı (Beklenen)

```
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
└─ {images,labels}/{train,val}/*.jpg|*.txt
└─ data.yaml
```

> Girdi kök yolu kodda `base_path = 'archive/train'` olarak tanımlıdır. Çıktı kök yolu `output_base_path = '/working/rdd2020_subset_yolo'`.

---

## 3) Kurulum

### Gereksinimler

* Python 3.9+
* PyTorch (CUDA varsa GPU ile)
* Ultralytics (YOLOv8)
* lxml veya xml parser (standart `xml.etree.ElementTree` yeterlidir)
* opsiyonel: `wandb`

```bash
pip install ultralytics==8.* tqdm pyyaml
# (GPU için) pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Windows’ta TFLite export sırasında ek bağımlılık gerekmeyebilir; sorun yaşarsanız `onnx`, `tensorflow` (CPU) kurulumu gerektirebilir. Ultralytics çoğu durumda bunları otomatik yönetir.

---

## 4) Veri Hazırlama (VOC XML → YOLO)

Kodun yaptığı işlemler:

1. Ülkeleri gezer: `countries = ['Czech','Japan','India']`
2. Her ülke için `annotations/xmls` altındaki XML’leri toplar.
3. `subset_ratio` ile altküme seçer (1 → %100, 0.2 → %20 gibi).
4. `split_ratio = 0.8` ile train/val ayırır.
5. Her görsel için VOC kutularını **YOLO formatına** (class x\_center y\_center w h, normalize) dönüştürür.
6. **Sınıf sözlüğünü** dinamik üretir: ilk gördüğü sınıfa 0, sonra 1…

### Çalıştırma (script olarak)

`prepare_rdd2020_yolo.py` gibi kaydedip çalıştırabilirsiniz:

```bash
python prepare_rdd2020_yolo.py \
  --base_path archive/train \
  --out /working/rdd2020_subset_yolo \
  --countries Czech Japan India \
  --subset_ratio 1.0 \
  --split_ratio 0.8
```

> Kodunuz sabit değişkenlerle çalışıyor; argparse eklemek isterseniz aşağıda örnek verildi (Bkz. *Ekler*).

### Üretilen `data.yaml`

Kod, `data.yaml` dosyasını otomatik yazar:

```yaml
train: /working/rdd2020_subset_yolo/images/train
val: /working/rdd2020_subset_yolo/images/val
nc: <sınıf_sayısı>
names: [<sınıf_0>, <sınıf_1>, ...]
```

> `names` sırası, veri hazırlama sırasında karşılaşılan sınıf sırasına göredir. Projelerde yeniden üretilebilirlik için bu sözlüğü sabitlemeniz önerilir (bkz. **İpuçları**).

---

## 5) YOLOv8 Eğitimi

Temel parametreler:

* **Model**: `yolov8s.pt` (hız/başarı dengesi). Hızlı denemeler için `yolov8n.pt`.
* **Epoch**: 20 (başlangıç için).
* **imgsz**: 640.
* **WANDB**: devre dışı.

### Komut (Python ile)

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

**Eğitim çıktıları**: `runs/detect/train/` altında `weights/best.pt`, `weights/last.pt`, `results.png`, `confusion_matrix.png` vb.

---

## 6) Modeli Kaydetme

Ultralytics eğitim sonunda zaten `best.pt` üretir. Ek olarak manuel kaydetmek isterseniz:

```python
model.save('best_model.pt')
```

---

## 7) TFLite’a Dışa Aktarma (FP16)

```python
from ultralytics import YOLO
m = YOLO('runs/detect/train/weights/best.pt')  # veya 'best_model.pt'
m.export(
    format='tflite',
    imgsz=640,
    half=True,  # FP16
    nms=True    # grafiğe NMS ekle
)
# çıktı: best_float16.tflite
```

> Notlar:
> • `imgsz` eğitimle **aynı** olmalı.
> • Bazı ortamlarda TensorFlow/FlatBuffers eksikse kurulum yapmanız gerekebilir.
> • Android/Flutter’da TFLite kullanırken giriş/çıkış tensör şekilleri ve NMS uyumluluğuna dikkat edin.

---

## 8) Hızlı Çıkarım (Inference) Örneği

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
res = model('sample.jpg')
res[0].show()         # OpenCV penceresi
res[0].save('out/')   # `out/` klasörüne kaydeder
```

---

## 9) Karşılaşılabilecek Hatalar & Çözümler

* **`Image file ... does not exist. Skipping.`**
  XML içindeki `filename` gerçek dosya adıyla uyuşmuyor olabilir. Çözüm:

  * XML’de `filename` boşsa kod zaten `xml_file` tabanlı `.jpg` dener.
  * Farklı uzantılar (JPG/jpg/png) varsa dönüştürme/senkronizasyon yapın.

* **`Size missing in annotation ...`**
  XML’de `<size>` yoksa o örnek atlanır. Çözüm: Kusurlu XML dosyalarını bulup düzeltin.

* **Bozuk XML / `ET.ParseError`**
  Hatalı XML kayıtları atlanır. Gerekirse bu dosyaları listeleyip temizleyin.

* **Sınıf isimleri düzensiz**
  Veri hazırlama sırasında dinamik sözlük oluşuyor. Eğitim tekrarlarında sıra değişebilir. Çözüm: Sınıf–ID sözlüğünü sabitleyin (bkz. **İpuçları**).

* **CUDA görünmüyor**
  `device=-1` ile CPU’da deneyin. CUDA sürümü ve PyTorch uyumluluğunu kontrol edin.

---

## 10) İyileştirme İpuçları

* **Sınıf sözlüğünü sabitleyin**: RDD2020’nin bilinen sınıflarını önceden tanımlayıp `class_name_to_id`’yi bu haritayla doldurun; veri hazırlama sırasında buna göre yazdırın. Böylece tekrar üretilebilirlik artar.
* **Augmentasyon**: `hyp` dosyası veya `train` argümanlarıyla `mosaic`, `mixup`, `hsv_h`, `degrees` gibi arttırmalarla oynayın.
* **Model boyutu**: `yolov8m/l` ile doğruluğu; `yolov8n` ile hızı test edin.
* **Epoch / LR**: 20 yerine 50–100 epoch denenebilir; erken durdurma (patience) ve `batch` büyüklüğü önemli.
* **imgsz**: 640→768/960 doğruluğu artırabilir ama eğitim süresini uzatır.
* **Class imbalance**: Nadir sınıflar için **oversampling** veya **weighted loss** araştırın.

---

## 11) Lisans & Atıf

* **RDD2020** lisans koşullarına uyun (kaynaktan kontrol edin).
* Makale/raporlarda: *RDD2020 dataset* ve *Ultralytics YOLOv8*’e atıf verin.

---

## 12) Kısa Komut Özeti

```bash
# 1) Veri dönüştürme
python prepare_rdd2020_yolo.py --base_path archive/train \
  --out /working/rdd2020_subset_yolo --countries Czech Japan India \
  --subset_ratio 1.0 --split_ratio 0.8

# 2) Eğitim
yolo detect train data=/working/rdd2020_subset_yolo/data.yaml \
  model=yolov8s.pt imgsz=640 epochs=20 device=0

# 3) Değerlendirme
yolo detect val model=runs/detect/train/weights/best.pt \
  data=/working/rdd2020_subset_yolo/data.yaml imgsz=640 device=0

# 4) Export (TFLite FP16)
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 half=True nms=True
```

---

## Ekler — Argparse’lı Veri Hazırlama İskeleti (Opsiyonel)

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
                    if not class_name: continue

                    if class_name not in class_name_to_id:
                        class_name_to_id[class_name] = len(class_name_to_id)
                    class_id = class_name_to_id[class_name]

                    b = obj.find('bndbox')
                    if b is None: continue
                    xmin = float(b.findtext('xmin'))
                    ymin = float(b.findtext('ymin'))
                    xmax = float(b.findtext('xmax'))
                    ymax = float(b.findtext('ymax'))

                    x_center = ((xmin + xmax) / 2) / width
                    y_center = ((ymin + ymax) / 2) / height
                    bw = (xmax - xmin) / width
                    bh = (ymax - ymin) / height

                    label_file.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

# data.yaml
data = {
    'train': os.path.join(output_base_path, 'images', 'train'),
    'val': os.path.join(output_base_path, 'images', 'val'),
    'nc': len(class_name_to_id),
    'names': list(class_name_to_id.keys()),
}
with open(os.path.join(output_base_path, 'data.yaml'), 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("Done. Classes:")
for k,v in class_name_to_id.items():
    print(v, k)
```

---

## 13) Streamlit Web UI (İsteğe Bağlı)

Paylaştığın `streamlit_app.py` dosyasını bu projeye doğrudan ekleyebilirsin. Uygulama tek/çoklu resim yükleme, eşi̇k ayarları (conf/IoU/imgsz), sonuç görselleştirme ve isteğe bağlı validation (data.yaml) destekler.

### Kurulum

```bash
pip install streamlit ultralytics pillow pandas numpy
# (GPU kullanacaksan) uygun PyTorch + CUDA sürümünü ayrıca kur
```

### Çalıştırma

```bash
streamlit run streamlit_app.py
```

* **Ağırlık yolu**: Sidebar → *Model ağırlığı (.pt)* alanına `runs/detect/train/weights/best.pt` (veya `best_model.pt`) ver.
* **İsteğe bağlı validation**: `data.yaml` yolunu girip **Validation çalıştır**’a bas.

### Notlar

* `@st.cache_resource` ile model bir kez yüklenir (sayfa yenilense de hızlı döner).
* `imgsz` eğitimde kullandığın boyutla aynı olmalı (örn. 640).
* Eğer **yükleme boyutu** sorunu yaşarsan `~/.streamlit/config.toml` içine:

  ```toml
  [server]
  maxUploadSize = 512
  ```

  ekleyebilirsin (MB cinsinden).
* Windows’ta yol ayırıcıları (`\` vs `/`) ve Türkçe karakterli patikalar sorun çıkarırsa düz ASCII klasör isimleri kullan.
* Torch/Ultralytics sürüm uyuşmazlığında: `pip show ultralytics torch` ile sürümleri kontrol edip Ultralytics 8.x + Torch sürümünü eşleştir.

### Küçük İyileştirmeler (opsiyonel)

* **Toplu çıktı indirme**: Çoklu sekmesinde her görsel için `st.download_button` ekleyebilirsin.
* **Sembol adları**: `names = model.model.names if hasattr(model, "model") else model.names` ifadesi farklı sürümlerde uyumluluk sağlar (kodunda zaten var).
* **CSV export**: `result_to_df` dönen tabloyu `df.to_csv` ile indirilebilir yap.
