"""
evaluate.py — Eğitilmiş best.pt modelini TEST bölmesinde değerlendirir.

Hocanın istediği metrikleri burada üretiyoruz:
  - Top-1 accuracy  (asıl not bundan veriliyor -> maksimize edilecek metrik)
  - Top-5 accuracy  (doğru sınıf, modelin en yüksek 5 tahmini içinde mi?)
  - Confusion matrix (model.val çağrısı runs/ altına otomatik kaydeder)
  - Sınıf bazlı precision / recall / F1 (sklearn classification_report) -> hangi
    sınıfların birbirine karıştığını görmek için (SONUCLAR.md yorumunda kullanılır).

Çalıştır: python evaluate.py
"""

import os
import glob
from sklearn.metrics import classification_report
from ultralytics import YOLO

WEIGHTS = "runs/plant_cls/weights/best.pt"   # train.py'nin ürettiği en iyi ağırlık
DATA = "data/dataset"                        # train/val/test kökü
TEST_DIR = os.path.join(DATA, "test")

# Eğitilmiş modeli yükle.
model = YOLO(WEIGHTS)

# --- 1) Topluca değerlendirme: Top-1 / Top-5 ---
# split="test" -> data/dataset/test üzerinde değerlendirir (varsayılan val değil!).
# Bu çağrı ayrıca confusion_matrix.png dahil grafikleri runs/ altına kaydeder.
metrics = model.val(data=DATA, split="test")
print(f"Top-1 accuracy: {metrics.top1:.4f}")
print(f"Top-5 accuracy: {metrics.top5:.4f}")

# --- 2) Sınıf bazlı rapor (precision / recall / F1) ---
# names: {0: 'Pepper__bell___healthy', 1: 'Tomato_Late_blight', ...}
names = model.names
name_to_idx = {v: k for k, v in names.items()}   # sınıf adı -> indeks (ters eşleme)

y_true, y_pred = [], []
# Her test görselini tek tek modele verip tahmin sınıfını topluyoruz.
for cls in sorted(os.listdir(TEST_DIR)):
    cls_path = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    for img in glob.glob(os.path.join(cls_path, "*")):
        r = model(img, verbose=False)[0]    # tek görsel sonucu
        y_pred.append(int(r.probs.top1))    # modelin en olası sınıf indeksi
        y_true.append(name_to_idx[cls])     # gerçek sınıf indeksi (klasör adından)

print("\n--- Sınıf bazlı rapor ---")
print(classification_report(
    y_true, y_pred,
    target_names=[names[i] for i in range(len(names))],
    digits=4,
))
