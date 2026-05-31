"""
predict.py — Tek bir yaprak fotoğrafından bitkiyi ve hastalığı tahmin eder.

Veri setindeki sınıf adları zaten 'Bitki_Hastalık' formatında olduğu için
TEK bir sınıflandırma modeli her iki soruyu da çözer: model bir sınıf tahmin
eder, biz de sınıf adını bitki + hastalık olarak AYRIŞTIRIP yazdırırız.

Çalıştır: python predict.py yaprak.jpg
"""

import sys
from ultralytics import YOLO

WEIGHTS = "runs/plant_cls/weights/best.pt"

# Görsel yolu komut satırından alınır (tek argüman alan script budur).
if len(sys.argv) < 2:
    raise SystemExit("Kullanım: python predict.py <görsel_yolu>")
img_path = sys.argv[1]

model = YOLO(WEIGHTS)
result = model(img_path)[0]                  # tek görselin tahmin sonucu

cls_name = result.names[result.probs.top1]   # en olası sınıf adı, ör. "Tomato_Late_blight"
conf = float(result.probs.top1conf)          # o sınıfın güven (olasılık) değeri

# --- Sınıf adını bitki + hastalık olarak ayrıştır ---
# PlantVillage ayraçları tutarsız (Pepper__bell___Bacterial_spot gibi). Önce çoklu
# alt çizgileri tek alt çizgiye indir, sonra parçalara böl. İlk parça = bitki, kalanı = durum.
# (Spec'e göre bu kaba ayrıştırma ödev için yeterli.)
clean = cls_name.replace("___", "_").replace("__", "_")
parts = clean.split("_")
plant = parts[0]
disease = " ".join(parts[1:]) if len(parts) > 1 else "healthy"

print(f"Bitki: {plant} | Durum: {disease} | Güven: {conf:.2%}")
