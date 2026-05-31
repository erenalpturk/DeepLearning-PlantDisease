"""
train.py — Önceden eğitilmiş YOLO11-cls modelini PlantVillage üzerinde fine-tune eder.

TRANSFER LEARNING MANTIĞI:
Sıfırdan eğitmek yerine, ImageNet üzerinde önceden eğitilmiş 'yolo11s-cls.pt'
AĞIRLIKLARINI (weights) başlangıç noktası alıyoruz. Bu ağırlıklar, kenar/doku/
şekil gibi genel görsel özellikleri zaten öğrenmiş durumda; biz sadece bu bilgiyi
kendi 15 sınıfımıza uyarlıyoruz (fine-tune). Bu yüzden az veriyle ve az epoch'ta
hızla yüksek doğruluğa ulaşırız.

AĞ / KATMAN NOTU (hoca sınavda soruyor):
- Model bir EVRİŞİMLİ SİNİR AĞI (CNN). Görsel, art arda gelen convolution
  katmanlarından geçer; her katman bir öncekinin çıktısı üzerinde küçük filtreler
  (çekirdekler) kaydırarak özellik haritaları üretir.
- STRIDE = filtrenin her adımda kaç piksel kaydığı. Stride>1 olduğunda özellik
  haritası küçülür (downsampling) -> daha az hesap, daha geniş "görüş alanı".
- Derinleştikçe katmanlar daha soyut özellikler öğrenir; en sonda tam-bağlı
  (fully-connected) sınıflandırma başı 15 sınıf için olasılık üretir.
- WEIGHTS = bu filtrelerin/bağlantıların öğrenilen sayısal değerleri. Eğitim,
  geri-yayılım (backprop) ile bu ağırlıkları kayıp (loss) azalacak şekilde günceller.

Çalıştır: python train.py
Çıktılar: runs/plant_cls/weights/best.pt  (en iyi ağırlık)
          runs/plant_cls/results.png      (loss + accuracy eğrileri)
"""

import torch
from ultralytics import YOLO

# --- Cihaz seçimi ---
# CUDA'lı GPU varsa onu; yoksa Apple Silicon'da MPS'i; o da yoksa CPU'yu kullan.
# (Mac'te CUDA olmaz; MPS belirtilmezse ultralytics CPU'ya düşüp çok yavaş kalabilir.)
if torch.cuda.is_available():
    device = 0                  # ilk CUDA GPU
elif torch.backends.mps.is_available():
    device = "mps"              # Apple Silicon GPU
else:
    device = "cpu"
print(f"Eğitim cihazı: {device}")

# Önceden eğitilmiş YOLO11 sınıflandırma modelini yükle (yoksa otomatik indirilir).
# 's' = small varyant. Daha yüksek doğruluk istenirse 'yolo11m-cls.pt' denenebilir.
model = YOLO("yolo11s-cls.pt")

# Fine-tune eğitimi.
model.train(
    data="data/dataset",   # train/ ve val/ alt klasörlerini içeren kök (ImageNet düzeni)
    epochs=20,             # PlantVillage hızlı yakınsar; yetmezse 30-40'a çıkar
    imgsz=224,             # Giriş görselleri 224x224'e YENİDEN BOYUTLANDIRILIR.
                           #   Tüm batch'in aynı boyutta olması ve önceden eğitilmiş
                           #   ağırlıkların beklediği giriş boyutuyla uyum için gerekli.
    batch=32,              # Aynı anda işlenen görsel sayısı. GPU belleği yetmezse düşür.
    patience=5,            # Erken durdurma: 5 epoch boyunca val iyileşmezse durur (overfit önler)
    project="runs",        # Çıktı kök klasörü
    name="plant_cls",      # Bu çalıştırmanın alt klasör adı -> runs/plant_cls/
    device=device,
)

# Eğitim bittiğinde en iyi ağırlık otomatik kaydedilir:
#   runs/plant_cls/weights/best.pt   (val doğruluğu en yüksek epoch)
#   runs/plant_cls/weights/last.pt   (son epoch)
# Değerlendirme ve tahmin best.pt ile yapılır.
print("Eğitim bitti -> en iyi ağırlık: runs/plant_cls/weights/best.pt")
