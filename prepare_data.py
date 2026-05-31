"""
prepare_data.py — Ham PlantVillage verisini train/val/test olarak böler.

NEDEN GEREKLİ?
YOLO11 sınıflandırma (classification) modeli, veriyi ImageNet tarzı bir KLASÖR
YAPISI üzerinden okur. Yani kök klasörün altında train/ val/ test/ olur ve her
birinin altında SINIF ADIYLA bir alt klasör bulunur; o alt klasörün içindeki tüm
görseller o sınıfa ait sayılır. Etiket = klasör adı. Ayrı bir etiket dosyası YOK.

Ham PlantVillage'da hazır bir train/val ayrımı OLMADIĞI için bu bölmeyi biz
yapıyoruz: her sınıfın görsellerini karıştırıp %80 train / %10 val / %10 test'e
ayırıyoruz ve data/dataset/{train,val,test}/<sınıf>/ yapısına KOPYALIYORUZ.

Çalıştır: python prepare_data.py
"""

import shutil
import random
from pathlib import Path

# --- Sabit yollar (spec gereği argparse/config yok, yollar burada sabit) ---
SRC = Path("data/raw/PlantVillage")   # Ham veri: her sınıf için bir alt klasör. Gerekirse düzelt.
DST = Path("data/dataset")            # Çıktı: train/val/test bölmelerini içeren kök klasör.

# %80 / %10 / %10 oranlarında bölüyoruz.
TRAIN_RATIO, VAL_RATIO = 0.8, 0.9     # [0, .8) train, [.8, .9) val, [.9, 1] test

# Sadece görüntü dosyalarını al (klasörde gizli dosya / başka şey olursa atlanır).
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# Tekrarlanabilirlik: aynı tohum -> her çalıştırmada AYNI bölme. Sonuçların
# karşılaştırılabilir olması için şart (rastgele bölme her seferinde değişmesin).
random.seed(42)


def main():
    if not SRC.exists():
        raise SystemExit(
            f"Ham veri bulunamadı: {SRC}\n"
            "Önce PlantVillage'ı indirip data/raw/ içine çıkar (bkz. CLAUDE.md / ODEV_TALIMAT.md)."
        )

    # Her bölmedeki toplam görsel sayısını sonunda yazdırmak için sayaç.
    totals = {"train": 0, "val": 0, "test": 0}

    # Sınıfları alfabetik sırada gez (sorted -> sınıf indekslemesi deterministik olsun).
    class_dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"{SRC} içinde sınıf klasörü yok. Yol doğru mu?")

    for cls_dir in class_dirs:
        # Bu sınıfın tüm görsellerini topla (uzantıya göre filtrele).
        images = [f for f in cls_dir.glob("*") if f.suffix.lower() in IMG_EXTS]
        random.shuffle(images)            # Karıştır ki bölme rastgele/dengeli olsun.

        n = len(images)
        i_train = int(n * TRAIN_RATIO)    # ilk %80
        i_val = int(n * VAL_RATIO)        # sonraki %10 (%80 -> %90 arası), kalan %10 test

        parts = {
            "train": images[:i_train],
            "val": images[i_train:i_val],
            "test": images[i_val:],
        }

        # Her bölme için sınıf klasörünü oluştur ve görselleri kopyala.
        for split, files in parts.items():
            out_dir = DST / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, out_dir / f.name)   # taşımıyoruz, kopyalıyoruz -> ham veri korunur
            totals[split] += len(files)

        print(f"{cls_dir.name}: {n} görsel bölündü "
              f"(train={len(parts['train'])}, val={len(parts['val'])}, test={len(parts['test'])})")

    # Kontrol amaçlı: her bölmedeki toplam görsel sayısı.
    print("\n--- TOPLAM ---")
    for split, count in totals.items():
        print(f"{split}: {count} görsel")
    print(f"Sınıf sayısı: {len(class_dirs)}")
    print(f"Çıktı kök klasörü: {DST.resolve()}")


if __name__ == "__main__":
    main()
