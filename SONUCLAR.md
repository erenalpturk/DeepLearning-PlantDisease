# Sonuçlar

## Model ve ayarlar
- Model: `yolo11s-cls.pt` (transfer learning / fine-tune)
- Mimari özeti: 47 katman, ~5.45M parametre, 12.0 GFLOPs
- Görüntü boyutu (imgsz): 224
- Epoch: 20, batch: 32, patience: 5
- Veri bölmesi: %80 train / %10 val / %10 test (seed=42) — 15 sınıf
  - train: 16504, val: 2064, test: 2070 görsel
- Eğitim cihazı: Tesla T4 GPU (CUDA), ~0.64 saat

## Test metrikleri (evaluate.py — test bölmesi, 2070 görsel)
- **Top-1 accuracy: 0.9986**
- Top-5 accuracy: 1.0000
- Macro avg F1: 0.9985 — Weighted avg F1: 0.9986

### Sınıf bazlı (sadece %100 olmayanlar)
| Sınıf | precision | recall | f1 | support |
|---|---|---|---|---|
| Tomato_Early_blight | 0.9804 | 1.0000 | 0.9901 | 100 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 0.9941 | 1.0000 | 0.9970 | 168 |
| Tomato__Target_Spot | 1.0000 | 0.9858 | 0.9929 | 141 |
| Tomato_healthy | 1.0000 | 0.9938 | 0.9969 | 160 |

Diğer 11 sınıfın tümü precision/recall/f1 = 1.0000.

## Grafikler
- Eğitim eğrileri (loss + accuracy): `runs/plant_cls/results.png`
- Confusion matrix: `runs/plant_cls/confusion_matrix.png`

## Yorum
Model test setinde %99.86 top-1 doğruluğa ulaştı; 2070 görselden yalnızca ~3 tanesi
yanlış sınıflandı. Hataların **hepsi domates (Tomato) sınıfları arasında** yoğunlaştı:
`Tomato__Target_Spot` örneklerinin 2'si kaçırıldı ve `Tomato_Early_blight`'a 2 yanlış
pozitif geldi — bu iki hastalığın yaprak üzerindeki kahverengi leke desenleri görsel
olarak çok benzediği için en olası karışan çift bunlar. Ayrıca 1 `Tomato_healthy`
görseli `Spider_mites` ile karıştı. Buna karşılık tüm biber (Pepper) ve patates
(Potato) sınıfları ile sağlıklı sınıfların çoğu kusursuz ayrıldı. PlantVillage'ın
temiz, sabit arka planlı laboratuvar görüntülerinden oluşması doğruluğun bu kadar
yüksek çıkmasının başlıca nedeni; gerçek dünya (saha) görsellerinde doğruluğun daha
düşük olması beklenir.
