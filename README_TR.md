# ArachneAI Small

ArachneAI Small, ArachneAI ekosisteminin sıfırdan model geliştirme ve eğitim deposudur. Ticari SaaS ürünü olan ArachneAI'den ve açık kaynak çıkarım motoru ArachneAIOpen'dan ayrı tutulur.

## Model mimarisi

Hedef model, yaklaşık **1.524 milyar parametreli** decoder-only Transformer mimarisidir. Depodaki yapılandırmanın gerçek parametre sayısı **1.523.714.048**'dir.

Mimari bileşenleri:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped-Query Causal Attention (GQA)
- SwiGLU feed-forward katmanı
- Residual bağlantılar
- İsteğe bağlı gradient checkpointing

Model rastgele başlangıç ağırlıklarıyla sıfırdan eğitilir. Üçüncü taraf önceden eğitilmiş ağırlıklar indirilmez, temel model olarak kullanılmaz veya fine-tuning yapılmaz.

## Kurulum

Python 3.10+ gerekir. PyTorch kurulumunu hedef CPU/CUDA ortamına uygun seçin.

```bash
python -m venv .venv
# Sanal ortamı etkinleştirin.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Veri hazırlama

Eğitim verisi, her satırında `text` alanı bulunan UTF-8 JSONL dosyasıdır:

```json
{"text":"Türkçe veya İngilizce lisanslı eğitim metni."}
```

Yalnızca model eğitimi için kullanım hakkı belgelenmiş verileri kullanın. Bir içeriğin internette erişilebilir olması, o içeriğin model eğitiminde kullanılabileceği anlamına gelmez. Ham veri kümelerini Git deposuna eklemeyin.

Önce mükerrer kayıtları ayıklayın, metni normalize edin ve belge seviyesinde deterministik eğitim/doğrulama ayrımı yapın:

```bash
arachne-small prepare-corpus --input data/source.jsonl --output-dir data/split --validation-fraction 0.01 --seed 1337 --source "Veri kaynağı ve sürümü" --license "Lisans veya yazılı izin"
```

Tokenizer yalnızca eğitim bölümüyle oluşturulur:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000
```

İki veri bölümünü aynı tokenizer ile ayrı ayrı kodlayın:

```bash
arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

Her token dosyası için SHA-256, tokenizer kimliği, özel token ID'leri ve token sayısını içeren JSON sidecar üretilir. Eğitim, dosya checksum'larını ve tokenizer eşleşmesini doğrulamadan başlamaz.

## Eğitim

```bash
arachne-small count-parameters --config configs/1_5b.yaml
arachne-small train --config configs/1_5b.yaml --train-data data/processed/train.bin --validation-data data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model
```

Küçük ölçekli yerel kod doğrulaması için `configs/dev.yaml` kullanılabilir. Tokenizer sözlüğü ve veri uzunluğu yapılandırmayla eşleşmelidir. 1.5B yapılandırması ciddi GPU belleği, depolama ve eğitim süresi gerektirir. Yapılandırmanın bulunması, model ağırlıklarının eğitilmiş olduğu anlamına gelmez.

Checkpoint'ten devam etmek için:

```bash
arachne-small train --config configs/1_5b.yaml --train-data data/processed/train.bin --validation-data data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --resume artifacts/arachne-small-1.5b/checkpoints/step-XXXXXXXX.pt
```

## Model paketi ve doğrulama

Eğitim tamamlandığında çıktı dizininde aşağıdaki inference paketi oluşturulur:

- `model.safetensors`: model ağırlıkları; pickle kullanılmaz.
- `tokenizer.model`: eğitimde kullanılan tam SentencePiece tokenizer.
- `config.json`: mimari ve tokenizer sözleşmesi.
- `artifact-manifest.json`: dosya SHA-256 değerleri ve eğitim metaverisi.
- `training.jsonl`: eğitim kaybı ve doğrulama ölçümleri.
- `checkpoints/`: optimizer/scaler durumunu içeren yerel devam dosyaları.

Paketi doğrulayın:

```bash
arachne-small verify-artifact --path artifacts/arachne-small-1.5b
```

ArachneAIOpen ile ortak artifact sözleşmesi [docs/ARTIFACT_FORMAT.md](docs/ARTIFACT_FORMAT.md) içinde tanımlanmıştır.

## Veri yönetişimi ve model kartı

- [Model Card](docs/MODEL_CARD.md)
- [Dataset Policy](docs/DATASET_POLICY.md)
- [Artifact Format](docs/ARTIFACT_FORMAT.md)

## Kod lisansı

Bu depoya henüz bir kod lisansı eklenmedi. Depoyu public yapmak, kodun yeniden kullanılmasına, değiştirilmesine veya dağıtılmasına otomatik izin vermez. Açık kaynak lisansı olarak tanımlamadan önce depo sahibinin lisans seçimi yapılmalıdır.

## Yayın durumu

Bu depo model mimarisini, veri hazırlama hattını, tokenizer üretimini, eğitim motorunu, checkpoint yönetimini ve inference artifact paketlemesini içerir. Şu anda eğitilmiş ağırlık veya eğitim korpusu içermez. Eğitim, yerel testler, doğrulama metrikleri ve donanım performans ölçümleri tamamlanmadan eğitilmiş model sürümü veya kalite iddiası yayımlanmayacaktır.
