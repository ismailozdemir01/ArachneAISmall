# ArachneAISmall

**ArachneAISmall, ArachneAI ekosistemi içinde Türkçe ve İngilizce odaklı, sıfırdan dil modeli geliştirme ve eğitim projesidir.**

Depo; lisanslı eğitim verisinin hazırlanması, tokenizer oluşturulması, decoder-only Transformer modelinin eğitilmesi, model paketlerinin değerlendirilmesi ve doğrulanmış modelin yerel API ile web arayüzünden sunulması için gerekli yazılımı içerir.

- **Hedef diller:** Türkçe ve İngilizce
- **Hedef model boyutu:** 1.5B sınıfı
- **Yapılandırılmış parametre sayısı:** 1.523.714.048
- **Eğitim yaklaşımı:** Rastgele başlangıç ağırlıkları; üçüncü taraf önceden eğitilmiş ağırlık kullanılmaz
- **Proje durumu:** Eğitim ve yayınlama hattının yazılımı geliştirildi; eğitilmiş model ağırlıkları henüz yayımlanmadı

[🇬🇧 English README](README.md)

---

## Projenin amacı

ArachneAISmall, kullanım hakkı belgelenmiş metin koleksiyonlarından sürümlenebilir dil modeli paketine kadar izlenebilir ve tekrarlanabilir bir geliştirme süreci oluşturmayı amaçlar. Mevcut bir önceden eğitilmiş modeli uyarlamak yerine modelin kendisi sıfırdan geliştirilir ve eğitilir.

1.5B yapılandırması bir mimari hedeftir; **bu boyutta bir modelin hâlihazırda eğitildiği anlamına gelmez**. Türkçe ve İngilizce hedef dillerdir. Dil kalitesi ve yetenek iddiaları ancak gerçek eğitim ve yayımlanmış değerlendirmelerle desteklenebilir.

## Model mimarisi

Yapılandırılmış decoder-only Transformer mimarisi şu bileşenleri kullanır:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped-Query Causal Attention (GQA)
- SwiGLU feed-forward katmanları
- Residual bağlantılar
- İsteğe bağlı gradient checkpointing
- Yapılandırılabilir model boyutları ve eğitim ayarları

Parametre sayısını yapılandırmadan doğrudan hesaplayın:

```bash
arachne-small count-parameters --config configs/1_5b.yaml
```

## Geliştirilen bileşenler

| Alan | Uygulama |
|---|---|
| Veri hazırlama | Unicode normalizasyonu, deterministik belge ayrımı, disk tabanlı mükerrer kayıt temizliği ve kaynak manifesti |
| Tokenizer | Eğitim bölümü üzerinden SentencePiece BPE eğitimi |
| Token veri kümeleri | Checksum ve tokenizer metaverisi içeren ikili token dosyaları |
| Eğitim motoru | PyTorch, AdamW, gradient accumulation, gradient clipping, warmup/cosine schedule ve değerlendirme |
| Checkpoint | Atomik kayıt ve doğrulamalı kaldığı yerden devam |
| Model paketleri | SafeTensors ağırlıkları, tokenizer, model yapılandırması ve SHA-256 manifesti |
| Paket doğrulama | Dosya bütünlüğü, tokenizer kimliği, tensor adları/şekilleri ve mimari kontrolleri |
| Çıkarım | FastAPI uç noktaları ve Türkçe/İngilizce tarayıcı arayüzü |
| Geri bildirim | Açık kullanıcı rızasıyla yerel SQLite kaydı |
| Sürekli öğrenme | Doğrulanmış temel modelden aday eğitim, baseline karşılaştırması, kontrollü yayınlama ve geri alma |

Uygulama ve işletim ayrıntıları `docs/` dizinindedir.

## Hızlı başlangıç

### Gereksinimler

- Python 3.10 veya üzeri
- Hedef CPU veya CUDA ortamına uygun PyTorch kurulumu
- UTF-8 JSONL biçiminde, kullanım hakkı belgelenmiş eğitim metinleri

Projeyi kurun:

```bash
python -m venv .venv
# Sanal ortamı etkinleştirin.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Veri kümesini hazırlama

JSONL dosyasındaki her satır bir `text` alanı içermelidir:

```json
{"text":"Lisanslı Türkçe veya İngilizce eğitim metni."}
```

Veri kaynağını ve lisansını kaydederek korpusu hazırlayın:

```bash
arachne-small prepare-corpus \
  --input data/source.jsonl \
  --output-dir data/split \
  --validation-fraction 0.01 \
  --seed 1337 \
  --source "Veri kaynağı ve sürümü" \
  --license "Lisans veya yazılı izin"
```

Tokenizer'ı yalnızca eğitim bölümüyle eğitin, ardından iki bölümü de kodlayın:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000

arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

Veri hattı token dosyalarının yanında bütünlük metaverisi üretir. Eğitim başlamadan önce token dosyalarının checksum değerleri ve tokenizer kimliği doğrulanır.

### Model eğitimi

```bash
arachne-small train \
  --config configs/1_5b.yaml \
  --train-data data/processed/train.bin \
  --validation-data data/processed/validation.bin \
  --tokenizer artifacts/tokenizer/arachne.model
```

1.5B hedefi yüksek GPU belleği, depolama kapasitesi ve uzun eğitim süresi gerektirir. Küçük yerel yazılım doğrulaması için tokenizer ve veri boyutları eşleşen `configs/dev.yaml` kullanılabilir. Bu test, 1.5B modelini eğitmez veya doğrulamaz.

Yalnızca model eğitimi için kullanım hakkı belgelenmiş veri kümelerini kullanın. Ham korpusları, kişisel verileri veya gizli içerikleri bu Git deposuna eklemeyin.

## Yerel çıkarım

Eğitim tamamlandıktan sonra doğrulanmış model paketinin yolunu tanımlayın ve servisi başlatın.

**Windows PowerShell**

```powershell
$env:ARACHNE_SMALL_ARTIFACT = "artifacts/arachne-small-1.5b"
arachne-small serve
```

**Linux / macOS**

```bash
export ARACHNE_SMALL_ARTIFACT="artifacts/arachne-small-1.5b"
arachne-small serve
```

Tarayıcıdan [http://127.0.0.1:8000](http://127.0.0.1:8000) adresini açın.

Web arayüzü gerçek çıkarım API'sine bağlanır; sahte veya simüle edilmiş model yanıtı üretmez. Doğrulanmış model paketi bulunmuyorsa metin üretimi kullanılamaz ve API HTTP 503 döndürür.

Sunucu varsayılan olarak yalnızca loopback arayüzünde dinler. Uzak erişim açıkça etkinleştirilmelidir; uzaktan yayın için kimlik doğrulamalı TLS reverse proxy ve ağ erişim denetimleri kullanılmalıdır.

### API uç noktaları

| Uç nokta | İşlev |
|---|---|
| `GET /api/health` | Servis sağlık durumu |
| `GET /api/status` | Yüklenen model paketi ve çalışma zamanı durumu |
| `POST /api/generate` | Metin üretimi |
| `POST /api/feedback` | Açık rızalı geri bildirim gönderimi |
| `GET /docs` | Etkileşimli OpenAPI dokümantasyonu |

Ayrıntılar: [Inference API](docs/INFERENCE_API.md).

## Kontrollü sürekli öğrenme

Geri bildirim süreci isteğe bağlıdır ve yerel ortamda çalışır. Uygun düzeltmeler, doğrulanmış temel modelden başlatılan bir aday modeli eğitmek için kullanılabilir. Aday, aynı doğrulama token akışı üzerinde temel modelle karşılaştırılır. Yapılandırılmış doğrulama kaybı eşiği geçilemezse yayınlama engellenir. Yayımlanan sürümler içerik hash'iyle adreslenir ve doğrulanmış geri alma mekanizmasını destekler.

Bu süreç, önceden eğitilmiş bir temel model paketi ve onunla eşleşen doğrulama verisi gerektirir. Bu özellik, sürekli öğrenmenin yayımlanmış bir model üzerinde hâlihazırda çalıştırıldığı anlamına gelmez.

Ayrıntılar: [Kontrollü Sürekli Öğrenme](docs/CONTINUAL_LEARNING.md).

## Dokümantasyon

- [Model Kartı](docs/MODEL_CARD.md)
- [Veri Kümesi Politikası](docs/DATASET_POLICY.md)
- [Model Paketi Biçimi](docs/ARTIFACT_FORMAT.md)
- [Inference API](docs/INFERENCE_API.md)
- [Kontrollü Sürekli Öğrenme](docs/CONTINUAL_LEARNING.md)

## Test ve yayın durumu

Mevcut otomatik test paketi Python 3.14.7 ve CPU tabanlı PyTorch ile yerel ortamda çalıştırıldı: **33 test başarılı**. Aynı ortamda Python bytecode derleme kontrolü ve komut satırı yardım kontrolü de başarıyla tamamlandı.

Bu kontroller yalnızca test paketinin kapsadığı yazılım davranışlarını doğrular. **Eğitilmiş model değerlendirmesi, benchmark, hız ölçümü veya Türkçe/İngilizce dil kalitesi kanıtı değildir.** Bu depoda şu anda eğitilmiş ağırlıklar veya eğitim korpusu yayımlanmamıştır.

Eğitilmiş bir sürüm yayımlanmadan önce gerçek eğitim çalıştırması, ölçümlenmiş değerlendirme sonuçları, model paketi doğrulaması ve donanım performans ölçümleri gereklidir.

## Depoların sorumlulukları

- **ArachneAISmall:** Model mimarisi, veri hazırlama, tokenizer, eğitim, değerlendirme ve model paketi üretimi.
- **ArachneAIOpen:** Belgelenmiş model paketi sözleşmesini kullanan ayrı çıkarım/çalışma zamanı projesi.
- **ArachneAI:** Ayrı ticari SaaS ürünü.

## Lisans

Bu depoda henüz bir kod lisansı seçilmemiştir. Depoyu public yapmak, kodu yeniden kullanma, değiştirme veya dağıtma izni vermez. Projeyi açık kaynak olarak tanımlamadan önce bir lisans eklenmelidir.
