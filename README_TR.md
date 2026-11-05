# ArachneAISmall

**ArachneAISmall, ArachneAI ekosistemi içinde sıfırdan Türkçe–İngilizce dil modeli geliştirme ve eğitim projesidir.** Veri yönetişimi, tokenizer eğitimi, decoder-only Transformer eğitimi, doğrulanabilir model paketleri, yerel çıkarım, açık rızaya dayalı geri bildirim, kontrollü sürekli öğrenme ve güvenlik sınırları belirlenmiş bildirimsel modül sistemini tek bir geliştirme hattında birleştirir.

- **Hedef diller:** Türkçe ve İngilizce
- **Hedef mimari ölçeği:** 1.5B sınıfı; otomatik ilk kurulum modeli yaklaşık 1.8M parametreli bootstrap yapılandırmasıdır
- **Yapılandırılmış parametre sayısı:** 1.523.714.048
- **Başlatma:** Rastgele başlangıç ağırlıkları; temel model olarak üçüncü taraf önceden eğitilmiş ağırlık kullanılmaz
- **Artifact biçimi:** SafeTensors çıkarım artifact'i ve weights-only `arachne_small.pt` PyTorch checkpoint'i
- **Mevcut yayın durumu:** Eğitim ve çalışma zamanı yazılımı depoda; ilk gerçek self-taught eğitimi yerel sunucu başlatıldığında otomatik yürütülür, yerel ağırlıklar Git'e gönderilmez

[🇬🇧 English README](README.md) · [Kurulum ve çalıştırma kılavuzu](KURULUM.md)

> **Self-taught çalışma şekli:** `arachne-small serve` arka planda Türkçe ve İngilizce Wikipedia sayfalarını sayfa bazında lisans kontrolüyle toplar, tokenizer/modeli sıfırdan eğitir ve `arachne_small.pt` üretir. İlk model küçük bir bootstrap modelidir; 1.5B değildir. Eğitim ve doğrulama tamamlanmadan model hazır sayılmaz. Süreç ve sınırlar için [Self-Taught](docs/SELF_TAUGHT.md) belgesine bakın.

---

## İçindekiler

- [Projenin kapsamı](#projenin-kapsamı)
- [Mimari](#mimari)
- [Kabiliyetler](#kabiliyetler)
- [Kurulum](#kurulum)
- [Korpus ve tokenizer hattı](#korpus-ve-tokenizer-hattı)
- [Eğitim ve checkpoint](#eğitim-ve-checkpoint)
- [Artifact doğrulama ve çıkarım](#artifact-doğrulama-ve-çıkarım)
- [HTTP API](#http-api)
- [Bildirimsel modül sistemi](#bildirimsel-modül-sistemi)
- [Rızaya dayalı geri bildirim ve sürekli öğrenme](#rızaya-dayalı-geri-bildirim-ve-sürekli-öğrenme)
- [Komut satırı referansı](#komut-satırı-referansı)
- [Güvenlik ve gizlilik](#güvenlik-ve-gizlilik)
- [Yapılandırma](#yapılandırma)
- [Test ve yayın durumu](#test-ve-yayın-durumu)
- [Depoların sorumlulukları](#depoların-sorumlulukları)
- [Dokümantasyon](#dokümantasyon)
- [Lisans](#lisans)

## Projenin kapsamı

ArachneAISmall, kullanım hakkı belgelenmiş eğitim verisinden sürümlenebilir ve doğrulanabilir dil modeli paketine kadar tekrarlanabilir bir süreç sunmayı hedefler. Model, üçüncü taraf önceden eğitilmiş checkpoint üzerinden uyarlanmak yerine rastgele başlangıç ağırlıklarıyla eğitilir.

Model geliştirme ve çalışma zamanı birbirinden ayrılmıştır. Eğitim hattı tanımlı bir artifact sözleşmesi üretir; çıkarım katmanı ağırlıkları yüklemeden önce bu sözleşmeyi doğrular. Kullanıcı konuşmaları gizlice eğitim verisi olarak toplanmaz ve geri bildirimler otomatik olarak doğru kabul edilmez.

## Mimari

Yapılandırılmış decoder-only Transformer mimarisi:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped-Query Causal Attention (GQA)
- SwiGLU ileri beslemeli katmanlar
- Residual bağlantılar
- Yapılandırılabilir model boyutları ve bağlam uzunluğu
- İsteğe bağlı gradient checkpointing
- SentencePiece BPE tokenizer

Yapılandırılmış parametre sayısını hesaplayın:

```bash
arachne-small count-parameters --config configs/1_5b.yaml
```

1.5B yapılandırması yüksek hesaplama kaynağı gerektiren bir hedeftir. Gerçek donanım ihtiyacı, hız, kalite ve dil kabiliyetleri ancak gerçek eğitim sonrasında ölçülebilir.

## Kabiliyetler

### Veri yönetişimi ve ön işleme

- Yapılandırılabilir metin alanıyla UTF-8 JSONL belgelerini okur.
- Veri kaynağı ve beyan edilen lisans/izin bilgisini kaydeder.
- Unicode ve boşluk normalizasyonu uygular.
- Disk tabanlı işlemeyle mükerrer belgeleri ayıklar.
- Deterministik, belge seviyesinde eğitim/doğrulama ayrımı yapar.
- Tokenizer eğitimi için metin dışa aktarır.
- SentencePiece BPE tokenizer'ını yalnızca eğitim bölümü üzerinden eğitir.
- Eğitim ve doğrulama verisini little-endian uint32 token akışlarına kodlar.
- Token dosyalarının yanında SHA-256 ve tokenizer/kaynak metaverisi üretir.
- Eğitim başlamadan önce token akışı ve tokenizer kimliğini doğrular.

### Model eğitimi

- PyTorch tabanlı decoder-only Transformer eğitim motoru.
- AdamW optimizer.
- Gradient accumulation ve gradient clipping.
- Warmup ve cosine learning-rate schedule.
- Yapılandırılmış aralıklarla doğrulama değerlendirmesi.
- İsteğe bağlı gradient checkpointing ve precision yapılandırması.
- Atomik checkpoint kaydı ve doğrulanmış kaldığı yerden devam.
- Doğrulanmış artifact üzerinden yeni optimizer durumu ile devam eğitimi.

### Artifact bütünlüğü ve uyumluluk

- Çıkarım ağırlıkları için SafeTensors; çıkarım yüklemesinde pickle tabanlı ağırlık yükleme kullanılmaz.
- Sürümlenmiş artifact ve manifest biçimleri.
- Model yapılandırması, tokenizer ve ağırlıklar için SHA-256 doğrulaması.
- Tokenizer sözlüğü ve özel token kimliği denetimleri.
- Mimari parametre sayısının tam doğrulanması.
- Tensor adları ve boyutlarının yükleme öncesi kontrolü.
- Strict state-dictionary yükleme.
- Artifact dosyalarında yol geçişi ve sembolik bağlantı reddi.
- İçerik adresli yayınlar ve doğrulanmış geri alma.

### Yerel çıkarım ve web arayüzü

- FastAPI uygulaması ve Uvicorn sunucusu.
- Türkçe/İngilizce tarayıcı arayüzü.
- Prompt ve sınırlandırılmış konuşma geçmişi doğrulaması.
- Üretim uzunluğu, temperature, top-k ve dil tercihi ayarları.
- Model hazır olma durumu, cihaz, CUDA kullanılabilirliği ve model metaverisi.
- `/docs` adresinde OpenAPI dokümantasyonu.
- Doğrulanmış artifact yoksa açık kullanılamaz durumu; sahte model yanıtı üretilmez.

### Bildirimsel modül sistemi

ArachneAISmall, küçük bildirimsel metin işleme modüllerini oluşturabilir, doğrulayabilir, etkinleştirebilir, keşfedebilir, listeleyebilir ve çalıştırabilir.

- Katı JSON manifest şeması ve semantik sürüm doğrulaması.
- Modül kimliği, açıklaması, işlem sayısı, test sayısı ve girdi boyutu sınırları.
- Desteklenen işlemler: `trim`, `lowercase`, `uppercase`, `prepend`, `append` ve `replace_literal`.
- Etkinleştirme öncesinde deterministik girdi/beklenen çıktı testleri zorunludur.
- Aday modüller etkinleştirilmeden önce doğrulanır ve test edilir.
- İçerik SHA-256 ile tanımlanır.
- Registry keşfi, değişmez 40 karakterlik Git commit yolunu kullanan HTTPS `raw.githubusercontent.com` kaynaklarıyla sınırlandırılmıştır.
- Yönlendirmeler, güvenilmeyen sunucular, boyut sınırını aşan belgeler, bozuk manifestler ve eksik/geçersiz kaynak digest değerleri reddedilir.
- Hatalı bir registry girdisi denetlenir; sonraki girdilerin işlenmesini engellemez.
- Ret kayıtları yerel JSONL denetim dosyasına eklenir; etkinleştirme başarısız olduğunda başarısız aday artifact temizlenir.
- Keyfî Python, shell, import, `eval` veya üretilmiş çalıştırılabilir kod modül biçiminin parçası değildir.

Modül sistemi sınırlı bir bildirimsel dönüşüm motorudur; sınırsız bir eklenti çalıştırma ortamı değildir.

### Rızaya dayalı geri bildirim ve kontrollü sürekli öğrenme

- Geri bildirim yalnızca açık kullanıcı rızasıyla yerel SQLite veritabanına kaydedilir.
- Prompt, yanıt, düzeltme, puan ve dil alanları doğrulanır.
- Belirgin kimlik bilgisi/iletişim verisi desenleri ve düşük sinyalli tekrarlar reddedilir.
- Operatör bekleyen kayıtları inceleyebilir ve açıkça onaylayabilir veya reddedebilir.
- İnceleme kararları değiştirilemez; reddedilen kayıt sessizce yeniden onaylanamaz.
- Yalnızca incelenmiş, onaylanmış, rızalı ve puan eşiğini geçen düzeltmeler dışa aktarılabilir.
- Dışa aktarılan kayıtlar bütünlük/kaynak metaverisi taşır.
- Aday model doğrulanmış temel artifact üzerinden ince ayar alır ve aynı sabit doğrulama token akışında temel modelle karşılaştırılır.
- Yapılandırılmış doğrulama kaybı gerileme eşiği aşılırsa yayınlama engellenir.
- Başarılı sürümler içerik adreslidir ve doğrulama sonrası geri alınabilir.
- Başarısız çevrimler geri bildirim kayıtlarını tüketmez; kayıtlar yalnızca başarılı yayınlamadan sonra tüketilmiş olarak işaretlenir.

Bu iş akışı canlı sohbet trafiğini gelişigüzel eğitim verisi olarak kullanmaz ve çalışan çıkarım sürecindeki modeli otomatik olarak değiştirmez.

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- Git
- Hedef CPU veya CUDA ortamıyla uyumlu PyTorch
- Eğitim iş akışları için kullanım hakkı belgelenmiş UTF-8 JSONL verisi

### Depoyu indirme ve sanal ortam oluşturma

**Windows PowerShell**

```powershell
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / WSL / macOS**

```bash
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Hedef donanıma uygun PyTorch sürümünü kurun. Yalnızca CPU kullanımı için:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA kullanacaksanız kurulu sürücü ve platformla eşleşen komutu [resmî PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) seçin.

Projeyi ve geliştirme/test bağımlılıklarını kurun:

```bash
python -m pip install -e ".[dev]"
```

Kurulumu doğrulayın:

```bash
python -c "import torch, fastapi, sentencepiece, safetensors; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
arachne-small --help
```

Windows ve Linux için tam kurulum adımları: [KURULUM.md](KURULUM.md).

## Korpus ve tokenizer hattı

Her JSONL satırı, boş olmayan bir `text` alanı (veya `--text-field` ile seçilen alan) içeren nesne olmalıdır:

```json
{"text":"Eğitim kullanım hakkı doğrulanmış bir metin belgesi."}
```

Tekrarlanabilir veri ayrımı yapın ve kaynak bilgisini kaydedin:

```bash
arachne-small prepare-corpus \
  --input data/source.jsonl \
  --output-dir data/split \
  --validation-fraction 0.01 \
  --seed 1337 \
  --source "Veri kaynağı ve sürümü" \
  --license "Lisans veya yazılı izin"
```

Eğitim metnini dışa aktarın ve tokenizer'ı yalnızca eğitim bölümüyle eğitin:

```bash
arachne-small export-text --input data/split/train.jsonl --output data/train-corpus.txt
arachne-small tokenizer --input data/train-corpus.txt --output-prefix artifacts/tokenizer/arachne --vocab-size 32000
```

Her iki bölümü de aynı tokenizer ile kodlayın:

```bash
arachne-small prepare-data --input data/split/train.jsonl --output data/processed/train.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
arachne-small prepare-data --input data/split/validation.jsonl --output data/processed/validation.bin --tokenizer artifacts/tokenizer/arachne.model --eos-id 3
```

Veri hattı token sayısı, kaynak ve token checksum değerleri, tokenizer kimliği, sözlük boyutu ve özel token kimliklerini içeren yan metaveri dosyaları üretir. Eğitim optimizasyonundan önce bu dosyaları doğrular.

## Eğitim ve checkpoint

1.5B hedef yapılandırmasıyla eğitim:

```bash
arachne-small train \
  --config configs/1_5b.yaml \
  --train-data data/processed/train.bin \
  --validation-data data/processed/validation.bin \
  --tokenizer artifacts/tokenizer/arachne.model
```

`configs/dev.yaml`, yalnızca eşleşen tokenizer ve token verisiyle küçük yerel yazılım kontrolleri içindir. 1.5B yapılandırmasının yerine geçmez ve 1.5B model kalitesini veya performansını doğrulamaz.

Eğitim checkpoint'leri doğrulanmış devam etmeyi destekler. Doğrulanmış bir model artifact üzerinden yeni optimizer durumu ile devam etmek için `--initial-artifact`; uyumlu eğitim checkpoint'inden devam etmek için `--resume` kullanın.

Ham veri kümelerini, kişisel verileri, özel korpusları, yerel checkpoint'leri veya model ağırlıklarını Git'e eklemeyin.

## Artifact doğrulama ve çıkarım

Sunucuya vermeden önce tamamlanmış artifact'i doğrulayın:

```bash
arachne-small verify-artifact --path artifacts/arachne-small-1.5b
```

Artifact; biçimde tanımlı yapılandırma, manifest, tokenizer ve SafeTensors ağırlıklarını içermelidir. Ayrıntılar: [Artifact Format](docs/ARTIFACT_FORMAT.md).

Artifact yolunu tanımlayın ve yerel servisi başlatın:

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

Adresler:

- Web arayüzü: http://127.0.0.1:8000/
- Sağlık: http://127.0.0.1:8000/api/health
- Çalışma zamanı durumu: http://127.0.0.1:8000/api/status
- Etkileşimli API dokümantasyonu: http://127.0.0.1:8000/docs

`ARACHNE_SMALL_ARTIFACT` tanımlanmadan servis sağlık/durum/web arayüzü incelemesi için çalışabilir; metin üretimi kullanılamaz. Başarılı sağlık yanıtı model ağırlıklarının yüklendiğini kanıtlamaz. `/api/status` yanıtında `ready: true` aranmalıdır.

## HTTP API

| Yöntem | Uç nokta | Kabiliyet |
|---|---|---|
| GET | `/api/health` | Süreç sağlık durumu |
| GET | `/api/status` | Model hazır olma, cihaz ve artifact durumu |
| POST | `/api/generate` | Yüklenmiş modelle sınırlandırılmış metin üretimi |
| POST | `/api/feedback` | Açık rızalı düzeltme geri bildirimi kaydı |
| GET | `/api/modules` | Etkin doğrulanmış modülleri listeleme |
| POST | `/api/modules/create` | Bildirimsel modül doğrulama ve etkinleştirme |
| POST | `/api/modules/plan` | Yüklenmiş modelden bildirimsel modül taslağı alıp doğrulama |
| POST | `/api/modules/discover` | Güvenilir registry kaynaklarından modül keşfi |
| POST | `/api/modules/{module_id}/run` | Etkin bildirimsel modülü çalıştırma |
| GET | `/docs` | OpenAPI dokümantasyonu |

Üretim istekleri sınırlandırılmış geçmiş (en fazla 20 mesaj), `max_new_tokens` (1–1024), temperature (0.05–2.0), top-k (0–500) ve `auto`/`tr`/`en` dil seçimini destekler. Sunucu istek şemasını ve toplam geçmiş boyutunu doğrular.

İstek/yanıt sözleşmeleri ve çalışma ayrıntıları: [Inference API](docs/INFERENCE_API.md).

## Bildirimsel modül sistemi

Bir modül manifesti tam olarak beş alan içerir: `id`, `version`, `description`, `operations` ve `tests`. Her modül 1–32 desteklenen işlem ve 2–100 deterministik test vektörü sağlamalıdır. Modül girdisi 100.000 karakterle sınırlandırılmıştır.

Örnek manifest:

```json
{
  "id": "text-cleaner",
  "version": "1.0.0",
  "description": "Metindeki boşlukları ve harf biçimini düzenler.",
  "operations": [
    {"op": "trim"},
    {"op": "lowercase"}
  ],
  "tests": [
    {"input": "  HELLO  ", "expected": "hello"},
    {"input": " Arachne ", "expected": "arachne"}
  ]
}
```

Doğrulanmış modül oluşturma ve çalıştırma:

```bash
arachne-small module-create --spec module.json --registry-dir data/modules
arachne-small module-list --registry-dir data/modules
arachne-small module-run --registry-dir data/modules --id text-cleaner --input "  HELLO  "
```

Değişmez registry index URL'lerinden keşif:

```bash
arachne-small module-discover \
  --registry-url "https://raw.githubusercontent.com/OWNER/REPO/<40-character-commit>/registry.json" \
  --registry-dir data/modules
```

Registry girdileri değişmez raw GitHub commit URL'sine ve doğrulanabilir SHA-256 kaynak digest değerine sahip olmalıdır. Yönlendirmeler ve güvenilmeyen sunucular reddedilir. Ret geçmişi yerel olarak korunur. Modül biçimi keyfî kaynak kodu yüklemez veya çalıştırmaz.

Model destekli planlama, yalnızca doğrulanmış eğitilmiş model yüklüyken `POST /api/modules/plan` üzerinden kullanılabilir. Üretilen JSON yine aynı yerel şema ve deterministik testlerden geçmek zorundadır; model çıktısı doğrulama olmadan güvenilir veya etkin kabul edilmez.

## Rızaya dayalı geri bildirim ve sürekli öğrenme

Geri bildirim API'si ve CLI açık rıza gerektirir. Geri bildirim yerel SQLite veritabanında kalır ve bu proje tarafından dışarı yüklenmez. Dışa aktarma öncesi operatör incelemesi zorunludur.

Örnek inceleme akışı:

```bash
arachne-small feedback-list --database data/feedback.sqlite3 --limit 100
arachne-small feedback-review --database data/feedback.sqlite3 --id 12 --approve
arachne-small feedback-review --database data/feedback.sqlite3 --id 13 --reject
arachne-small feedback-export --database data/feedback.sqlite3 --output data/approved-feedback.jsonl --min-rating 4
```

Kontrollü öğrenme çevrimi; doğrulanmış temel artifact, eşleşen tokenizer/doğrulama akışı, eğitim yapılandırması ve yeni bir çalışma dizini gerektirir:

```bash
arachne-small learn-cycle \
  --database data/feedback.sqlite3 \
  --base-artifact artifacts/arachne-small-1.5b \
  --config configs/1_5b.yaml \
  --validation-data data/processed/validation.bin \
  --work-dir artifacts/learning-cycle-001 \
  --release-dir artifacts/releases
```

Aday, aynı doğrulama akışında temel modelle karşılaştırılır. İzin verilen doğrulama kaybı gerilemesinin varsayılan değeri sıfırdır. Yalnızca doğrulamadan geçen aday yayınlanır. Geri alma, yayın işaretçisini değiştirmeden önce hedef sürümü doğrular:

```bash
arachne-small rollback-release --release-dir artifacts/releases
```

Çıkarım süreci artifact'i başlangıçta yükler. Yayınlanmış veya geri alınmış sürümü kullanmak için sunucuyu seçilen release diziniyle yeniden başlatın. Çalışma anında sıcak model değiştirme (hot swap) uygulanmaz.

## Komut satırı referansı

| Komut | İşlev |
|---|---|
| `prepare-corpus` | Korpus normalizasyonu, mükerrer temizliği, ayrımı ve kaynak kaydı |
| `export-text` | JSONL metnini tokenizer eğitimi için dışa aktarma |
| `tokenizer` | SentencePiece BPE eğitimi |
| `prepare-data` | JSONL verisini bütünlük metaverisiyle token akışına kodlama |
| `train` | Model eğitimi veya devam eğitimi |
| `count-parameters` | Yapılandırmadan parametre sayısını hesaplama |
| `verify-artifact` | Model artifact doğrulaması |
| `serve` | API ve web arayüzünü başlatma |
| `feedback-add/list/review/export` | Rızalı düzeltmeleri toplama ve yönetme |
| `evaluate-artifact` | Sabit doğrulama akışında artifact değerlendirme |
| `promote-candidate` | Adayı baseline kapısından geçirip yayınlama |
| `rollback-release` | Doğrulanmış sürümü geri yükleme |
| `learn-cycle` | Kontrollü geri bildirimle ince ayar iş akışı |
| `module-create/discover/list/run` | Bildirimsel modül yönetimi |

Komut seçenekleri için `arachne-small --help` veya `arachne-small <command> --help` kullanın.

## Güvenlik ve gizlilik

- Varsayılan bağlantı yalnızca yerel loopback adresinedir (`127.0.0.1:8000`).
- Uzak bağlantı `ARACHNE_SMALL_ALLOW_REMOTE=1` ile açıkça etkinleştirilmedikçe kapalıdır.
- Uzak mod en az 32 karakterlik benzersiz `ARACHNE_SMALL_API_TOKEN` gerektirir; korumalı API yolları Bearer kimlik doğrulaması ister.
- Uzak dağıtım öncesinde TLS, kimlik doğrulamalı reverse proxy, ağ kısıtlamaları ve hız/bağlantı limitleri kullanın.
- API yanıtlarında anti-sniffing, anti-framing, no-referrer ve no-store başlıkları bulunur.
- İstek gövdeleri, prompt, geçmiş, modül manifestleri ve geri bildirim alanları boyut ve şema açısından sınırlandırılır.
- Geri bildirim toplama açık rıza ve yerel kayıt gerektirir.
- Eğitim verisinin kaynak ve kullanım haklarını operatör belgelemelidir.
- Artifact yolları, checksum, tokenizer metaverisi, tensor adları ve boyutları çıkarım öncesi doğrulanır.
- Registry keşfi sunucu izin listesi, değişmez commit URL'leri, digest kontrolü, yönlendirme engeli ve indirme sınırları kullanır.
- Bildirimsel modüller keyfî Python veya shell kodu çalıştıramaz.
- Token, özel veri kümesi, kullanıcı verisi, checkpoint ve model ağırlıklarını herkese açık Git geçmişine koymayın.

Uzak dağıtım operatörün sorumluluğundadır. Geliştirme sunucusunu doğrudan genel internete açmayın.

## Yapılandırma

| Değişken | Varsayılan | İşlev |
|---|---|---|
| `ARACHNE_SMALL_ARTIFACT` | tanımsız | Doğrulanmış çıkarım artifact dizini |
| `ARACHNE_SMALL_HOST` | `127.0.0.1` | Bağlanılacak adres |
| `ARACHNE_SMALL_PORT` | `8000` | TCP portu |
| `ARACHNE_SMALL_ALLOW_REMOTE` | `0` | Uzak bağlantı için açık izin |
| `ARACHNE_SMALL_API_TOKEN` | tanımsız | Bearer token; uzak bağlantıda zorunlu |
| `ARACHNE_SMALL_FEEDBACK_DB` | `data/feedback.sqlite3` | Yerel geri bildirim veritabanı |
| `ARACHNE_SMALL_MODULE_DIR` | `data/modules` | Yerel modül registry ve denetim verisi |
| `ARACHNE_SMALL_MODULE_REGISTRIES` | tanımsız | Başlangıç keşfi için noktalı virgülle ayrılmış registry URL'leri |
| `ARACHNE_SMALL_MODULE_SCAN_INTERVAL` | `3600` | Modül keşif aralığı |
| `ARACHNE_SMALL_AUTO_LEARN` | `0` | İsteğe bağlı arka plan öğrenme işçisini etkinleştirir |
| `ARACHNE_SMALL_LEARN_INTERVAL` | `21600` | Öğrenme işçisi aralığı |
| `ARACHNE_SMALL_TRAINING_CONFIG` | tanımsız | Otomatik öğrenme için eğitim yapılandırması |
| `ARACHNE_SMALL_VALIDATION_TOKENS` | tanımsız | Otomatik öğrenme için sabit doğrulama token akışı |
| `ARACHNE_SMALL_RELEASE_DIR` | `artifacts/releases` | İçerik adresli sürüm dizini |

Otomatik öğrenme, açıkça etkinleştirilip doğru yapılandırılmadıkça kapalıdır. Doğrulanmış temel artifact, eğitim yapılandırması ve doğrulama token'ları gerektirir. Ayrıntılar: [Inference API](docs/INFERENCE_API.md) ve [Kontrollü Sürekli Öğrenme](docs/CONTINUAL_LEARNING.md).

## Test ve yayın durumu

Yerel testleri çalıştırın:

```bash
python -m pytest -q --tb=short
```

Bu komut testleri yerel bilgisayarda çalıştırır; GitHub Actions gerektirmez. Test sonucu yalnızca komutun çalıştırıldığı kesin kod revizyonu ve ortam için raporlanmalıdır.

Depo şu anda yazılım ve doğrulama hattını içerir; yayımlanmış eğitilmiş model içermez. Yazılım testleri dil kalitesi, doğruluk, güvenlik, benchmark performansı, eğitimin tamamlandığı veya çıkarım hızını kanıtlamaz. Eğitilmiş sürüm için gerçek eğitim, veri kaynağı kayıtları, ölçümlenmiş değerlendirme metrikleri, artifact doğrulaması ve donanım ölçümleri gereklidir.

## Depoların sorumlulukları

- **ArachneAISmall:** Model mimarisi, veri hazırlama, tokenizer, eğitim, değerlendirme, artifact, yerel çıkarım, geri bildirim yönetişimi, kontrollü öğrenme ve bildirimsel modüller.
- **ArachneAIOpen:** Belgelenmiş artifact sözleşmesini kullanan ayrı çalışma zamanı projesi.
- **ArachneAI:** Ayrı ticari SaaS ürünü.

## Dokümantasyon

- [Kurulum ve çalıştırma kılavuzu](KURULUM.md)
- [Model Kartı](docs/MODEL_CARD.md)
- [Veri Kümesi Politikası](docs/DATASET_POLICY.md)
- [Artifact Biçimi](docs/ARTIFACT_FORMAT.md)
- [Inference API](docs/INFERENCE_API.md)
- [Kontrollü Sürekli Öğrenme](docs/CONTINUAL_LEARNING.md)

## Lisans

Kaynak kodu [Apache License 2.0](LICENSE) ile lisanslanmıştır. Depo kodunun lisansı; harici veri kümelerini, üçüncü taraf varlıkları veya gelecekte yayımlanacak eğitilmiş model ağırlıklarını kendiliğinden kapsamaz. Bu materyaller için ayrıca geçerli lisans ve kullanım koşulları gerekir.


## Otomatik Self-Taught Öğrenme

`arachne-small serve` başlatıldığında self-taught işçisi varsayılan olarak arka planda çalışır. Türkçe ve İngilizce Wikipedia REST API'lerinden sayfa bazında lisansı doğrulanabilen CC BY-SA 3.0/4.0 metinleri toplar; sayfa ve revision atıflarını korur; veri kümesini ayırır; tokenizer'ı eğitir; ilk küçük Transformer modelini sıfırdan eğiterek `arachne_small.pt` oluşturur. Sonraki döngülerde yeni verilerle aday model eğitir, baseline ile aynı doğrulama kümesinde karşılaştırır ve yalnızca kayıp değeri gerilemeyen adayı etkinleştirir.

- İlk bootstrap yaklaşık 1.8M parametredir; 1.5B hedef model değildir.
- Varsayılan döngü aralığı 6 saattir.
- Ham sohbetler ve modelin kendi ürettiği yanıtlar otomatik eğitim verisi yapılmaz.
- `GET /api/status` self-taught durumunu gösterir.
- Yerel korpus ve model ağırlıkları Git'e eklenmez.

Ayrıntılar: [docs/SELF_TAUGHT.md](docs/SELF_TAUGHT.md).
