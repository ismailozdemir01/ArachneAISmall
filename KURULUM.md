# ArachneAISmall Kurulum ve Çalıştırma Kılavuzu

Bu belge, ArachneAISmall deposunu Windows PowerShell veya Linux/WSL üzerinde indirip kurmak, yerel API/web arayüzünü başlatmak ve çalışma durumunu doğrulamak için hazırlanmıştır.

> **Model durumu:** Depo, model mimarisi ve eğitim/çıkarım yazılımını içerir. Eğitilmiş 1.5B model ağırlıkları bu depoda yayımlanmış değildir. Bu nedenle aşağıdaki kurulum API ve web arayüzünü çalıştırır; doğrulanmış bir model paketi ayrıca sağlanmadıkça metin üretimi etkinleşmez. Uygulama rastgele ağırlıklarla sahte yanıt üretmez.

## 1. Gereksinimler

- Python 3.10 veya üzeri
- Git
- İnternet bağlantısı (kaynak kodunu ve Python bağımlılıklarını indirmek için)
- PyTorch'un hedef sisteme uygun CPU veya CUDA kurulumu
- Geliştirme/test komutları için yeterli disk alanı

Python ve Git sürümlerini kontrol edin:

```powershell
python --version
git --version
```

Python komutu bulunamazsa Windows'ta Python 3 kurulumunu tamamlayıp terminali yeniden açın.

## 2. Depoyu indirme

Projeyi indirmek istediğiniz klasörde:

```powershell
git clone https://github.com/ismailozdemir01/ArachneAISmall.git
cd ArachneAISmall
```

Mevcut bir klonda güncel `main` dalına geçmek için:

```powershell
git switch main
git pull --ff-only origin main
```

## 3. Sanal ortam oluşturma

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PowerShell yürütme ilkesi etkinleştirme betiğini engelliyorsa sistem ilkesini değiştirmek zorunda değilsiniz. Sanal ortamın Python dosyasını doğrudan kullanın:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. PyTorch ve proje bağımlılıklarını yükleme

Önce makinenize uygun PyTorch paketini seçin. CPU kurulumu için:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA kullanacaksanız CUDA sürümünüzle uyumlu kurulum komutunu resmi PyTorch kurulum seçicisinden alın: https://pytorch.org/get-started/locally/

Ardından proje ve geliştirme/test bağımlılıklarını kurun:

```bash
python -m pip install -e ".[dev]"
```

Kurulumun ve temel bağımlılıkların kontrolü:

```bash
python -c "import torch, fastapi, sentencepiece, safetensors; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
arachne-small --help
```

Komut satırı komutu bulunamazsa sanal ortamın etkin olduğundan emin olun. Windows'ta etkinleştirmeden çalıştırmak için:

```powershell
.\.venv\Scripts\arachne-small.exe --help
```

## 5. API ve web arayüzünü başlatma

Model ağırlıkları olmadan da servis durumunu ve web arayüzünü çalıştırabilirsiniz. Depo kök dizininde, sanal ortam etkinken:

```powershell
arachne-small serve
```

Alternatif doğrudan ASGI başlatma komutu:

```powershell
python -m uvicorn arachne_small.server:app --host 127.0.0.1 --port 8000
```

Sunucu terminalini açık bırakın. Varsayılan yerel adresler:

- Web arayüzü: http://127.0.0.1:8000/
- Sağlık kontrolü: http://127.0.0.1:8000/api/health
- Model/çalışma zamanı durumu: http://127.0.0.1:8000/api/status
- OpenAPI arayüzü: http://127.0.0.1:8000/docs

PowerShell'de ikinci bir terminal açıp servis kontrollerini çalıştırın:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/status
```

Beklenen sağlık yanıtı:

```json
{
  "status": "ok",
  "service": "arachne-small"
}
```

Model paketi yapılandırılmamışsa `/api/status` yanıtında `ready: false` ve `state: "artifact_missing"` görülmesi beklenir. Bu, HTTP servisinin çalışmadığı anlamına gelmez; çıkarım için doğrulanmış model paketi bulunmadığını belirtir.

## 6. Doğrulanmış model paketiyle çıkarım

Gerçek metin üretimi için eğitim tamamlanmış ve proje artifact doğrulamasından geçen bir model paketi gerekir. Paket dizininde model ağırlıkları, tokenizer, model yapılandırması ve bütünlük manifesti bulunmalıdır.

Windows PowerShell:

```powershell
$env:ARACHNE_SMALL_ARTIFACT = (Resolve-Path "artifacts\arachne-small-1.5b").Path
arachne-small serve
```

Linux / WSL:

```bash
export ARACHNE_SMALL_ARTIFACT="$PWD/artifacts/arachne-small-1.5b"
arachne-small serve
```

Yukarıdaki dizin yalnızca örnek yoldur. Gerçek, tamamlanmış ve doğrulanabilir artifact dizininizin yolunu kullanın. Henüz böyle bir paketiniz yoksa bu ortam değişkenini tanımlamayın. Sunucu yükleme hatasını loglar; doğrulama başarısız olduğunda model hazır durumuna geçmez.

Sunucu çalışırken:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/status
```

Yanıtta `ready: true` ve `state: "ready"` görülmeden gerçek model çıkarımı hazır kabul edilmemelidir.

## 7. Küçük geliştirme yapılandırması ve gerçek eğitim

`configs/dev.yaml`, hızlı yazılım doğrulaması için küçük bir model yapılandırmasıdır. Bu yapılandırma 1.5B modelin yerine geçmez ve yayımlanabilir model kalitesi kanıtı değildir.

Gerçek eğitim hattı için:

1. Model eğitimi için kullanım hakkı belgelenmiş veri hazırlayın.
2. Her satırında `text` alanı bulunan UTF-8 JSONL dosyası oluşturun.
3. README'deki korpus hazırlama ve tokenizer adımlarını uygulayın.
4. Eğitim ve doğrulama token dosyalarını oluşturun.
5. Uygun donanım ve bellek kapasitesiyle eğitim komutunu çalıştırın.
6. Oluşan artifact paketini doğrulamadan servis için kullanmayın.

Örnek eğitim komutu (önceden hazırlanmış veri dosyalarını gerektirir):

```bash
arachne-small train \
  --config configs/1_5b.yaml \
  --train-data data/processed/train.bin \
  --validation-data data/processed/validation.bin \
  --tokenizer artifacts/tokenizer/arachne.model
```

1.5B yapılandırması yüksek GPU belleği, depolama ve uzun eğitim süresi gerektirir. Eğitim verisi, tokenizer veya model ağırlıkları bu komutla otomatik olarak indirilmez.

## 8. Yerel testleri çalıştırma

Sanal ortam etkinken:

```bash
python -m pytest -q --tb=short
```

Testler yerel makinede çalışır. Bu komut GitHub Actions/CI tetiklemez.

## 9. Güvenli çalışma ayarları

- Varsayılan olarak yalnızca `127.0.0.1` üzerinde çalışın.
- Yerel geliştirme için `ARACHNE_SMALL_ALLOW_REMOTE=1` ayarlamayın.
- Sunucuyu doğrudan internete açmayın.
- Uzak erişim zorunluysa en az 32 karakterlik güçlü ve benzersiz `ARACHNE_SMALL_API_TOKEN` tanımlayın; `ARACHNE_SMALL_ALLOW_REMOTE=1` ayarını yalnızca TLS, kimlik doğrulamalı reverse proxy ve ağ erişim kontrolleri hazır olduğunda kullanın.
- API token'ını kaynak koda, Git'e, ekran görüntülerine veya loglara koymayın.
- Yalnızca lisans ve model eğitimi kullanım hakkı doğrulanmış veri kümelerini kullanın.
- `.env`, ham veri, yerel veritabanı, checkpoint ve model artifact dosyalarını depoya göndermeyin.

## 10. Sorun giderme

### `arachne-small` komutu bulunamıyor

Sanal ortamın etkin olduğunu kontrol edin ve proje kurulumunu tekrarlayın:

```bash
python -m pip install -e ".[dev]"
```

### `ModuleNotFoundError: arachne_small`

Komutları depo kök dizininde çalıştırın ve doğru sanal ortam Python'unu kullandığınızı doğrulayın:

```bash
python -c "import sys, arachne_small; print(sys.executable); print(arachne_small.__file__)"
```

### PyTorch kurulumu başarısız

Python sürümünüz ve işletim sisteminiz için mevcut PyTorch wheel desteğini resmi PyTorch kurulum seçicisinden kontrol edin. CUDA paketi ile kurulu NVIDIA sürücüsü/CUDA ortamı uyumlu olmalıdır. Önce CPU kurulumu ile servis ve yazılım testlerini doğrulayabilirsiniz.

### 8000 portu kullanımda

Başka bir süreç portu kullanıyorsa onu kapatın veya farklı port seçin:

```powershell
$env:ARACHNE_SMALL_PORT = "8001"
arachne-small serve
```

Ardından http://127.0.0.1:8001 adresini açın.

### Servis çalışıyor ama üretim HTTP 503 dönüyor

`/api/status` adresini kontrol edin. `artifact_missing` veya `error` durumu, doğrulanmış model paketinin yüklenmediğini gösterir. Sadece sunucuyu başlatmak model eğitmez; gerçek ve doğrulanmış artifact gereklidir.

### Sunucuyu durdurma

Sunucunun çalıştığı terminale odaklanıp `Ctrl+C` basın.

---

## İlgili belgeler

- [Türkçe proje açıklaması](README_TR.md)
- [İngilizce proje açıklaması](README.md)
- [Inference API ve çalışma zamanı](docs/INFERENCE_API.md)
- [Model kartı](docs/MODEL_CARD.md)
- [Veri kümesi politikası](docs/DATASET_POLICY.md)
- [Artifact biçimi](docs/ARTIFACT_FORMAT.md)
- [Kontrollü sürekli öğrenme](docs/CONTINUAL_LEARNING.md)
