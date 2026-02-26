# Turkcell Dijital Asistan (Teknocan) Projesi 🚀

Bu projeyi geliştirirken temel altyapı olarak **[TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)** açık kaynak kütüphanesini kullandım. MuseTalk'un gerçek zamanlı dudak senkronizasyonu yeteneğini alarak, tamamen kurumsal ve stabil bir Turkcell Asistan platformuna dönüştürdüm. 

### Kendi Eklediklerim ve Geliştirdiğim Özellikler ✨

MuseTalk'un orijinal altyapısının üzerine kendi yazdığım ve projeye kazandırdığım özellikler şunlardır:

1. **CEO-Ready Profesyonel Arayüz (UI)**
   - Tamamen tek safyalı (single-page), aşağı kaydırmaya gerek kalmayan kompakt bir Gradio arayüzü tasarladım.
   - Turkcell kurumsal renklerini (Lacivert ve Sarı) entegre edip arka plan rengini güncelledim.
   - Turkcell'in saydam (arka plansız) logosunu başlığın yanına hizalayarak "Dijital Asistan Stüdyosu" adında kurumsal bir dashboard oluşturdum.
2. **Mac (MPS) ve Stabilite Optimizasyonları**
   - MuseTalk sadece CUDA (Nvidia GPU) üzerine tasarlandığı için Mac bilgisayarlarda çalışmıyordu. Kodlara derinlemesine müdahale ederek tam **MPS (Apple Silicon GPU)** desteği ekledim.
   - Cihaz çakışmalarını, Float16/Float32 matematik hatalarını giderdim.
3. **Gerçek Zamanlı (Realtime) Hızlandırma**
   - Her video oluşturmada AI modellerinin baştan yüklenmesi sorununu çözdüm. Modelleri bellekte tutarak (cache) üretim süresini ciddi oranda düşürdüm.
4. **Türkçe TTS (Ses Sentezi) Entegrasyonu**
   - Projenin sadece sessiz çalışmasını engelleyip, arka planda dinamik olarak metinleri Türkçe insan sesine dönüştüren yapıyı (edge-tts) kodlara bağladım.
5. **Güvenli Hata Yönetimi**
   - Sistemin "yüz bulamadığında" arkada verdiği çirkin Python loglarını (ZeroDivisionError vb.) yakalayarak, kullanıcıya sarı ve şık bir uyarı çıkaran ("Yüz bulunamadı, farklı görsel deneyin") kapalı devre bir hata yönetimi ekledim.

### Vercel Deployment Adımları 🌐

Projeyi Vercel üzerinden yayınlamak istediğini biliyorum. Yayına alırken ayarlarını şu şekilde yapmalısın:

- **Canlı Link (Hedeflenen):** https://avatar.vercel.app/
- **Root Dosyası (Entrypoint):** `app_avatar_generator.py`

*(Not: Bu proje arka planda çok büyük yapay zeka modelleri ve PyTorch/FFmpeg altyapısı barındırdığı için, Vercel'in standart ücretsiz sunucularında (Serverless Functions 250MB sınırı) boyut ve GPU limiti nedeniyle hata alabilirsin. Eğer Vercel'de `slug` boyutu aşılırsa, kodu Vercel yerine "HuggingFace Spaces" veya Railway/Render gibi Docker tabanlı ve GPU desteği sunabilen bir platforma yüklemen gerekebilir, ancak projenin ana taşıyıcı dosyası her koşulda `app_avatar_generator.py` olacaktır.)*

Sevgilerle! 😊
