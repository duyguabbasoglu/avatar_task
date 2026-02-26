## Turkcell Dijital Asistan Projesi

Bu projeyi geliştirirken temel altyapı olarak **[TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)** açık kaynak kütüphanesini kullandım. MuseTalk'un gerçek zamanlı dudak senkronizasyonu yeteneğini alarak, tamamen kurumsal ve stabil bir Turkcell Asistan platformuna dönüştürdüm. 

### Kendi Eklediklerim ve Geliştirdiğim Özellikler

1. **Mac (MPS) ve Stabilite Optimizasyonları**
   - MuseTalk sadece CUDA (Nvidia GPU) üzerine tasarlandığı için Mac bilgisayarlarda çalışmıyordu. Kodlara derinlemesine müdahale ederek tam MPS (Apple Silicon GPU) desteği ekledim.
   - Cihaz çakışmalarını, Float16/Float32 matematik hatalarını giderdim.
2. **Realtime Hızlandırma**
   - Her video oluşturmada AI modellerinin baştan yüklenmesi sorununu çözdüm. Modelleri bellekte cache ederek üretim süresini ciddi oranda düşürdüm.
3. **Türkçe TTS Entegrasyonu**
   - Projenin sadece sessiz çalışmasını engelleyip, arka planda dinamik olarak metinleri Türkçe insan sesine dönüştüren yapıyı edge-tts kodlara bağladım.
4. **Gradio UI**
