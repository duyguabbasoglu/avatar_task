import app_avatar_generator as app
import asyncio
import os

def main():
    print("Initializing models...")
    app.init_musetalk()
    
    text = "Turkcell olarak yapay zekâyı yalnızca operasyonel verimliliği artıran bir teknoloji olarak değil, aynı zamanda müşteri deneyimini yeniden tanımlayan stratejik bir dönüşüm alanı olarak konumlandırıyoruz. Ağ yönetiminden müşteri hizmetlerine, siber güvenlikten kişiselleştirilmiş dijital servislerin geliştirilmesine kadar pek çok alanda yapay zekâ çözümlerini aktif şekilde entegre ediyoruz."
    image_path = "/Users/dyliax/Desktop/musetalkdemo/avatar/avatar.png"
    
    print("Generating video...")
    try:
        def progress(value, desc):
            print(f"[{value*100:.0f}%] {desc}")
            
        final_video, _ = app.generate_video(image_path, text, progress)
        print("Success! Saved to:", final_video)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    main()
