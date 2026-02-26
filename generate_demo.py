import asyncio
import os
import shutil
from app_chat import generate_speech, AVATAR_IMAGE_PATH

async def main():
    text = "İşyerinde kesinti oldu. Ümraniye'de çok varmış, durumu hemen inceliyoruz."
    audio_path = "/Users/dyliax/Desktop/musetalkdemo/results/demo_audio.mp3"
    print("Ses üretiliyor...")
    await generate_speech(text, audio_path)
    
    print("MuseTalk inference float16 ile başlatılıyor...")
    # YAML oluştur
    temp_yaml = "/Users/dyliax/Desktop/musetalkdemo/results/demo_inference.yaml"
    yaml_content = f"""
task_0:
  video_path: "{AVATAR_IMAGE_PATH}"
  audio_path: "{audio_path}"
  bbox_shift: 0
"""
    with open(temp_yaml, 'w') as f:
        f.write(yaml_content)
        
    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", temp_yaml,
        "--result_dir", "/Users/dyliax/Desktop/musetalkdemo/results/demo_final",
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15",
        "--use_float16",
        "--ffmpeg_path", "ffmpeg"
    ]
    
    import subprocess
    subprocess.run(cmd, check=True)
    
    import glob
    videos = glob.glob("/Users/dyliax/Desktop/musetalkdemo/results/demo_final/v15/*.mp4")
    # _concat olan dosyaları elemenin pratik yolu
    videos = [v for v in videos if '_concat' not in v]
    
    if videos:
        final_dest = "/Users/dyliax/Desktop/Teknocan_Ornek_Video.mp4"
        shutil.copy(videos[0], final_dest)
        print(f"\\n>>> BAŞARILI! Örnek video masaüstüne kaydedildi: {final_dest}")
    else:
        print("Video bulunamadı.")

if __name__ == "__main__":
    asyncio.run(main())
