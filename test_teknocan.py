import app_avatar_generator as app
import asyncio
import os

def main():
    print("Initializing models...")
    app.init_musetalk()
    
    text = "Ümraniye bölgesinde ufakça 3 adet kesinti tespit öğrenilmiştir öğretmen ÖĞRETMEN"
    image_path = "avatar/teknocan.webp"
    
    print("Generating video...")
    try:
        # Dummy progress function
        def progress(value, desc):
            print(f"[{value*100:.0f}%] {desc}")
            
        final_video, _ = app.generate_video(image_path, text, progress)
        print("Success! Saved to:", final_video)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    main()
