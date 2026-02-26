import asyncio
import os
import subprocess
import glob
from pathlib import Path
import shutil
import edge_tts
import gradio as gr

# Sabitler ve yollar
WORKSPACE_DIR = "/Users/dyliax/Desktop/musetalkdemo"
AVATAR_IMAGE_PATH = os.path.join(WORKSPACE_DIR, "avatar", "teknocan_avatar.png")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "results", "chat_output")
FFMPEG_PATH = "ffmpeg"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKSPACE_DIR, "avatar"), exist_ok=True)


if not os.path.exists(AVATAR_IMAGE_PATH):
    import urllib.request
    urllib.request.urlretrieve("https://images.unsplash.com/photo-1560250097-0b93528c311a?q=80&w=600&auto=format&fit=crop", AVATAR_IMAGE_PATH)


async def generate_speech(text: str, output_path: str):
    voice = "tr-TR-AhmetNeural" 
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

#mock llm
def generate_llm_response(user_message: str) -> str:
    
    responses = [
        "Size nasıl yardımcı olabilirim?",
        "Bu konuda hemen bilgi veriyorum.",
        "Anladım, Turkcell olarak bu durumu çözmek için yanınızdayız.",
        "Harika bir soru! Detaylara geçiyorum."
    ]
    import random
    if "merhaba" in user_message.lower():
        return "Merhaba! Ben Turkcell dijital asistanınız Teknocan. Size nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_message.lower():
        return "Teşekkür ederim, iyiyim. Turkcell dünyasında size destek olmak için buradayım!"
    
    return f"Söylediğiniz '{user_message}' konusunu anladım. {random.choice(responses)}"

def run_musetalk_inference(audio_path: str, image_path: str):
    video_basename = "response_video"
    result_dir = os.path.join(OUTPUT_DIR, "musetalk_res")
    
    command = [
        "python", "-m", "scripts.inference",
        "--inference_config", "configs/inference/test.yaml",
        "--result_dir", result_dir,
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15",
        "--ffmpeg_path", FFMPEG_PATH,
    ]
    

    temp_yaml = os.path.join(OUTPUT_DIR, "temp_inference.yaml")
    yaml_content = f"""
task_0:
  video_path: "{image_path}"
  audio_path: "{audio_path}"
  bbox_shift: 0
"""
    with open(temp_yaml, 'w') as f:
        f.write(yaml_content)

    command[4] = temp_yaml  # inference_config'i kendi yaml'ımızla değiştiriyoruz
    
    print("MuseTalk çalıştırılıyor:", " ".join(command))
    
    try:
        subprocess.run(command, check=True, cwd=WORKSPACE_DIR)
        
        # Sonuç dizininden videoyu bulalım
        res_video_pattern = os.path.join(result_dir, "task_0", f"*.mp4")
        videos = glob.glob(res_video_pattern)
        if videos:
            return videos[0]
        else:
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"MuseTalk hatası: {e}")
        return None

async def chat_interaction(user_text, history):
    """
    Gradio web arayüzünden gelen mesaja yanıt verir, TTS ve MuseTalk ile videoyu oluşturur.
    """
    if not user_text.strip():
        return history, None
        
    print(f"Kullanıcı mesajı: {user_text}")
    
    # 1. LLM Yanıtı oluştur
    response_text = generate_llm_response(user_text)
    print(f"Teknocan yanıtı: {response_text}")
    
    # 2. TTS ile sesi oluştur
    audio_output = os.path.join(OUTPUT_DIR, "current_response.mp3")
    await generate_speech(response_text, audio_output)
    print("Ses üretildi:", audio_output)
    
    # 3. Ses dosyasıyla videoyu oluştur
    video_output = run_musetalk_inference(audio_output, AVATAR_IMAGE_PATH)
    
    # Sohbet geçmişini güncelle
    history.append((user_text, response_text))
    
    return history, video_output

# gradio ui
def build_ui():
    """
    Kullanıcı arayüzünü oluşturur.
    Turkcell kurumsal renklerine (sarı-lacivert) uygun, modern ve sade bir tasarım hedeflenmiştir.
    """
    
    custom_css = """
    #avatar_img img {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-height: 400px;
        object-fit: contain;
    }
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    """
    
    with gr.Blocks(title="Teknocan - Turkcell Dijital Asistan", css=custom_css) as demo:
        gr.Markdown(
            """
            # <center>📱 Teknocan - Turkcell Dijital Asistan</center>
            <center><b>Gerçek zamanlı dudak senkronizasyonu ile konuşan AI avatarınız</b></center>
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=5):
                # Sohbet Geçi̇mi̇şi̇ (Gradio 4+ type='messages' önerir, ancak fallback tuple kullanıyoruz)
                chatbot = gr.Chatbot(label="Sohbet", height=450, bubble_full_width=False)
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Teknocan'a mesajınızı yazın... (Örn: Merhaba Teknocan)", 
                        label="", 
                        show_label=False, 
                        lines=1, 
                        scale=8
                    )
                    submit_btn = gr.Button("Gönder 📩", variant="primary", scale=2)
                    
                gr.Markdown("*Not: MuseTalk video üretimi makinenizin donanımına göre birkaç saniye sürebilir.*")
                
            with gr.Column(scale=4):
                gr.Markdown("### Teknocan Yanıtı")
                # Yanıt geldiğinde video gösterilecek
                video_out = gr.Video(label="Video Yanıt", autoplay=True, show_label=False, format="mp4")
                
                # Başlangıçta statik resmi gösteriyoruz
                image_out = gr.Image(value=AVATAR_IMAGE_PATH, label="Avatar", elem_id="avatar_img", visible=True, show_label=False)

        # Butona tıklandığında veya Enter'a basıldığında chat_interaction çağrılır.
        submit_btn.click(
            fn=lambda t, h: asyncio.run(chat_interaction(t, h)), 
            inputs=[msg, chatbot], 
            outputs=[chatbot, video_out]
        )
        
        # Kullanıcı enter tuşuna basarak da gönderebilir
        msg.submit(
            fn=lambda t, h: asyncio.run(chat_interaction(t, h)), 
            inputs=[msg, chatbot], 
            outputs=[chatbot, video_out]
        )
        
    return demo

if __name__ == "__main__":
    print("Teknocan başlatılıyor... Lütfen bekleyin.")
    app = build_ui()
    # share=False lokalde çalışması içindir. public url için share=True yapın.
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
