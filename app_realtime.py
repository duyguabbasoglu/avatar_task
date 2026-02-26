import asyncio
import os
import time
import argparse
import gradio as gr
import edge_tts
import torch
from omegaconf import OmegaConf

# --- MuseTalk Real-Time Integration ---
import scripts.realtime_inference as rt_infer
import musetalk.utils.utils as musetalk_utils
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor

WORKSPACE_DIR = "/Users/dyliax/Desktop/musetalkdemo"
AVATAR_IMAGE_PATH = os.path.join(WORKSPACE_DIR, "avatar", "teknocan_avatar.png")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "results", "chat_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

avatar_instance = None

def init_musetalk():
    global avatar_instance
    print("\\n[SİSTEM] MuseTalk Gerçek Zamanlı (Real-Time) Modeli Yükleniyor... Lütfen Bekleyin.")
    
    # Argümanları simüle et
    class Args: pass
    args = Args()
    args.version = "v15"
    args.ffmpeg_path = "ffmpeg"
    args.gpu_id = 0
    args.vae_type = "sd-vae"
    args.unet_config = "./models/musetalkV15/musetalk.json"
    args.unet_model_path = "./models/musetalkV15/unet.pth"
    args.whisper_dir = "./models/whisper"
    args.bbox_shift = 0
    args.batch_size = 4
    args.extra_margin = 10
    args.fps = 25
    args.audio_padding_length_left = 2
    args.audio_padding_length_right = 2
    args.parsing_mode = "jaw"
    args.left_cheek_width = 90
    args.right_cheek_width = 90
    args.skip_save_images = False
    
    rt_infer.args = args

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    rt_infer.device = device
    
    vae, unet, pe = musetalk_utils.load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    # Mac MPS için float32, CUDA için float16
    if device.type == "cuda":
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
    else:
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        
    rt_infer.vae = vae
    rt_infer.unet = unet
    rt_infer.pe = pe
    rt_infer.timesteps = torch.tensor([0], device=device)
    
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    rt_infer.audio_processor = audio_processor
    rt_infer.whisper = whisper
    rt_infer.weight_dtype = weight_dtype
    
    fp = FaceParsing(left_cheek_width=int(args.left_cheek_width), right_cheek_width=int(args.right_cheek_width))
    rt_infer.fp = fp
    
    print("[SİSTEM] Avatar arabelleğini kontrol ediyorum (varsa diskten yüklenir).")
    try:
        avatar_instance = rt_infer.Avatar(
            avatar_id="teknocan",
            video_path=AVATAR_IMAGE_PATH,
            bbox_shift=0,
            batch_size=args.batch_size,
            preparation=False  # Hazırlığı önceden yapmış olmalıyız (demo_prep ile)
        )
    except Exception as e:
        print(f"[HATA] Avatar önbelleği bulunamadı, preparation=True ile oluşturuluyor. {e}")
        avatar_instance = rt_infer.Avatar(
            avatar_id="teknocan",
            video_path=AVATAR_IMAGE_PATH,
            bbox_shift=0,
            batch_size=args.batch_size,
            preparation=True
        )
    print("[SİSTEM] MuseTalk başarıyla bellek üzerine yüklendi ve ~2sn gecikme için hazır.")

# --- Sohbet Sistemleri ---

async def generate_speech(text: str, output_path: str):
    voice = "tr-TR-AhmetNeural" 
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def generate_llm_response(user_message: str) -> str:
    import random
    responses = [
        "Size nasıl yardımcı olabilirim?",
        "Bu konuda hemen bilgi veriyorum.",
        "Anladım, Turkcell olarak bu durumu çözmek için yanınızdayız."
    ]
    if "merhaba" in user_message.lower():
        return "Merhaba! Ben Turkcell dijital asistanınız Teknocan. Size nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_message.lower():
        return "Teşekkür ederim, iyiyim. Turkcell dünyasında size destek olmak için buradayım!"
    elif "işyerinde" in user_message.lower() or "kesinti" in user_message.lower():
        return "İşyerinde kesinti oldu. Ümraniye'de çok varmış, durumu hemen inceliyoruz."
    
    return f"Söylediğiniz '{user_message}' konusunu anladım. {random.choice(responses)}"

async def chat_interaction(user_text, history):
    if not user_text.strip():
        return history, None
        
    print(f"\\n--- Yeni İstek: {user_text} ---")
    
    response_text = generate_llm_response(user_text)
    
    audio_output = os.path.join(OUTPUT_DIR, f"resp_{int(time.time())}.mp3")
    await generate_speech(response_text, audio_output)
    
    # Bellekteki MuseTalk Modülasyonunu Kullanarak Gecikmesiz Render
    vid_basename = f"video_{int(time.time())}"
    print("[SİSTEM] Video üretimi başlıyor...")
    start_time = time.time()
    
    avatar_instance.inference(
        audio_path=audio_output,
        out_vid_name=vid_basename,
        fps=25,
        skip_save_images=False
    )
    
    print(f"[SİSTEM] Video üretimi {time.time() - start_time:.2f} saniye sürdü.")
    video_output = os.path.join(avatar_instance.video_out_path, vid_basename + ".mp4")
    
    history.append((user_text, response_text))
    return history, video_output

# --- GRADIO ARAYÜZÜ ---

def build_ui():
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
            <center><b>Gerçek zamanlı hızda dudak senkronizasyonu ile konuşan AI avatarınız</b></center>
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Sohbet", height=450, bubble_full_width=False)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Teknocan'a mesajınızı yazın...", label="", show_label=False, lines=1, scale=8)
                    submit_btn = gr.Button("Gönder 📩", variant="primary", scale=2)
                
                gr.Markdown("*Not: MuseTalk modelleri bellekte yüklendiğinde asistan sadece 1-3 saniyede video yanıt üretir (Real-Time Modu).*")
                
            with gr.Column(scale=4):
                gr.Markdown("### Teknocan Yanıtı")
                video_out = gr.Video(label="Video Yanıt", autoplay=True, show_label=False, format="mp4")
                image_out = gr.Image(value=AVATAR_IMAGE_PATH, label="Avatar", elem_id="avatar_img", visible=True, show_label=False)

        submit_btn.click(
            fn=lambda t, h: asyncio.run(chat_interaction(t, h)), 
            inputs=[msg, chatbot], 
            outputs=[chatbot, video_out]
        )
        msg.submit(
            fn=lambda t, h: asyncio.run(chat_interaction(t, h)), 
            inputs=[msg, chatbot], 
            outputs=[chatbot, video_out]
        )
        
    return demo

if __name__ == "__main__":
    init_musetalk()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
