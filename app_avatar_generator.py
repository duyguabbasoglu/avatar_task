import gradio as gr
import edge_tts
import asyncio
import os
import time
import shutil
import torch
from omegaconf import OmegaConf

# --- MuseTalk Real-Time Integration ---
import scripts.realtime_inference as rt_infer
import musetalk.utils.utils as musetalk_utils
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor

WORKSPACE_DIR = "/Users/dyliax/Desktop/musetalkdemo"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "results", "custom_avatar_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model references
vae = None
unet = None
pe = None
timesteps = None
audio_processor = None
whisper = None
weight_dtype = None
fp = None
device = None
models_loaded = False

def init_musetalk():
    global vae, unet, pe, timesteps, audio_processor, whisper, weight_dtype, fp, device, models_loaded
    if models_loaded:
        return
        
    print("\\n[SİSTEM] MuseTalk Modelleri Belleğe Yükleniyor... Lütfen Bekleyin.")
    
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
    
    models_loaded = True
    print("[SİSTEM] Modeller başarıyla yüklendi! Video üretimi artık çok daha hızlı olacak.")


async def generate_speech(text: str, output_path: str):
    voice = "tr-TR-AhmetNeural" 
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def generate_video(image_path, text, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Lütfen bir avatar resmi yükleyin.")
    if not text.strip():
        raise gr.Error("Lütfen avatara söyletmek için bir metin girin.")
        
    if not models_loaded:
        progress(0.1, desc="Yapay zeka modelleri belleğe alınıyor (Sadece ilk sefer)...")
        init_musetalk()
        
    ts = int(time.time())
    
    progress(0.2, desc="Ses sentezleniyor (TTS)...")
    audio_path = os.path.join(OUTPUT_DIR, f"audio_{ts}.mp3")
    asyncio.run(generate_speech(text, audio_path))
    
    progress(0.4, desc="Avatar analiz ediliyor ve video üretiliyor (Bellekten)...")
    
    # Her yeni istekte (özel avatar için) avatar_instance dynamik olarak yaratılıp cache'e atılmalı
    # Bu sayede aynı resmi 2. kez gönderirseniz daha da hızlı sürer.
    try:
        avatar_instance = rt_infer.Avatar(
            avatar_id=f"custom_avatar_{ts}",
            video_path=image_path,
            bbox_shift=0,
            batch_size=4,
            preparation=True
        )
        
        if not getattr(avatar_instance, "input_latent_list_cycle", None):
            raise gr.Error("Yüz bulunamadı! Turkcell logosu gibi yüzü olmayan nesneler harekete geçirilemez. Lütfen insan figürü veya net yüze sahip bir görsel yükleyin.")
        
        vid_basename = f"avatar_demo_{ts}"
        
        avatar_instance.inference(
            audio_path=audio_path,
            out_vid_name=vid_basename,
            fps=25,
            skip_save_images=False
        )
        
        progress(0.9, desc="Video kopyalanıyor...")
        video_out_path = os.path.join(avatar_instance.video_out_path, vid_basename + ".mp4")
        
        # Orijinal istenen "avatar_demo.mp4" adıyla da Desktop'a kaydedelim
        desktop_dest = os.path.expanduser("~/Desktop/avatar_demo.mp4")
        shutil.copy(video_out_path, desktop_dest)
        
        progress(1.0, desc="Tamamlandı!")
        return video_out_path, video_out_path
        
    except Exception as e:
        print(f"GİZLİ SİSTEM HATASI OLUŞTU (Kullanıcıya Gösterilmeyecek): {e}")
        raise gr.Error("Sistemde geçici bir yoğunluk var veya görseliniz yapay zeka analiz standartlarına uygun bulunmadı. Lütfen net bir insan yüzüne sahip yeni bir görsel ile tekrar deneyin.")


def build_ui():
    import base64
    logo_path = "/Users/dyliax/Desktop/musetalkdemo/turkcell_logo_nobg.png"
    logo_src = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            b64_logo = base64.b64encode(img_file.read()).decode()
            logo_src = f"data:image/png;base64,{b64_logo}"

    custom_css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        background-color: #1e293b !important;
    }
    .turkcell-header {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        padding: 5px;
        background-color: transparent;
        margin-bottom: 5px;
    }
    .turkcell-header-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        height: 100%;
    }
    .turkcell-title {
        font-weight: 700;
        font-size: 28px;
        color: #ffc900 !important;
        margin-top: 0;
        margin-bottom: 5px;
    }
    .turkcell-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.0em;
        margin-top: 0;
        margin-bottom: 0;
    }
    .primary-btn {
        background-color: #ffc900 !important;
        color: #1e293b !important;
        font-weight: bold !important;
        border: none !important;
        height: 45px;
    }
    .primary-btn:hover {
        background-color: #e6b500 !important;
    }
    label {
        color: #94a3b8 !important;
    }
    h3 {
        color: #e2e8f0 !important;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    """
    
    with gr.Blocks(title="Turkcell Yapay Zeka Dijital Asistan", css=custom_css, theme=gr.themes.Default(neutral_hue="slate")) as demo:
        gr.HTML(
            f"""
            <div class="turkcell-header">
                <img src="{logo_src}" alt="Turkcell Logo" style="height: 100px; margin-right: 20px;">
                <div class="turkcell-header-content">
                    <h1 class="turkcell-title">Dijital Asistan Stüdyosu</h1>
                    <p class="turkcell-subtitle">Müşteri deneyimini yapay zeka ile yeniden tanımlayın!<br/>Kurumsal metinlerinizi saniyeler içinde doğal dudak senkronizasyonlu videolara dönüştürün.</p>
                </div>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Sözcü / Asistan Görseli Yükle")
                avatar_input = gr.Image(label="Avatar Seç", type="filepath", height=280)
                
                gr.Markdown("### 2. Canlandırılacak Metin")
                text_input = gr.Textbox(
                    placeholder="Örn: Turkcell olarak yapay zekâyı...", 
                    label="Asistanın Okuyacağı Metin", 
                    lines=3
                )
                
                submit_btn = gr.Button("Asistan Videosunu Oluştur 🚀", variant="primary", elem_classes=["primary-btn"])
                
            with gr.Column(scale=1):
                gr.Markdown("### 3. Jenerasyon Sonucu")
                video_out = gr.Video(label="Canlandırılmış Video", format="mp4", height=280)
                download_btn = gr.File(label="Videoyu İndir")

        submit_btn.click(
            fn=generate_video,
            inputs=[avatar_input, text_input],
            outputs=[video_out, download_btn]
        )
        
    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
