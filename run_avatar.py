import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from gtts import gTTS
    import torch
    import librosa
except ImportError as e:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AvatarGenerator:
    """
    Handles the generation of avatar videos from text input, including
    TTS generation and video synthesis using LongCat-Video.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
        self.longcat_dir = self.base_dir / "LongCat-Video"
        self.output_dir = self.base_dir / "outputs"
        self.assets_dir = self.longcat_dir / "assets" / "avatar"
        self.weights_dir = self.base_dir / "weights" / "LongCat-Video-Avatar"
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


    def check_dependencies(self):
        """Checks if necessary dependencies are installed."""
        missing = []
        try:
            import gtts
        except ImportError:
            missing.append("gtts")
        try:
            import torch
        except ImportError:
            missing.append("torch")
        try:
            import librosa
        except ImportError:
            missing.append("librosa")
            
        if missing:
             logger.error(f"❌ Missing dependencies: {', '.join(missing)}. Please run: pip install -r requirements.txt")
             sys.exit(1)

        if not self.longcat_dir.exists():
            logger.error(f"❌ LongCat-Video submodule not found at {self.longcat_dir}. Please clone it.")
            sys.exit(1)

        if not self.longcat_dir.exists():
            logger.error(f"❌ LongCat-Video submodule not found at {self.longcat_dir}. Please clone it.")
            sys.exit(1)

    def generate_audio(self, text: str, output_path: Path, lang: str = 'tr') -> Path:
        """Generates TTS audio from text."""
        try:

            logger.info(f"🎤 Generating audio for: '{text}'")
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(output_path))
            logger.info(f"✅ Audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Failed to generate audio: {e}")
            raise

    def create_config(self, audio_path: Path, image_path: str, prompt: str) -> Path:
        """Creates the JSON configuration for LongCat-Video."""
        # Calculate relative path for audio to be used in config (which is used inside LongCat-Video)
        # We need the path relative to where the script is run, or absolute
        # using absolute path is safer
        
        config = {
            "prompt": prompt,
            "cond_image": image_path,
            "cond_audio": {
                "person1": str(audio_path.absolute())
            }
        }
        
        config_path = self.assets_dir / "turkish_avatar.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Config saved to {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
            raise

    def run_generation(self, config_path: Path, checkpoint_dir: Optional[str] = None):
        """Runs the LongCat video generation script."""
        script_path = self.longcat_dir / "run_demo_avatar_single_audio_to_video.py"
        ckpt_dir = checkpoint_dir if checkpoint_dir else str(self.weights_dir)
        
        # Construct command
        # Note: running with python directly, assuming env is set up
        cmd = [
            sys.executable,
            str(script_path),
            "--input_json", str(config_path),
            "--checkpoint_dir", ckpt_dir,
            "--stage_1", "ai2v", # Defaulting to ai2v as per demo
            "--num_segments", "1",
            "--resolution", "480p",
            "--output_dir", str(self.output_dir)
        ]

        logger.info(f"🚀 Starting video generation... This may take a while.")
        logger.info(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        # Ensure PYTHONPATH includes LongCat-Video so imports work
        env["PYTHONPATH"] = f"{str(self.longcat_dir)}:{env.get('PYTHONPATH', '')}"

        try:
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"✅ Video generation completed. Check {self.output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Video generation failed with code {e.returncode}")
            raise


def main():
    parser = argparse.ArgumentParser(description="LongCat Turkish Avatar Generator")
    parser.add_argument("--text", type=str, required=True, help="Text to speak (Turkish)")
    parser.add_argument("--image", type=str, default="assets/avatar/single/man.png", help="Path to reference image (relative to LongCat-Video)")
    parser.add_argument("--prompt", type=str, default="A professional person speaking Turkish.", help="Video content description")
    parser.add_argument("--checkpoint", type=str, help="Path to model weights")
    
    args = parser.parse_args()

    generator = AvatarGenerator()
    generator.check_dependencies()

    # 1. Generate Audio
    audio_path = generator.base_dir / "turkish_audio.mp3"
    generator.generate_audio(args.text, audio_path)

    # 2. Create Config
    # Check if image path is absolute or relative to LongCat
    # The default is relative to LongCat-Video root as per original script structure
    config_path = generator.create_config(audio_path, args.image, args.prompt)

    # 3. Run Generation
    try:
        generator.run_generation(config_path, args.checkpoint)
    except Exception as e:
        logger.error("Failed to run generation pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()
