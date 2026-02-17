import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock dependencies before importing run_avatar
sys.modules["gtts"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["librosa"] = MagicMock()

# Add project root to path to import run_avatar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_avatar import AvatarGenerator

class TestAvatarGenerator(unittest.TestCase):

    def setUp(self):

        os.makedirs("/tmp/test_avatar_task", exist_ok=True)
        self.generator = AvatarGenerator(base_dir="/tmp/test_avatar_task")
        # Mocking directories to avoid actual FS operations errors in tests
        # self.generator.longcat_dir = MagicMock() # Removed to avoid str() issues


    @patch('run_avatar.gTTS')
    def test_generate_audio(self, mock_gtts_cls):
        # Setup mock
        mock_tts_instance = MagicMock()
        mock_gtts_cls.return_value = mock_tts_instance
        
        # Run
        text = "Merhaba dünya"
        output_path = Path("/tmp/test_audio.mp3")
        self.generator.generate_audio(text, output_path)
        
        # Verify
        mock_gtts_cls.assert_called_with(text=text, lang='tr', slow=False)
        mock_tts_instance.save.assert_called_with(str(output_path))

    def test_create_config(self):
        # Run
        audio_path = Path("/tmp/audio.mp3")
        image_path = "assets/image.png"
        prompt = "Test prompt"
        
        # We need to mock open to avoid writing to disk, or just let it write to tmp
        # Let's use a temporary directory for the generator in setUp normally, 
        # but here we can just mock the file writing verify the json content
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                self.generator.create_config(audio_path, image_path, prompt)
                
                args, _ = mock_json_dump.call_args
                config_data = args[0]
                
                self.assertEqual(config_data['prompt'], prompt)
                self.assertEqual(config_data['cond_image'], image_path)
                self.assertEqual(config_data['cond_audio']['person1'], str(audio_path.absolute()))

    @patch('subprocess.run')
    def test_run_generation(self, mock_subprocess):
        config_path = Path("/tmp/config.json")
        checkpoint = "/tmp/ckpt"
        
        self.generator.run_generation(config_path, checkpoint)
        
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        
        self.assertIn("run_demo_avatar_single_audio_to_video.py", cmd[1])
        self.assertIn("--input_json", cmd)
        self.assertIn(str(config_path), cmd)
        self.assertIn(checkpoint, cmd)
        self.assertIn("ai2v", cmd)

if __name__ == '__main__':
    unittest.main()
