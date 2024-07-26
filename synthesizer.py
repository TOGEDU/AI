import torch
import torchaudio
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class XTTSynthesizer:
  def __init__(self, config_path, tokenizer_path, checkpoint_path, speaker_reference_path):
    self.config_path = config_path
    self.tokenizer_path = tokenizer_path
    self.checkpoint_path = checkpoint_path
    self.speaker_reference_path = speaker_reference_path

    # CUDA 사용 가능 여부 확인
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    self.model = self.load_model()

  def load_model(self):
    print("Loading model...")
    config = XttsConfig()

    print(f"Loading config from: {self.config_path}")
    config.load_json(self.config_path)
    print("Config loaded successfully")

    model = Xtts.init_from_config(config)

    print(f"Loading checkpoint from: {self.checkpoint_path}")
    print(f"Loading tokenizer from: {self.tokenizer_path}")
    try:      
      checkpoint_dir = str(Path(self.checkpoint_path).parent) + "/"
      model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, checkpoint_path=self.checkpoint_path, vocab_path=self.tokenizer_path, use_deepspeed=False) 
      print("Model checkpoint loaded successfully")
    except Exception as e:
      print(f"Error loading checkpoint or tokenizer: {e}")
      raise
    
    # model.cuda() gpu가 없으면 작동을 하지 못 함.
    model.to(self.device)
    return model

  def synthesize(self, text):
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.speaker_reference_path])

    print("Inference...")
    out = self.model.inference(
      text,
      "ko",
      gpt_cond_latent,
      speaker_embedding,
      temperature=0.7,  # 필요시 사용자 정의 파라미터 추가
    )

    return torch.tensor(out["wav"]).unsqueeze(0)

  def save_wav(self, wav_tensor, output_wav_path):
    torchaudio.save(output_wav_path, wav_tensor, 24000)