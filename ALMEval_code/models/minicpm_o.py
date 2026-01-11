import librosa
import torch
from .base import BaseModel


class MiniCPMO(BaseModel):
    NAME = 'minicpm-o'

    def __init__(
        self,
        model_path='OpenBMB/MiniCPM-o-2_6',
        **kwargs,
    ):
        assert model_path is not None
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import MiniCPM-o dependencies.\n"
                "Please make sure you have installed the correct transformers version"
            ) from e
        self.sample_rate = 16000
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

    def generate_inner(self, msgs):
        prompts = msgs.get('prompts') or []
        content = []
        for item in prompts:
            if item['type'] == 'text':
                content.append(item['value'])
            elif item['type'] == 'audio':
                audio_input, _ = librosa.load(item['value'], sr=self.sample_rate, mono=True)
                content.append(audio_input)
        messages = [{"role": "user", "content": content}]
        output = self.model.chat(msgs=messages, tokenizer=self.tokenizer)
        print('【minicpm-o output】:', output)
        return output
