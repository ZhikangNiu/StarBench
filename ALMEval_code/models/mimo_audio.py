import os
import sys
import torch
from .base import BaseModel


class MimoAudioModel(BaseModel):
    NAME = 'mimo-audio'

    def __init__(
        self,
        model_folder=None,
        model_path='models/MiMo-Audio-7B-Instruct',
        tokenizer_path='models/MiMo-Audio-Tokenizer',
        thinking=False,
        **kwargs,
    ):
        assert model_folder is not None, "model_folder must be set to the MiMo-Audio source directory."
        abs_model_folder = os.path.abspath(model_folder)
        sys.path.append(abs_model_folder)
        try:
            from src.mimo_audio.mimo_audio import MimoAudio
            from src.mimo_audio.process_speechdata import InputSegment
            from src.mimo_audio.modeling_mimo_audio import MiMoStopper
        except ImportError as e:
            raise ImportError(
                "Failed to import MiMo-Audio dependencies.\n"
                "Please ensure the MiMo-Audio repo is available and its requirements are installed."
            ) from e

        self.InputSegment = InputSegment
        self.MiMoStopper = MiMoStopper
        resolved_model_path = self.resolve_path(model_path, abs_model_folder)
        resolved_tokenizer_path = self.resolve_path(tokenizer_path, abs_model_folder)
        self.model = MimoAudio(resolved_model_path, resolved_tokenizer_path)
        self.thinking = thinking
        torch.cuda.empty_cache()

    @staticmethod
    def resolve_path(path, base_dir):
        """Resolves a model asset path relative to a base directory when possible.

        Args:
            path: Absolute or relative path to resolve.
            base_dir: Base directory used to resolve relative paths.

        Returns:
            The resolved path if a base-relative candidate exists; otherwise the input path.
        """
        if path is None:
            return None
        if os.path.isabs(path) or not base_dir:
            return path
        candidate = os.path.join(base_dir, path)
        return candidate if os.path.exists(candidate) else path

    def prepare_prompts_with_concating_audios(self, msg_or_prompts, thinking=False):
        prompts = msg_or_prompts
        if isinstance(msg_or_prompts, dict):
            prompts = msg_or_prompts.get('prompts') or []
        segments = [
            self.InputSegment(
                text="<|im_start|>user\n",
                speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                text_zeroemb_idx=self.model.empty_token,
            )
        ]
        for idx, item in enumerate(prompts):
            if item['type'] == 'text':
                text = item['value']
                if (
                    idx + 1 < len(prompts)
                    and prompts[idx + 1]['type'] == 'audio'
                    and text
                    and not text.endswith((' ', '\n', '\t'))
                ):
                    text += ' '
                segments.append(
                    self.InputSegment(
                        text=text,
                        speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.empty_token,
                    )
                )
            elif item['type'] == 'audio':
                audio_tokenized = self.model.preprocess_input(item['value'])
                segments.append(
                    self.InputSegment(
                        audio=audio_tokenized,
                        speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                        text_zeroemb_idx=self.model.empty_token,
                    )
                )

        segments.append(
            self.InputSegment(
                text="<|im_end|>\n",
                speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                text_zeroemb_idx=self.model.empty_token,
            )
        )
        segments.append(
            self.InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                text_zeroemb_idx=self.model.empty_token,
            )
        )
        if not thinking:
            segments.append(
                self.InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.empty_token,
                )
            )
        else:
            segments.append(
                self.InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.model.speech_zeroemb_idx,
                    text_zeroemb_idx=self.model.empty_token,
                )
            )
        return self.model.get_input_ids(segments)

    def generate_inner(self, msgs):
        prompts = msgs.get('prompts') or []
        if not prompts:
            return ""
        stopping_criteria = [
            self.MiMoStopper(
                stop_tokens=[self.model.tokenizer.eos_token_id, self.model.im_end_idx],
                group_size=self.model.group_size,
                audio_channels=self.model.audio_channels,
            )
        ]
        input_ids = self.prepare_prompts_with_concating_audios(prompts, thinking=self.thinking)
        output = self.model.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="audio_understanding",
        )
        print('【mimo-audio output】:', output)
        return output
