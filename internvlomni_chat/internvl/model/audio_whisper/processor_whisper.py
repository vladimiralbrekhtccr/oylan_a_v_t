from transformers.processing_utils import ProcessorMixin
import torch
import librosa


class WhisperProcessor(ProcessorMixin):
    attributes = ["feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor"
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def get_T_after_cnn(self,L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def __call__(self, *args, **kwargs):
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", 16000)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
            mel_len = L // 160
            audio_len_after_cnn = self.get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
            inputs['audio_len_after_cnn'] = torch.tensor(audio_len_after_cnn, dtype=torch.long)
            inputs['audio_token_num'] = torch.tensor(audio_token_num, dtype=torch.long)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)
