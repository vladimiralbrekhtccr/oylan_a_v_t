import warnings
from typing import Any, List, Optional, Tuple, Union
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import torch.distributed as dist
import torch.utils.checkpoint
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from internvl.conversation import get_conv_template
from .modeling_internvl_chat import InternVLChatModel
from .configuration_internvl_audio_chat import InternVLChatAudioConfig
from internvl.model.audio_whisper.modeling_whisper import AudioWhisperModel



def load_audio(audio_file, audio_processor):
    audio_values, _ = librosa.load(audio_file, sr=16000) # sample rate should be 16000
    
    audio_process_values = audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
    input_features = audio_process_values['input_features']
    audio_len_after_cnn = audio_process_values['audio_len_after_cnn']
    audio_token_num = audio_process_values['audio_token_num']
                

    audio_input = {'audio_values': input_features,
                   'audio_len_after_cnn': audio_len_after_cnn,
                   'audio_token_num': audio_token_num,
                   }
    return audio_input


class InternVLChatAudioModel(InternVLChatModel):

    def __init__(self, config: InternVLChatAudioConfig, vision_model=None, language_model=None, audio_model=None):
        super().__init__(config, vision_model, language_model)
        if audio_model is not None:
            self.audio_model = audio_model
        else:
            self.audio_model = AudioWhisperModel(config.audio_config)

        audio_hidden_size = config.audio_config.d_model
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp2 = nn.Sequential(
                nn.LayerNorm(audio_hidden_size),
                nn.Linear(audio_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )   # mlp2: audio feature mapping

        self.audio_context_token_id = None

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def extract_audio_feature(self, audio_values, audio_len_after_cnn):
        # print("EXTRACT AUDIO FEATURE")
        # print("--------------------------------")
        # Исходная форма audio_values: [B, num_audios, 1, channels, T]
        # Удаляем размерность, равную 1, по оси 2:
        audio_values = audio_values.squeeze(2)  # -> [B, num_audios, channels, T]
        # Объединяем первые два измерения: [B*num_audios, channels, T]
        B, num_audios, C, T = audio_values.shape
        audio_values = audio_values.reshape(B * num_audios, C, T)
        # Делаем audio_len_after_cnn одномерным: из [B, num_audios] в [B*num_audios]
        audio_len_after_cnn = audio_len_after_cnn.view(-1)
        
        max_len_in_batch = int(torch.max(audio_len_after_cnn).item())
        # Создаем padding_mask размера [B*num_audios, max_len_in_batch]
        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch],
                                dtype=audio_values.dtype,
                                device=audio_values.device)
        # Для каждого аудио зануляем первые audio_len_after_cnn[index] позиций
        for index in range(audio_values.size(0)):
            padding_mask[index, :int(audio_len_after_cnn[index].item())] = 0
        # print("goes to audio model")
        # print("padding_mask", padding_mask.shape)
        # print("audio_values", audio_values.shape)
        # print("audio_len_after_cnn", audio_len_after_cnn)
        
        # Передаем обработанные признаки в аудио модель
        last_hidden_state = self.audio_model(
            input_features=audio_values, 
            attention_mask=padding_mask, 
            audio_len_after_cnn=audio_len_after_cnn
        )
        # print("ENCODER OUTPUT")
        # print("--------------------------------")
        # print("last_hidden_state", last_hidden_state.shape)
        audio_embeds = self.mlp2(last_hidden_state)
        # print("audio_embeds", audio_embeds.shape)
        # print("--------------------------------")
        return audio_embeds



    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            audio_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            audio_flags: Optional[torch.LongTensor] = None,
            audio_len_after_cnn: Optional[torch.LongTensor] = None,
            audio_token_num: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("forward")
        # Приведение аудио флагов и числа токенов к одномерному виду
        if audio_flags is not None:
            audio_flags = audio_flags.view(-1)
        if audio_token_num is not None:
            audio_token_num = audio_token_num.flatten()
        if audio_len_after_cnn is not None:
            audio_len_after_cnn = audio_len_after_cnn.flatten()


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        # print("input_embeds shape after reshape:", input_embeds.shape)
        # print("--------------------------------")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        img_selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[img_selected] = input_embeds[img_selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[img_selected].shape={input_embeds[img_selected].shape}, vit_embeds.shape={vit_embeds.shape}')
            n_token = img_selected.sum()
            input_embeds[img_selected] = input_embeds[img_selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            audio_batch_size = audio_values.shape[0]
            print(f'audio batch size: {audio_batch_size}, audios per sample: {audio_batch_size / B}')

        # print("Extracting audio features...")
        audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)  # Ожидается форма [num_audios, n_frame, C]
        # print("Finished extracting audio features.")
        # print("audio_embeds shape:", audio_embeds.shape)
        # print("--------------------------------")

        # Здесь происходит обработка аудио токенов
        output_audios = []
        total_tokens = 0
        # print("Processing individual audio tokens for LLM injection:")
        for i in range(len(audio_token_num)):
            token_num = int(audio_token_num[i].item())
            # print(f"Audio {i+1}: audio_len_after_cnn = {audio_len_after_cnn[i].item()}, token_num = {token_num}")
            # Здесь мы извлекаем эмбеддинги для i-го аудио, используя token_num, то есть отдельно для каждого аудио.
            audio_slice = audio_embeds[i][:token_num]
            # print(f"Audio {i+1} slice shape: {audio_slice.shape}")
            output_audios.append(audio_slice)
            total_tokens += token_num
        # print("Total audio tokens processed:", total_tokens)

        if len(output_audios):
            output_audios = torch.cat(output_audios, dim=0)
            audio_selected = (input_ids == self.audio_context_token_id)
            num_context_tokens = int(audio_selected.sum().item())
            #print("Number of audio context tokens in prompt:", num_context_tokens)
            #print("Shape of concatenated audio embeddings:", output_audios.reshape(-1, C).shape)
            if total_tokens != num_context_tokens:
                print(f"Warning: mismatch: prompt expects {num_context_tokens} tokens, but processed {total_tokens} tokens. Adjusting...")
                if total_tokens > num_context_tokens:
                    output_audios = output_audios[:num_context_tokens]
                    total_tokens = num_context_tokens
                else:
                    pad_size = num_context_tokens - total_tokens
                    pad_tensor = torch.zeros(pad_size, C, device=output_audios.device, dtype=output_audios.dtype)
                    output_audios = torch.cat([output_audios, pad_tensor], dim=0)
                    total_tokens = num_context_tokens
            input_embeds[audio_selected] = input_embeds[audio_selected] * 0.0 + output_audios.reshape(-1, C)

        #print("Final input_embeds shape before LLM:", input_embeds.shape)
        #print("--------------------------------")

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        # Логика вычисления loss остается без изменений...
        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)
            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



    def Audio_chat(self, tokenizer, pixel_values, audio,  question, generation_config, history=None, return_history=False,num_patches_list=None, 
                   IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',AUDIO_START_TOKEN='<audio>',AUDIO_END_TOKEN='</audio>',
                   AUDIO_CONTEXT_TOKEN='<AUDIO_CONTEXT>',verbose=None):

        if history is None and audio is not None:
            if question is None:
                question = '<audio>\n'
            else:
                question = '<audio>\n' + question
                
        if history is None and pixel_values is not None:
            if question is None:
                question = '<image>\n'
            else:
                question = '<image>\n' + question
            
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.audio_context_token_id = audio_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        if audio is not None:
            audio_tokens = AUDIO_START_TOKEN + AUDIO_CONTEXT_TOKEN * audio['audio_token_num'] + AUDIO_END_TOKEN
            query = query.replace('<audio>', audio_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        audio['audio_len_after_cnn'] = torch.tensor([audio['audio_len_after_cnn']])
        audio['audio_token_num'] = torch.tensor([audio['audio_token_num']])
        generation_output = self.generate(
            pixel_values=pixel_values,
            audio_values=audio['audio_values'].to(self.device, dtype=self.dtype),
            audio_len_after_cnn=audio['audio_len_after_cnn'],
            audio_token_num=audio['audio_token_num'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query.replace(AUDIO_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}', '<audio>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            visual_features: Optional[torch.FloatTensor] = None,
            audio_values: Optional[torch.FloatTensor] = None, # audio features [1, 128, 3000]
            audio_len_after_cnn: Optional[bool] = None,
            audio_token_num: Optional[bool] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        print("--------------------------------")
        print("generate")
        # assert self.img_context_token_id is not None
        # assert self.audio_context_token_id is not None

        vit_embeds = None
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)

        if vit_embeds is not None:
            selected = (input_ids == self.img_context_token_id)
            input_embeds[selected] = vit_embeds.reshape(-1, C)

        if audio_values is not None and audio_len_after_cnn is not None and audio_token_num is not None:
            audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)
            print("Full audio embeds shape:", audio_embeds.shape)
            output_audios = []
            total_tokens = 0

            print("\nProcessing individual audios:")
            for i in range(len(audio_token_num)):
                token_num = int(audio_token_num[i].item())
                audio = audio_embeds[i][:token_num]
                print(f"Audio {i+1}: audio_len_after_cnn={audio_len_after_cnn[i].item()}, token_num={token_num}, audio slice shape={audio.shape}")
                output_audios.append(audio)
                total_tokens += token_num

            output_audios = torch.cat(output_audios, dim=0)
            print("\nTotal audio tokens to be injected into LLM:", total_tokens)

            selected = (input_ids == self.audio_context_token_id)
            print("Number of audio context tokens in prompt:", int(selected.sum().item()))
            assert int(selected.sum().item()) == total_tokens, \
                f"Mismatch: prompt expects {selected.sum().item()} audio tokens, but processed {total_tokens} tokens"
            
            input_embeds[selected] = output_audios.reshape(-1, C)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs