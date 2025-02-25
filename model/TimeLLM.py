from utils.conf import Configuration
from utils.patching import Patching
from utils.patching import calculate_patch_num
from utils.mapping import Mapping

import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

class LLM4SSS(nn.Module):
    def __init__(self, conf: Configuration):
        super(LLM4SSS, self).__init__()
        
        self.conf = conf
        self.llm_path = conf.getEntry("llm_path")
        self.llm_type = conf.getEntry("llm_type")
        self.llm_layers_num = conf.getEntry("llm_layers")
        self.reduced_vocab_size = conf.getEntry("reduced_vocab_size")
        self.dim_llm = self.conf.getEntry("dim_llm")
        self.dim_ff = conf.getEntry("dim_ff")
        self.device = conf.getEntry("device")
        self.prompt_len = self.conf.getEntry("prompt_len")
        
        self.llm_pos = self.llm_path + self.llm_type + "/"
        self.patch_num = calculate_patch_num(conf)
        
        if self.llm_type == 'gpt2':
            self.llm = GPT2Model.from_pretrained(self.llm_pos)  # pyright: ignore
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.llm_pos)
            self.tokenizer.pad_token = '[PAD]'
        elif self.llm_type == 'bert':
            self.llm = BertModel.from_pretrained(self.llm_pos)  # pyright: ignore
            self.tokenizer = BertTokenizer.from_pretrained(self.llm_pos)
            self.tokenizer.pad_token = '[PAD]'
        elif self.llm_type == 'llama7b':
            self.llm = LlamaModel.from_pretrained(self.llm_pos) # pyright: ignore
            self.tokenizer = LlamaTokenizer.from_pretrained(self.llm_pos, local_files_only=True)
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

        for _, (name, param) in enumerate(self.llm_model.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        # self.word_embeddings: (vocab_size, dim_llm)
        self.vocab_size = self.word_embeddings.shape[0]
        self.patching = Patching(conf)
        self.reduce_layer = nn.Linear(self.vocab_size, self.reduced_vocab_size)
        self.mapping = Mapping(conf)
        self.recover1 = nn.Linear(self.dim_llm, self.dim_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.recover2 = nn.Linear(self.patch_num + self.prompt_len, self.patch_num)
        self.out_projection = FlattenHead(self.conf, self.patch_num)
        
        
    def forward(self, batch):
        # batch(batch_size, len_series)
        
        self.seq_overview = self.conf.getEntry("seq_overview")
        self.batch_size = self.conf.getEntry("batch_size")
        self.device = self.conf.getEntry("device")
        
        min_batch = torch.min(batch, dim=1)
        max_batch = torch.max(batch, dim=1)
        median_batch = torch.median(batch, dim=1).values
        trends = batch.diff(dim=1).sum(dim=1)
        # batch.diff(dim=1): (batch_size, len_series - 1)
        # trends: (batch_size)
        
        prompts = []
        for i in range(min_batch.values.shape[0]):
            min_batch_i = str(min_batch.values[i])
            max_batch_i = str(max_batch.values[i])
            median_batch_i = str(median_batch[i])
            prompt = (
                f"<|start_prompt|>Dataset sequence overview: {self.seq_overview}"
                "Input statistics: "
                f"min value {min_batch_i}, "
                f"max value {max_batch_i}, "
                f"median value {median_batch_i}, "
                f"the trend of input is {'upward' if trends[i] > 0 else 'downward'}, "
            )
            prompts.append(prompt)
            # prompts: (batch_size)
        
        prompts = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=2048).input_ids
        prompts_embedding = self.llm_model.get_input_embeddings()(prompts)
        # prompts_embedding: (batch_size, prompt_len, dim_llm)
        
        current_length = prompts_embedding.size(1)

        if current_length > self.prompt_len:
            prompts_embedding = prompts_embedding[:, :self.prompt_len, :]
        elif current_length < self.prompt_len:
            pad_embedding = self.llm_model.get_input_embeddings()(
                torch.tensor([self.tokenizer.pad_token_id] * (self.prompt_len - current_length))
                .unsqueeze(0)
                .repeat(prompts_embedding.size(0), 1)
            )
            prompts_embedding = torch.cat([prompts_embedding, pad_embedding], dim=1)
        
        batch_patched = self.patching(batch)
        # batch_patched: (batch_size, patch_num, dim_model)
        
        source_embeddings = self.reduce_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # source_embeddings: (reduced_vocab_size, dim_llm)
        
        batch_mapped = self.mapping(batch_patched, source_embeddings, source_embeddings)
        # batch_mapped: (batch_size, patch_num, dim_llm)
    
        llm_imput_embeddings = torch.cat([prompts_embedding, batch_mapped], dim=1)
        # llm_imput_embeddings: (batch_size, patch_num + prompt_len, dim_llm)
        
        with torch.no_grad():
            llm_output = self.llm_model(inputs_embeds = llm_imput_embeddings).last_hidden_state
            # llm_output: (batch_size, patch_num + prompt_len, dim_llm)
        
        llm_output = self.recover1(llm_output)
        # llm_output: (batch_size, patch_num + prompt_len, dim_ff)
        
        llm_output = self.relu(llm_output)
        llm_output = self.dropout(llm_output)
        
        llm_output = self.recover2(llm_output.permute(0, 2, 1)).permute(0, 2, 1)
        # llm_output: (batch_size, patch_num, dim_ff)
        
        series_reduce = self.out_projection(llm_output)
        # series_reduce: (batch_size, len_reduce)
        
        return series_reduce


class FlattenHead(nn.Module):
    def __init__(self, conf: Configuration, patch_num, head_dropout=0.1):
        super(FlattenHead, self).__init__()
        
        dim_ff = conf.getEntry("dim_ff")
        len_reduce = conf.getEntry("len_reduce")
        device = conf.getEntry("device")
        
        self.flatten = nn.Flatten(start_dim=-2).to(device)
        self.linear = nn.Linear(patch_num*dim_ff, len_reduce).to(device)
        self.dropout = nn.Dropout(head_dropout).to(device)


    def forward(self, x):
        # llm_output: (batch_size, patch_num, dim_ff)
        
        x = self.flatten(x)
        # llm_output: (batch_size, patch_num * dim_ff)
        
        x = self.linear(x)
        # llm_output: (batch_size, len_reduce)
        
        x = self.dropout(x)
        # llm_output: (batch_size, len_reduce)
        
        return x