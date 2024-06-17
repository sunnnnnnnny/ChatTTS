
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ..utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat

def infer_code(
    models,
    text, 
    spk_emb = None,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.3, 
    repetition_penalty = 1.05,
    max_new_token = 2048,
    **kwargs
):
    # models : vocos dvae gpt spk_stat decoder tokenizer
    device = next(models['gpt'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
        
    if not isinstance(temperature, list):
        temperature = [temperature] * models['gpt'].num_vq
    
    if spk_emb is not None:
        text = [f'[Stts][spk_emb]{i}[Ptts]' for i in text] 
    else:  # ['[Stts][empty_spk][speed_5]很 多 人 觉 得 [uv_break] 哎 [uv_break] ， 想 把 英 语 学 的 好 [uv_break] ， 这 个 单 词 就 是 一 个 不 能 少 。 一 个 个 的 [uv_break] 是 死 背 单 词 [uv_break] 。 那 知 道 的 单 词 多 了 当 然 会 是 好 事 。 可 是 [uv_break] 除 了 考 试 以 外 ， 或 是 在 写 作 阅 读 以 外 [uv_break] ， 在 我 们 这 种 中 国 式 的 哑 巴 英 语 上 [uv_break] ， 我 们 缺 少 是 词 汇 量 么 ？[Ptts]']
        text = [f'[Stts][empty_spk]{i}[Ptts]' for i in text]
    # text_tokens: key[data] = input_ids, token_type_ids, attention_mask [1,108] [1,108] all_zero [1,108] all_one
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    input_ids = text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq)  # [1,108,4]
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)  # [1,108] all_true
    
    inputs = {
        'input_ids': input_ids,
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }

    emb = models['gpt'].get_emb(**inputs)  # [1,108,768]
    if spk_emb is not None:
        emb[inputs['input_ids'][..., 0] == models['tokenizer'].convert_tokens_to_ids('[spk_emb]')] = \
            F.normalize(spk_emb.to(device).to(emb.dtype)[None].expand(len(text), -1), p=2.0, dim=1, eps=1e-12)  
    
    num_code = models['gpt'].emb_code[0].num_embeddings - 1  # 626 - 1 = 625
    
    LogitsWarpers = []  # LogitsWarpers : TopPLogitsWarper, TopKLogitsWarper
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = [] # LogitsProcessors : CustomRepetitionPenaltyLogitsProcessorRepeat
    if repetition_penalty is not None and repetition_penalty != 1:  # repetition_penalty = 1.05
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(\
            repetition_penalty, num_code, 16))
    
    result = models['gpt'].generate(
        emb, inputs['input_ids'], 
        temperature = torch.tensor(temperature, device=device),  # temperature [0.3,0.3,0.3,0.3]
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = num_code,  # num_code = 625
        max_new_token = max_new_token,   # max_new_token = 2048
        infer_text = False,
        **kwargs
    )  # result: keys: ids->list, [832,4], attentions->list lens=833,hiddens->list [832,768]
    
    return result


def refine_text(
    models, 
    text,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.7, 
    repetition_penalty = 1.0,
    max_new_token = 384,
    prompt = '',
    **kwargs
):
    
    device = next(models['gpt'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
    
    assert len(text), 'text should not be empty'
    # ['[Sbreak]很多人觉得，想把英语学的好，单词一个不能少。一个个的是死背单词。知道的单词多了当然会是好事。可是除了考试以外，或是在写作阅读以外，在我们中国式的哑巴英语上，我们缺少是词汇量么？[Pbreak]']
    text = [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]  # add Sbreak and Pbreak
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device) # text_token[data]: input_ids [1,90] token_type_ids [1,90] attention_mask [1,90]
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device) # text_mask [1,90] all_true

    inputs = {
        'input_ids': text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq),
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, len(models['tokenizer']), 16))
    
    result = models['gpt'].generate(
        models['gpt'].get_emb(**inputs), inputs['input_ids'], 
        temperature = torch.tensor([temperature,], device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = torch.tensor(models['tokenizer'].convert_tokens_to_ids('[Ebreak]'), device=device)[None], 
        max_new_token = max_new_token, 
        infer_text = True,
        **kwargs
    )  # results: ids{list:1}  [100,], attentions{list:101}, hiddens{list, 0}
    return result