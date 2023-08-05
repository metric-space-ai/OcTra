import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_models(device, model_name_dict):
    model_dict = {}
    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer

    return model_dict