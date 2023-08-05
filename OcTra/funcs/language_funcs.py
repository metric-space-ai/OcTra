import time
import nltk

from transformers import pipeline

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def detect_language(text,LID):
    predictions = LID.predict(text)
    detected_lang_code = predictions[0][0].replace("__label__", "")
    return detected_lang_code

def translation(model_name, 
                sentence_mode, selection_mode, 
                source, target, 
                text, 
                flores_codes, 
                model_dict, device):
    start_time = time.time()

    # Determine the source language
    if selection_mode == "Auto-detect":
        detected_lang_code = detect_language(text)
        flores_source_code = detected_lang_code
        source_code = flores_source_code
    else:
        if source == "Auto-detect":  # Make sure we don't use "Auto-detect" as a key
            return {'error': "Source language cannot be 'Auto-detect' when selection mode is manual."}
        source_code = flores_codes.get(source)
        if not source_code:
            return {'error': f"Source language {source} not found in flores_codes."}


    target_code = flores_codes[target]
    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_code, tgt_lang=target_code, device=device)

    if sentence_mode == "Sentence-wise":
        sentences = sent_tokenize(text)
        translated_sentences = []
        for sentence in sentences:
            translated_sentence = translator(sentence, max_length=400)[0]['translation_text']
            translated_sentences.append(translated_sentence)
        output = ' '.join(translated_sentences)
    else:
        output = translator(text, max_length=400)[0]['translation_text']

    end_time = time.time()

    result = {
        'inference_time': end_time - start_time,
        'source_language': source_code,
        'target_language': target_code,
        'result': output
    }
    return result