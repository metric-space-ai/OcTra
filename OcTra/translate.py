import gradio as gr
import fasttext
from .configurations.get_constants import constantConfig
from .configurations.get_hyperparameters import hyperparameterConfig
from pyngrok import ngrok
from .funcs.models import load_models

ngrok.set_auth_token('2QjI3u9txqyB9chbDt1rUJF5th4_31BsnCZFuiZTUoeVMeyui')

# load the constants
constants       = constantConfig()
hyperparameters = hyperparameterConfig()

# load the models
model_dict = load_models(hyperparameters.device, constants.model_name_dict)

import time
import nltk

from transformers import pipeline

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
LID = fasttext.load_model("OcTra/lid218e.bin")

def detect_language(text,LID):
    predictions = LID.predict(text)
    detected_lang_code = predictions[0][0].replace("__label__", "")
    return detected_lang_code

def translation(model_name, 
                sentence_mode, selection_mode, 
                source, target, 
                text):
    start_time = time.time()

    # Determine the source language
    if selection_mode == "Auto-detect":
        detected_lang_code = detect_language(text,LID)
        flores_source_code = detected_lang_code
        source_code = flores_source_code
    else:
        if source == "Auto-detect":  # Make sure we don't use "Auto-detect" as a key
            return {'error': "Source language cannot be 'Auto-detect' when selection mode is manual."}
        source_code = constants.flores_codes.get(source)
        if not source_code:
            return {'error': f"Source language {source} not found in flores_codes."}


    target_code = constants.flores_codes[target]
    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline(
        'translation', model=model, 
        tokenizer=tokenizer, src_lang=source_code, tgt_lang=target_code, 
        device=hyperparameters.device)

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

def test_it(text, lang):

    return f'{text} is in {lang}'

def main():
    with gr.Blocks(title = "Octopus Translation App") as octopus_translator:
        with gr.Row():
            gr.Audio(source="microphone")
        
        with gr.Row():
            model_param = gr.Radio(['0.6B', '1.3B', '3.3B'], value='1.3B', label='NLLB Model size', interactive=True)
            input_type  = gr.Radio(['Whole text', 'Sentence-wise'],value='Sentence-wise', label="Translation Mode", interactive=True)
            select_type = gr.Radio(['Manually select','Auto-detect'],value='Auto-detect', label="Source Language Selection Mode", interactive=True)
        
        with gr.Row():
            source_language = gr.Dropdown(list(constants.flores_codes.keys()), value='English', label='Source (if manually selecting)', interactive=True)
            target_language = gr.Dropdown(list(constants.flores_codes.keys()), value='German', label='Target', interactive=True)

        with gr.Row():
            input_text  = gr.Textbox(lines=5, label="Input text")
            output_text = gr.JSON(label='Translated text')
        
        input_text.submit(
            translation, 
            inputs=[
                    model_param, 
                    input_type, select_type,
                    source_language, target_language,
                    input_text],
            outputs=[output_text]
            )

    octopus_translator.launch(favicon_path="OcTra/assets/metric-favicon.png", share=True)
