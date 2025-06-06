import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
from dotenv import load_dotenv

load_dotenv()

def load_base_model_and_tokenizer():
    """
    Carrega o modelo base e o tokenizer do Hugging Face usando o token 
    armazenado de forma segura no arquivo .env.
    """
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    #model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    if not hf_token:
        raise ValueError("Token do Hugging Face não encontrado! Verifique se seu arquivo .env está correto e no lugar certo.")

    # quantization_config = BitsAndBytesConfig(
    #      load_in_4bit=True, 
    #      bnb_4bit_quant_type="nf4", 
    #      bnb_4bit_compute_dtype=torch.float16
    #  )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        #quantization_config=quantization_config, 
        device_map="auto", 
        token=hf_token,
        torch_dtype=torch.float16
    )
    return model, tokenizer

def create_langchain_llm(model, tokenizer):
    """
    Cria um objeto LLM compatível com o LangChain a partir de um modelo e tokenizer locais.
    Este objeto será usado nas 'chains' do LangChain.
    """
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

