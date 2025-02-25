import re
import base64
import json
import requests
from os import path as osp
import collections
from termcolor import colored
from run_llava import llava_proxy
import random
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from PIL import Image
from openai import OpenAI
import time
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from prompts import scoring_prompt
import os
import openai

# global model
model = None

# nltk.download('punkt')
# nltk.download('punkt_tab')

API_KEY = ""
API_URL = "https://api.openai.com"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

client = OpenAI(api_key="", 
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )

# model_path = 'llava-v1.6-34b'
model_path = 'llava-v1.6-vicuna-7b'


global qwen_client
qwen_client = OpenAI(api_key="", 
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )


def load_model(model_name):
    global model
    global client
    if model_name == "llava":
        model_path = '/home/czx/models/llava-v1.6-vicuna-7b'
        model = llava_proxy(model_path, None)
        gen_func = llava_generate
    elif model_name == "llava_34b":
        model_path = 'liuhaotian/llava-v1.6-34b'
        model = llava_proxy(model_path, None)
        gen_func = llava_generate
    elif model_name == "qwen":
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:1", trust_remote_code=True).eval()
        gen_func = qwen_generate
    elif model_name == "mini-cpm":
        global tokenize
        torch.manual_seed(0)
        # model_name = "/home/czx/models/MiniCPM-V-2_6"
        model_name = 'openbmb/MiniCPM-V-2_6'
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
        gen_func = cpm_generate
    elif model_name == "qwen-max":
        gen_func = qwen_api_generate
    elif model_name == "qwen2-7b":
        gen_func = qwen2_api_generate
    elif model_name == "qwq":
        gen_func = qwq_api_generate
    elif model_name == "doubao-lite":
        gen_func = doubao_lite_generate
    elif model_name == "doubao-pro":
        gen_func = doubao_pro_generate
    elif model_name == "gemini":
        gen_func = gemini_generate
    elif model_name == "gpt4o":
        gen_func = gpt4o_generate
    elif model_name.startswith("step"):
        global step_client
        step_client = OpenAI(api_key="",
                            base_url="https://api.stepfun.com/v1")
        if model_name == "step-1o":
            model="step-1o-vision-32k"
        elif model_name == "step":
            model="step-1v-8k"
        gen_func = step_generate
    return gen_func


def gpt4o_generate(text, image_paths=[], temp=0, presence_penalty=None):
    # print(text)
    num = 5
    res = ""
    refuse_num = 0
    messages = [{"role": "user", "content": text}]
    while num > 0 and (not res):
        try:
            content = [{"type": "text", "text": text}]
            # content = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
            
            # data = {"model": "gpt-4o-mini-2024-07-18",
            #         "messages": [{"role": "user", "content": content}],
            #         "temperature": temp
            #         }
            # data = json.dumps(data)
            # response = requests.post(API_URL, headers=HEADERS, data=data, timeout=180)
            # response_json = response.json()
            # res = response_json['choices'][0]['message']['content']
            
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[{"role": "user", "content": content}],
                temperature=temp,
            )
            res = response.choices[0].message.content

            if res.startswith("I'm sorry") or res.startswith("I'm unable to") or "I can't assist" in res:
                refuse_num += 1
                if refuse_num >= 5:
                    return "Refusal"
                raise Exception("Refusal")
        except Exception as e:
            time.sleep(1)
            traceback.print_exc()
            num -= 1
    
    return res

def get_gpt4o_score(question, meme_image_paths, meme_text, answer, ref_ans):
    judge_prompt = scoring_prompt.replace('{meme_text}', meme_text).replace(
        '{question}', question).replace('{ref_answer}', ref_ans).replace('{answer}', answer)
    
    for _ in range(5):
        score_res = gpt4o_generate(judge_prompt, meme_image_paths, temp=0)
        if len(re.findall(r'\[\[.*?\]\]', score_res.strip())) == 0:
            match = re.search(r"Rating:\s*(\d+)", score_res.strip().replace('[', '').replace(']', ''))
            if match:
                score = float(match.group(1))
                break
            else:
                print("score invalid")
                # print(score_res)
                score = None
        else:
            score = float(re.findall(
                r'\[\[.*?\]\]', score_res.strip())[-1].replace('[[', '').replace(']]', ''))
            break
    return score_res, score

base_url = "https://s..."
api_version = "2024-03-01-preview"
ak = "..."
model_name = "gpt-4o-2024-05-13"
client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=ak,
)

def gpt4o_generate_imglist(text, image_paths=[], temp=1, presence_penalty=None):
    # print(text)
    num = 5
    res = ""
    refuse_num = 0
    messages = [{"role": "user", "content": text}]
    while num > 0 and (not res):
        try:
            # content = [{"type": "text", "text": text}]
            content = []
            for image_path in image_paths:
                if image_path.startswith("http"):
                    content.append({"type": "image_url", "image_url": {"url": image_path}})
                else:  
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
            content.append({"type": "text", "text": text})
            
            # data = {"model": "gpt-4o-2024-05-13",
            #         "messages": [{"role": "user", "content": content}],
            #         "temperature": temp}
            # data = json.dumps(data)
            # response = requests.post(API_URL, headers=HEADERS, data=data)
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[{"role": "user", "content": content}],
                temperature=temp,
            )

            res = response.choices[0].message.content
            if res.startswith("I'm sorry") or res.startswith("I'm unable to") or "I can't assist" in res:
                refuse_num += 1
                if refuse_num >= 5:
                    return "Refusal"
                raise Exception("Refusal")
        except Exception as e:
            traceback.print_exc()
            time.sleep(1)
            num -= 1
    
    return res


def claude_3p5_generate(text, image_paths=[], temp=None, presence_penalty=None):
    print(text)
    num = 50
    res = ""
    messages = [{"role": "user", "content": text}]
    while num > 0 and len(res)==0:
        try:
            content = [{"type": "text", "text": text}]
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
            
            data = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": content}]}
            data = json.dumps(data)
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            traceback.print_exc()
            num -= 1
    
    return res

def llava_generate(text, image_paths):
    sep = ","
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": None,
        "query": text,
        "conv_mode": None,
        "image_file": sep.join(image_paths),
        "sep": sep,
        "temperature": 0,
        "top_p": 0.9,
        "num_beams": 1,
        "do_sample": True,
        "max_new_tokens": 1024
    })()

    return model.generate(args)

def qwen_api_generate(text, image_path):

    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = image_to_base64(image_path[0])
    cnt = 5  
    completion = None
    while cnt:
        try:
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-plus",
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            break
        except Exception as e:
            print(e)
            cnt -=1
    if not completion: return None
    res = completion.choices[0].message.content
    return res

def qwen2_api_generate(text, image_path):

    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = image_to_base64(image_path[0])
    cnt = 5  
    completion = None
    while cnt:
        try:
            completion = qwen_client.chat.completions.create(
                model="qwen2.5-vl-7b-instruct",
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            break
        except Exception as e:
            print(e)
            cnt -=1
    if not completion: return None
    res = completion.choices[0].message.content
    return res


def qwq_api_generate(text, image_path):

    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = image_to_base64(image_path[0])
    cnt = 5
    completion = None
    while cnt:
        try:
            completion = qwen_client.chat.completions.create(
                model="qwq-32b-preview",
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            break
        except Exception as e:
            print(e)
            cnt -=1
    if not completion: return None
    res = completion.choices[0].message.content
    return res

def qwen_generate(text, image_paths):
    content = []
    for image_path in image_paths:
        content.append({'image': image_path})
    content.append({'text': text})
    query = tokenizer.from_list_format(content)
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response

def cpm_generate(text, image_paths):
    image = Image.open(image_paths[0]).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, text]}]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    return answer.replace('<|endoftext|>','')

def doubao_lite_generate(text, image_paths=[], temp=None, presence_penalty=None):
    num = 5
    res = ""
    text = "我需要你务必使用英文回答所有接下来的请求。\n"+text
    while num > 0 and len(res)==0:
        try:
            content = [{"type": "text", "text": text}]
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

            data = {"model": "", "messages": [{"role": "user", "content": content}],"temperature": 0}
            data = json.dumps(data)
            response = requests.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                                        headers={
                                                "Content-Type": "application/json",
                                                "Authorization": "Bearer ",
                                                "temperature": '0'
                                            },data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            traceback.print_exc()
            num -= 1
    
    return res

def doubao_pro_generate(text, image_paths=[], temp=None, presence_penalty=None):
    num = 5
    res = ""
    text = "我需要你务必使用英文回答所有接下来的请求。\n"+text
    while num > 0 and len(res)==0:
        try:
            content = [{"type": "text", "text": text}]
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

            data = {"model": "", "messages": [{"role": "user", "content": content}],"temperature": 0}
            data = json.dumps(data)
            response = requests.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                                        headers={
                                                "Content-Type": "application/json",
                                                "Authorization": "Bearer ",
                                                "temperature": '0'
                                            },data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            traceback.print_exc()
            num -= 1
    
    return res
def gemini_generate(text, image_paths=[], temp=None, presence_penalty=None):
    num = 5
    res = ""
    while num > 0 and len(res)==0:
        try:
            content = [{"type": "text", "text": text}]
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

            data = {"model": "gemini-pro", "messages": [{"role": "user", "content": content}],"temperature": 0}
            data = json.dumps(data)
            response = requests.post("https://api.oneabc.org/v1/chat/completions",
                                        headers={
                                                "Content-Type": "application/json",
                                                "Authorization": "Bearer ",
                                                "temperature": '0'
                                            },data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
        except Exception as e:
            print(response_json)
            traceback.print_exc()
            num -= 1
    
    return res

def step_generate(text, image_path):

    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = image_to_base64(image_path[0])
    cnt = 5  
    completion = None
    while cnt:
        try:
            completion = step_client.chat.completions.create(
                        model=model,
                        temperature=0,
                        messages=[
                            {
                                "role": "system",
                                "content": "你是由阶跃星辰提供的AI聊天助手，你除了擅长中文，英文，以及多种其他语言的对话以外，还能够根据用户提供的图片，对内容进行精准的内容文本描述与分析。",
                            },
                            {
                                "role": "system",
                                "content": "接下来的所有问题，请你使用英语进行回答。",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": text,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        )
            break
        except Exception as e:
            print(e)
            cnt -=1
    if not completion: return None
    res = completion.choices[0].message.content
    return res
      

def bm25_retrieve(query, corpus, top_n=3):
    tokenized_corpus = [word_tokenize(doc['misbelief'].lower()) for doc in corpus]
    tokenized_query = word_tokenize(query.lower())

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return top_indices, [corpus[i] for i in top_indices]
