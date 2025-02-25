# 修改了prompt，不再输入文字
import json
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse
import traceback
from utils import *
from tqdm import tqdm
import concurrent.futures
from collections import Counter
import prompts
from collections import defaultdict

global taxonomy, cnt, new_cat_count

NUM_PARALLEL = 8


def get_gpt4o_score(question, meme_image_paths, meme_text, answer, ref_ans):
    prompt = {"name": "single-v1-ref", "type": "single", "system_prompt": "You are a helpful assistant.", "prompt_template":
              '''[Instruction]\nGiven the meme, with the Text: "{meme_text}" embedded in the image, please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the correctness(high priority), helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You will be given a high-quality reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer and identify the mistakes in the assistant's answer, then provide a short explanation. Be as objective as possible. Please be very careful in giving a 10. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]''', "description": "Prompt for general questions", "category": "religion", "output_format": "[[rating]]"}
    judge_prompt = prompt["prompt_template"].replace('{meme_text}', meme_text).replace(
        '{question}', question).replace('{ref_answer}', ref_ans).replace('{answer}', answer)
    score_res = gpt4o_generate(judge_prompt, meme_image_paths, temp=0)
    return score_res

def cate_classify(split, datalist):
    # datalist: [{"text":..., "image":...}, ...]
    res = []

    with tqdm(total=len(datalist), desc="Progress", unit="step") as pbar:
        for data in tqdm(datalist):
            pbar.set_description(f"Step {split}")
            
            prompt = prompts.prompt_cls1
            prompt = prompt.replace(r"{taxonomy}", json.dumps(taxonomy["meme_harmfulness"],indent=1)).\
                            replace(r"{text_content}",data['text'])
            try:
                num_voters = 3
                vote_res = []
                tasks = []
                new_task = dict()
                
                def ask_voters():
                    try:
                        response = gpt4o_generate(prompt, [data["image"]],temp = 1)
                        if '```json' in response:
                            response = re.search(r'```json\n(.+?)```',
                                                response, re.DOTALL).group(1)
                        response1 = json.loads(response)
                        return response1
                    except Exception as e:
                        print("ask_voters: ",response, e)

                for _ in range(num_voters*2):
                    if len(vote_res) >= num_voters: break # make sure at least 3 votes
                    ans = ask_voters()
                    if not ans: continue
                    vote_res.append(ans)
                
                for v in vote_res:
                    for t in v["task"]:
                        if t in taxonomy["meme_harmfulness"].keys():
                            tasks.append(t)
                    if v.get("new category", False):
                        for t in v["new category"].keys():
                            if t not in new_task.keys():
                                new_task[t] = v["new category"][t]
                
                # for current cats in taxonomy
                count = Counter(tasks)
                tasks = [item for item in count.keys() if count[item] >= 2]
                
                if len(tasks)+len(new_task) < 1:
                    continue
                    # print(data)
                
                # judge
                for task in new_task.keys():
                    # filter
                    match = re.search(r"[^a-zA-Z0-9]", task)
                    c =  match.group(0) if match else '.'
                        
                    task_l = task.split(c)
                    if len(task_l)>2:
                        continue
                    elif len(task_l) == 2:
                        if task_l[0] in taxonomy["meme_harmfulness"].keys():
                            continue
                    
                    prompt_judge = prompts.prompt_judge_new_cat
                    prompt_judge = prompt_judge.replace(r"{taxonomy}", json.dumps(taxonomy["meme_harmfulness"],indent=1)).\
                                                replace(r"{category}", json.dumps({task: new_task[task]}))
                    ans = gpt4o_generate(prompt_judge, [])
                    if ans.startswith("Yes"):
                        # examine
                        prompt = prompts.cls_judge  
                        prompt = prompt.replace(r"{category}", json.dumps({task: new_task[task]}))
                        ans = gpt4o_generate(prompt, [data["image"]])
                        if ans.startswith("Yes"):
                            taxonomy["meme_harmfulness"][task] = new_task[task]
                            taxonomy["meme_harmfulness_understanding"].append(task)
                            tasks.append(t)
                            new_cat_count[task]+=1
                            print("new cat", task)
                        
                for task in tasks:
                    temp = {"text": data["text"],
                            "image": data["image"].replace('/ori/','/erased/'),
                            "label": data["label"],
                            "task": task}
                    res.append(temp)
                
                # filter long-tail cats
                if cnt%100 == 0:
                    for cat in new_cat_count.keys():
                        try:
                            if new_cat_count[cat] < 5:
                                del new_cat_count[cat]
                                del taxonomy["meme_harmfulness"][cat]
                                taxonomy["meme_harmfulness_understanding"].remove(cat)
                        except Exception:
                            continue
                            
            except Exception as e:
                traceback.print_exc()
    return res

if __name__ == '__main__':
    
    data_path = "../data/data5k/full_data.json"
    cat_path = '../data/base_cat.json'
    
    save_path = data_path.replace("full_data.json","") + "cls_res_full.json"
    while os.path.exists(save_path):
        save_path = save_path.replace(".json","_1.json",)
    
    with open(data_path, 'r') as f:
        cls_res = json.load(f)
    with open(cat_path, 'r') as f:
        categories = json.load(f)
    taxonomy = categories
    
    # parallel with NUM_PARALLEL
    datalist = []
    split_num = len(cls_res) // NUM_PARALLEL
    for i in range(0, len(cls_res), split_num):
        datalist.append(cls_res[i:i + split_num])

    print("Start classification with data len:", len(cls_res))
    
    res = []
    cnt = 1
    new_cat_count = defaultdict(int)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        answers = [executor.submit(cate_classify,f"{i}/{split_num}",d) for i,d in enumerate(datalist)]
        for ans in concurrent.futures.as_completed(answers):
            res.extend(ans.result())
    cls_res = res

    with open(save_path, 'w') as f:
        json.dump(cls_res, f, ensure_ascii=False, indent=1)
    
    with open(save_path.replace('.json', '_cat.json'), 'w') as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=1)
