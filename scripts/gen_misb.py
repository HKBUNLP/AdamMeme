from utils import*
import json
from tqdm import tqdm
import concurrent.futures
from prompts import ref_ans_judge as ref_ans_judge_template 
import traceback

cat_prompt = \
{
    "religion": {
        "prompt": "Analyze how this meme may harm individuals or communities based on their religious beliefs, considering potential promotion of intolerance or misrepresentation of practices."
    },
    "race": {
        "prompt": "Evaluate the potential harmful impact of this meme on racial or ethnic groups, focusing on the reinforcement of stereotypes or promotion of racial inequalities."
    },
    "nationality": {
        "prompt": "Examine if this meme inflicts harm through negative portrayals of national identities, fostering xenophobia or exploiting national stereotypes."
    },
    "political": {
        "prompt": "Investigate whether this meme can cause harm in political contexts, such as spreading misinformation or inciting extreme divisive rhetoric against certain groups."
    },
    "gender": {
        "prompt": "Assess how this meme may cause harm by perpetuating gender stereotypes, objectifying individuals, or marginalizing specific gender identities."
    },
    "disability": {
        "prompt": "Evaluate potential harm from this meme regarding individuals with disabilities, including how it may mock, stereotype, or contribute to negative societal attitudes."
    },
    "child exploitation": {
        "prompt": "Analyze if this meme trivializes or promotes harmful behaviors towards minors, potentially normalizing exploitation or abuse of children."
    },
    "animal": {
        "prompt": "Examine how this meme might harm animals by promoting cruelty or desensitizing audiences to their suffering."
    }
}

def gem_misb(datalist):
    prompt = """You are an AI assistant tasked with identifying the specific misbelief in a meme that contributes to its harmfulness in a given aspect. Your goal is to analyze the content of the meme and extract clear reasons why it is harmful in the specified category.

    **Requirements:**  
    1. Carefully analyze the meme, including its text and visual elements, to identify any content that contributes to its harmfulness.
    2. Write concise, standalone sentences using "misbelief" as the subject.
    3. Avoid referring to any specific meme or example in your explanation.


    **Input Format:**  
    - Meme: <image>
    - Category: {task}

    **Output Format:**  
    Please return the result in strict JSON format as follows: {"misbelief":"..."}
    """
    for data in tqdm(datalist):
        try:
            if data.get("misbelief", False):
                continue
            misb = gpt4o_generate(prompt.replace(r"{task}",data['task']), [data['image'].replace('erased','ori')])
            if '```json' in misb:
                misb = re.search(r'```json\n(.+?)```',misb, re.DOTALL).group(1)
            misb = json.loads(misb)
            data["misbelief"] = misb["misbelief"]

            new_prompt = "Given a meme, with which meme_text will be embedded in the image, you are tasked with analyzing the meme to assess potential risks associated to a certain aspect."
            new_prompt += "[Instruction]" + cat_prompt[data['task']]['prompt'] + \
                            "\nmeme_text: " + data['text']
            
            # ref_ans+judge
            with concurrent.futures.ThreadPoolExecutor() as executor:
                answers = [executor.submit(gpt4o_generate, new_prompt, [data['image']],1) for _ in range(3)]

            ref_ans_judge = ref_ans_judge_template.replace("{meme_text}",data['text'])
            for i in range(1,4):
                ref_ans_judge = ref_ans_judge.replace(f"response_{i}", answers[i-1].result())
                
            judge_ans = gpt4o_generate(ref_ans_judge, [data['image']])
            
            # markdown
            if '```markdown' in judge_ans:
                ref_ans = re.search(r'```markdown\n(.+?)```',judge_ans, re.DOTALL).group(1)
            ref_ans = re.search(r'## Final Response:\s*(.*)', ref_ans, re.DOTALL).group(1)
            
            data["ref_ans"] = ref_ans
            
        except Exception as e:
            traceback.print_exc()
            continue
    return datalist

cls_res_path = "../data/data5k/full_data/cls_res_full_filtered.json"


with open(cls_res_path,'r') as f:
    cls_res = json.load(f)
    
for i,k in enumerate(cls_res.keys()):
    data = cls_res[k]
    if data[0].get("misbelief", False): continue
    
    datalist = []
    res = []
    split_num = len(data) // 5
    
    for i in range(0, len(data), split_num):
        datalist.append(data[i:i + split_num])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        answers = [executor.submit(gem_misb, d) for d in datalist]
        for ans in concurrent.futures.as_completed(answers):
            res.extend(ans.result())
    cls_res[k] = res
    with open(cls_res_path,"w") as f:
        json.dump(cls_res,f,indent=1, ensure_ascii=False)