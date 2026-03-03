import json
import random

random.seed(42)

with open("data/source_info.jsonl", "r") as f:
    sources = [json.loads(line) for line in f]
with open("data/response.jsonl", "r") as f:
    responses = [json.loads(line) for line in f]
    
    
data = []
for i in range(len(sources)):
    source_id = sources[i]["source_id"]
    task_type = sources[i]["task_type"]
    source = sources[i]["source"]
    source_info = sources[i]["source_info"]
    prompt = sources[i]["prompt"]
    
    for j in range(i*6, i*6+6):
        if responses[j]["source_id"] != source_id:
            print("error")
        id_name =responses[j]["id"]
        model = responses[j]["model"]
        temperature = responses[j]["temperature"]
        labels = responses[j]["labels"]
        split = responses[j]["split"]
        quality = responses[j]["quality"]
        response = responses[j]["response"]
        
        data.append({
            "id_name": id_name,
            "source_id": source_id,
            "task_type": task_type,
            "model": model,
            "source": source,
            "source_info": source_info,
            "response": response,
            "labels": labels,
            "split": split,    
            "prompt": prompt, # データ生成時に使用されたプロンプト
            "temperature": temperature,
            "quality": quality,
        })

for d in data: # hallucination検出用のプロンプト
    input_text = ""
    if d["task_type"] == "QA":
        input_text = f"""Below is a question:
{d["source_info"]["question"]}
Below are related passages:
{d["source_info"]["passages"]}
Below is an answer:
{d["response"]}
Your task is to determine whether the answer contains either or both of the following two types of hallucinations:
1. conflict: instances where the answer presents direct contraction or opposition to the passages;
2. baseless info: instances where the answer includes information which is not substantiated by or inferred from the passages.
Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}.
Output:
"""
    elif d["task_type"] == "Summary":
        input_text = f"""Below is the original news:
{d["source_info"]}
Below is a summary of the news:
{d["response"]}
Your task is to determine whether the summary contains either or both of the following two types of hallucinations:
1. conflict: instances where the summary presents direct contraction or opposition to the original news;
2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news.
Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination list": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination list": []}}.
Output:
"""
    d["input_text"] = input_text

# 今回はData2txtのタスクは含めない
train_data = [d for d in data if d["split"] == "train" and d["task_type"] != "Data2txt"]
test_data = [d for d in data if d["split"] == "test" and d["task_type"] != "Data2txt"]
dev_data_qa = random.sample([d for d in train_data if d["task_type"] == "QA"], 900)
dev_data_sum = random.sample([d for d in train_data if d["task_type"] == "Summary"], 900)
dev_data = dev_data_qa + dev_data_sum
train_data = [d for d in train_data if d not in dev_data]

with open("data/ft_train.jsonl", "w") as f:
    for d in train_data:
        f.write(json.dumps(d)+"\n")
with open("data/ft_dev.jsonl", "w") as f:
    for d in dev_data:
        f.write(json.dumps(d)+"\n")
with open("data/ft_test.jsonl", "w") as f:
    for d in test_data:
        f.write(json.dumps(d)+"\n")