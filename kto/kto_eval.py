import torch
import re
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ======================
# 1. 模型与数据路径配置
# ======================
seed=43

model_path = "/home/yangch25/Qwen3/Qwen3-1.7B/" 
output_file = f"eval_qwen.txt_{seed}"

# model_path = f"Qwen3-1.7B_math500_kto_final_{seed}"  
# output_file = "eval_kto_{seed}.txt"
data_path = "data/test.jsonl" 



# ======================
# 2. 加载模型与Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  

# ======================
# 3. Prompt模板 & 提取函数
# ======================

from kto_train import make_prompt, extract_answer_from_completion

# ======================
# 4. 加载测试集
# ======================
def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj["problem"], obj["answer"]))
    return data

dataset = load_dataset(data_path)
print(f"✅ Loaded {len(dataset)} samples for evaluation.")

# ======================
# 5. 拆分训练/验证子集（仅测试用）
# ======================
def split_train_eval(dataset, train_size=400, eval_size=100, seed=42):
    """从列表中随机划分训练和验证"""
    random.seed(seed)
    random.shuffle(dataset)
    train_data = dataset[:train_size]
    eval_data = dataset[train_size:train_size + eval_size]
    return train_data, eval_data

_, dataset = split_train_eval(dataset, train_size=400, eval_size=100, seed=seed)
print(f"🧪 Using {len(dataset)} samples for evaluation.")

# ======================
# 6. 推理与评估
# ======================
generation_config = GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    do_sample=False,
    max_new_tokens=1024,
)

correct = 0


with open(output_file, "w", encoding="utf-8") as f:  
    for i, (q, gold) in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = make_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = full_text[len(prompt):].strip()
        pred = extract_answer_from_completion(gen_text)
    
        if pred == gold:
            correct += 1
            

        info_lines = [
            "\n" + "-"*80,
            f"Q: {q}",
            "模型输出：",
            gen_text,
            f"Pred: '{pred}' | Gold: '{gold}' | {'✅' if pred==gold else '❌'}"
        ]
        info_str = "\n".join(info_lines) + "\n"
    
        # 写入文件
        f.write(f"\nindex_{i}")
        f.write(info_str)
    
        # # 打印前几个样例
        # if i < 5:
        #     print(info_str)

accuracy = correct / len(dataset)
print(f"\n🎯 Evaluation finished! Accuracy: {accuracy:.2%}")
