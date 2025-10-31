import os
import torch
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel  

# ======================
# 1. 配置
# ======================
seed = 42

model_path = "/home/huangzhenting/Qwen3/Qwen3-1.7B"         
adapter_path = f"./orpo_final_model_{seed}"          

output_file = f"eval_orpo_{seed}.txt"
data_path = "data/test.jsonl"

# ======================
# 2. 加载 tokenizer + 基础模型 + LoRA 适配器
# ======================
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True,
)

# 加载 LoRA 适配器（如果存在）
if adapter_path and os.path.exists(adapter_path):
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"✅ 已加载 LoRA 适配器: {adapter_path}")

model.eval()  

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ======================
# 3. Prompt 和答案提取（复用你的函数）
# ======================

def make_prompt(problem):
    return (
        "You are a precise mathematical problem solver.\n"
        "Solve the problem step-by-step, then provide the final answer in exactly one place.\n"
        "Read the question carefully and make sure your \\boxed{} contains what is being asked for.\n\n"
        
        "**MATHEMATICAL ACCURACY**: Check all calculations carefully, especially when:\n"
        "- Performing algebraic manipulations\n" 
        "- Using the Euclidean algorithm or divisibility arguments\n"
        "- Subtracting or combining expressions\n"
        "- Working with modular arithmetic\n"
        "Verify each step before proceeding to the next.\n\n"
        
        "**FINAL ANSWER FORMAT RULES**:\n"
        "- Enclose **ONLY the final, fully simplified answer** inside a single LaTeX \\boxed{} expression.\n"
        "- The content must be a pure mathematical object: number, integer, fraction, or comma-separated list (if multiple answers are requested).\n"
        "- NEVER include:\n"
        "  • Text, labels, or units (e.g., 'grade', 'students', '%', 'th')\n"
        "  • LaTeX text commands like \\mathrm{}, \\text{}, ^{\\mathrm{th}}, etc.—even if they appear in the problem\n"
        "  • Equations, assignments, or arithmetic expressions (e.g., '8 + 2', 'x = 5', '8 + 2 = 10')\n"
        "  • Variables or symbolic expressions (e.g., '10^x') when a numerical answer is expected\n"
        "- If the problem provides concrete data (tables, percentages, counts), you MUST compute a numerical result.\n"
        "- NEVER copy formatting from the problem statement. If it says '12^{\\mathrm{th}} grade', output \\boxed{12}.\n\n"

        "**Examples of CORRECT vs INCORRECT**:\n"
        "WRONG: \\boxed{12^{\\mathrm{th}}} → CORRECT: \\boxed{12}\n"
        "WRONG: \\boxed{75\\%} → CORRECT: \\boxed{75}\n"
        "WRONG: \\boxed{10^x} (when solvable) → CORRECT: \\boxed{2}\n"
        "WRONG: \\boxed{8 + 2 = 10} → CORRECT: \\boxed{10}\n"
        "WRONG: \\boxed{a = 2} → CORRECT: \\boxed{2}\n"
        "WRONG: \\boxed{answer is 12} → CORRECT: \\boxed{12}\n"
        "CORRECT: Solutions are -1, 0, 5 and list requested → \\boxed{-1, 0, 5}\n\n"

        f"### Problem:\n{problem}\n\n### Solution:\n"
    )

def extract_answer_from_completion(text: str) -> str:
    start_idx = text.find(r'\boxed{')
    if start_idx == -1:
        start_idx = text.find(r'\\boxed{')
        if start_idx == -1:
            return "nobox"
        else:
            start_idx += len(r'\\boxed{')
            brace_start = start_idx
    else:
        start_idx += len(r'\boxed{')
        brace_start = start_idx

    depth = 1
    i = brace_start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        content = text[brace_start:i-1]
        return content.strip().replace('\n', '').strip()
    else:
        return "nobox"

# ======================
# 4. 加载数据
# ======================
def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj["problem"], str(obj["answer"]).strip()))
    return data

dataset = load_dataset(data_path)
print(f"✅ Loaded {len(dataset)} samples for evaluation.")

def split_train_eval(dataset, train_size=400, eval_size=100, seed=42):
    """将 Dataset 拆分为训练集和 eval 集"""
    total_size = len(dataset)
    if train_size + eval_size > total_size:
        raise ValueError(f"train_size+eval_size={train_size+eval_size} 超过了数据总量 {total_size}")

    random.seed(seed)
    random.shuffle(dataset)

    train_data = dataset[:train_size]
    eval_data = dataset[train_size:train_size + eval_size]
    
    return train_data, eval_data
    
_, eval_dataset = split_train_eval(dataset, train_size=400, eval_size=100, seed=seed)

print(f"🧪 Using {len(eval_dataset)} samples for evaluation.")

# ======================
# 6. 推理评估
# ======================
generation_config = GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    do_sample=False,
    max_new_tokens=1024,
)

correct = 0

with open(output_file, "w", encoding="utf-8") as f:
    for i, (q, gold) in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        prompt = make_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
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
            f"Pred: '{pred}' | Gold: '{gold}' | {'✅' if pred == gold else '❌'}"
        ]
        info_str = "\n".join(info_lines) + "\n"
    
        f.write(f"\nindex_{i}")
        f.write(info_str)

accuracy = correct / len(eval_dataset)
print(f"\n🎯 Evaluation finished! Accuracy: {accuracy:.2%}")

