from unsloth import FastLanguageModel

import os
import json
import pickle
import torch
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import KTOTrainer, KTOConfig
from tqdm import tqdm

import re


# 前置设置
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_models():
    # 获取当前进程的 GPU ID
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {local_rank}] Loading model on {device}")

    # === 主模型 ===
    model_name = "/home/yangch25/Qwen3/Qwen3-1.7B"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,  
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=False,          
        trust_remote_code=True,
        device_map={"": device},     
        use_gradient_checkpointing=True,
    )

    # 添加 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing=True,
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # === 参考模型 ===
    print(f"[Rank {local_rank}] Loading reference model on {device}")
    reference_model, _ = FastLanguageModel.from_pretrained(
        model_name = model_name,  
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=False,
        trust_remote_code=True,
        device_map={"": device},     
        use_gradient_checkpointing=True,
    )
    reference_model.eval()

    return model, tokenizer, reference_model


def load_math_dataset(file_path):
    """加载本地 JSONL 数据集，并在 completion 中显式包含答案"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            completion = item["solution"].strip() + f"\n\n\\[\\boxed{{{item['answer']}}}\\]"

            data.append({
                "prompt": make_prompt(item["problem"]),
                "completion": completion, 
                "answer": item["answer"],
                "subject": item["subject"],
                "level": item["level"]
            })
            
    return Dataset.from_list(data)

def split_train_eval(dataset, train_size=400, eval_size=100, seed=42):
    """将 Dataset 拆分为训练集和 eval 集"""
    total_size = len(dataset)
    if train_size + eval_size > total_size:
        raise ValueError(f"train_size+eval_size={train_size+eval_size} 超过了数据总量 {total_size}")

    # 打乱顺序
    dataset = dataset.shuffle(seed=seed)
    
    # 用 select() 选择索引
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    
    return train_dataset, eval_dataset

def generate_comparison_samples(dataset, reference_model, tokenizer, num_samples=100, seed=42):
    """生成对比样本"""
    cache_file = f"kto_cache_seed{seed}.pkl"
    if os.path.exists(cache_file):
        print(f"📁 从缓存加载对比样本: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            return Dataset.from_list(cached_data)
    
    print("🔄 生成对比样本...")
    processed_data = []
    batch_size = 4
    num_samples = min(num_samples, len(dataset))
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_items = dataset[batch_start:batch_end]
        batch_prompts = batch_items["prompt"]
        batch_desired = batch_items["completion"]
        
        try:
            batch_inputs = tokenizer(
                [make_prompt(p) for p in batch_prompts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(reference_model.device)
            
            with torch.no_grad():
                batch_outputs = reference_model.generate(
                    batch_inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            for i, prompt in enumerate(batch_prompts):
                output_ids = batch_outputs[i][len(batch_inputs.input_ids[i]):]
                reference_completion = tokenizer.decode(output_ids, skip_special_tokens=True)
                processed_data.extend([
                    {"prompt": prompt, "completion": batch_desired[i], "label": True},
                    {"prompt": prompt, "completion": reference_completion, "label": False}
                ])
        except Exception as e:
            print(f"❌ 批处理出错: {e}")
            continue
    
    with open(cache_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"💾 对比样本已保存到: {cache_file}")
    
    return Dataset.from_list(processed_data)
      

    
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
    """
    从生成文本中提取 \boxed{...} 内的内容，支持任意嵌套花括号。
    如果找不到，返回 "nobox"。
    """
    # 查找所有 \boxed{ 的位置
    start_idx = text.find(r'\boxed{')
    if start_idx == -1:
        start_idx = text.find(r'\\boxed{')  # 兼容双反斜杠
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

    
def main():
    # 1. 模型
    model, tokenizer, reference_model = setup_models()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    seed=43
    
    # 2. 数据
    raw_dataset = load_math_dataset("data/test.jsonl")

    # 拆分训练集和 eval 集
    train_dataset, eval_dataset = split_train_eval(raw_dataset, train_size=400, eval_size=100, seed=seed)
    
    # 3. 对比样本
    # kto_dataset = generate_comparison_samples(raw_dataset, reference_model, tokenizer, num_samples=500, seed=seed)
    kto_dataset = train_dataset.map(lambda x: {"label": True})

    
    # 5. KTO 配置
    kto_config = KTOConfig(
        output_dir=f"./output_Qwen3-1.7B_kto_math500_{seed}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        max_grad_norm=0.3,
        logging_steps=10,
        save_steps=100,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=0,
        remove_unused_columns=False,
        report_to=None,
        fp16 = True,
    )

    # 6. KTOTrainer 
    trainer = KTOTrainer(
        model=model,
        ref_model=reference_model,
        args=kto_config,
        train_dataset=kto_dataset,

        processing_class=tokenizer,
        peft_config=None,
        remove_unused_columns=False,
        max_length=2048,              
        max_prompt_length=1024,       
    )
    
    # 7. 训练
    print("开始 KTO 训练...")
    trainer.train()
    
    # 8. 保存
    trainer.save_model(f"./Qwen3-1.7B_math500_kto_final_{seed}")
    print("训练完成！")

    

if __name__ == "__main__":
    main()
