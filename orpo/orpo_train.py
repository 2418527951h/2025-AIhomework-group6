import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import ORPOTrainer, ORPOConfig

# 环境设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRUST_REMOTE_CODE"] = "true"


# 生成 prompt
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


# 加载模型（标准 transformers + peft）
def setup_models(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # PEFT LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# 加载数学数据集并生成 chosen/rejected 对
def load_math_dataset(file_path):
    """加载数学数据集，生成 chosen（正确答案）和 rejected（错误答案）对"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            problem = item["problem"]
            answer = str(item["answer"]).strip()
            solution = item.get("solution", "").strip()
            
            # chosen: 包含正确答案的 completion
            chosen_completion = solution + f"\n\n\\boxed{{{answer}}}"
            
            # rejected: 生成一个错误的 completion（可以用简单的方式生成，比如错误答案）
            # 这里我们创建一个明显错误的答案作为 rejected
            wrong_answer = "0" if answer != "0" else "1"
            rejected_completion = solution + f"\n\n\\boxed{{{wrong_answer}}}"
            
            data.append({
                "prompt": make_prompt(problem),
                "chosen": chosen_completion,
                "rejected": rejected_completion,
                "answer": answer,  # 保留用于后续评估
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


# 主函数
def main():
    MODEL_NAME = "/home/huangzhenting/Qwen3/Qwen3-1.7B"
    DATA_PATH = "data/test.jsonl"
    seed = 42

    model, tokenizer = setup_models(MODEL_NAME)
    raw_dataset = load_math_dataset(DATA_PATH)
    train_dataset, eval_dataset = split_train_eval(raw_dataset, train_size=400, eval_size=100, seed=seed)

    config = ORPOConfig(
        output_dir=f"./orpo_output_{seed}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        max_grad_norm=0.3,
        logging_steps=10,
        save_steps=100,
        beta=0.1,
        remove_unused_columns=False,
        fp16=True,
        report_to="none",
        max_prompt_length=512,
        max_length=1536,
    )

    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"./orpo_final_model_{seed}")
    print("✅ ORPO 训练完成！")

if __name__ == "__main__":
    main()

