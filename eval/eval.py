import json
import re
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import sympify, simplify
from sympy.core.sympify import SympifyError

system_pe = "You are a helpful AI assistant. When presented with questions, think step by step to reach conclusions. {{ content | trim }} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."

def normalize_answer(ans):
    if ans is None:
        return None
    try:
        ans = ans.strip()
        expr = sympify(ans, evaluate=True)
        return simplify(expr)
    except (SympifyError, TypeError, ValueError):
        return ans


def extract_boxed_answer(text):
    # 尝试匹配 \boxed{...}，支持简单嵌套（实际中通常无嵌套）
    match = re.search(r'\\boxed\{([^{}]*)\}', text)
    if match:
        return match.group(1).strip()
    # 更宽松的匹配（允许花括号内有转义等）
    match = re.search(r'\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', text)
    if match:
        return match.group(1).strip()
    return None


def extract_final_answer(output_text):
    think_split = output_text.split('<\\think>')
    after_think = think_split[-1] if len(think_split) > 1 else output_text
    return extract_boxed_answer(after_think)


def compare_answers(pred, gold):
    if pred is None or gold is None:
        return False
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    try:
        if hasattr(norm_pred, 'equals') and hasattr(norm_gold, 'equals'):
            return bool(norm_pred.equals(norm_gold))
        else:
            return str(norm_pred) == str(norm_gold)
    except Exception:
        return str(pred).strip() == str(gold).strip()

def sanitize_model_name(model_name):
    # 将模型名转为合法文件名（如 meta-llama/Llama-3-8b -> Llama-3-8b）
    return os.path.basename(model_name.rstrip('/'))
from tqdm import tqdm  # 进度条库，可 pip install tqdm


def evaluate_dataset_batch(model, tokenizer, jsonl_path, device, max_new_tokens=512, batch_size=8):
    total = 0
    correct = 0
    total_tokens = 0
    total_time = 0.0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    questions = []
    gold_answers = []
    for line in lines:
        item = json.loads(line)
        question = item.get("question", "") or item.get("Problem", "") or item.get("problem", "")
        gold_answer = item.get("answer", "") or item.get("solution", "") or item.get("Solution", "")
        questions.append(f"{system_pe}\nQuestion:\n{question}\nAnswer:")
        gold_answers.append(gold_answer)

    num_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {os.path.basename(jsonl_path)}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(questions))
        batch_questions = questions[start:end]
        batch_gold = gold_answers[start:end]

        # ✅ PRIMARY FIX: Move the tokenized inputs to the correct device.
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        end_time = time.time()
        total_time += (end_time - start_time)

        for j, output in enumerate(outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            pred_answer_str = extract_final_answer(output_text)
            if compare_answers(pred_answer_str, batch_gold[j]):
                correct += 1
            num_input_tokens = inputs['input_ids'][j].shape[0]
            num_output_tokens = output.shape[0]
            total_tokens += num_output_tokens - num_input_tokens
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    return {
        'dataset': os.path.basename(jsonl_path),
        'size': total,
        'accuracy': accuracy,
        'tokens_per_sec': tokens_per_sec,
        'total_time_sec': total_time,
        'total_tokens': total_tokens
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=False,
                        default='/mnt/bn/rllm-lf/mlx/users/chenhaoyu.awk/models/Qwen3-1.7B-BF16')
    parser.add_argument('--data_dir', type=str, required=False,
                        default='/mnt/bn/rllm-lf/mlx/users/chenhaoyu.awk/lxh_eval/data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_new_tokens', type=int, default=16888)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
        trust_remote_code=True
    )

    # ✅ SECONDARY FIX: Correctly and robustly move the model to the specified device.
    model.eval()
    model.to(args.device)

    jsonl_files = [f for f in os.listdir(args.data_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.data_dir}")

    results = []
    for file in sorted(jsonl_files):
        full_path = os.path.join(args.data_dir, file)
        result = evaluate_dataset_batch(
            model, tokenizer, full_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )
        results.append(result)
        print(f"Dataset: {result['dataset']}")
        print(f"  Size: {result['size']}")
        print(f"  Accuracy: {result['accuracy'] * 100:.2f}%")
        print(f"  Tokens/sec: {result['tokens_per_sec']:.2f}")
        print(f"  Total time (s): {result['total_time_sec']:.2f}")
        print("-" * 40)

    clean_name = sanitize_model_name(args.model_name_or_path)
    output_file = f"{clean_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
