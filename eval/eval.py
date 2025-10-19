import json
import re
import os
import time
import torch

from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import sympify, simplify
from sympy.core.sympify import SympifyError
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

system_pe = "You are a helpful AI assistant. When presented with questions, think step by step to reach conclusions.\n{{ content }}\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
prompt_template = Template(system_pe)

def normalize_answer(ans):
    if ans is None:
        return None
    try:
        ans = ans.strip()
        expr = sympify(ans, evaluate=True)
        return simplify(expr)
    except (SympifyError, TypeError, ValueError):
        return ans
    except Exception as e:
        print(f"Error normalizing answer: {ans}, error: {e}")
        return ans

def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^{}]*)\}', text)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r'\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*?)\}', text)
    if matches:
        return matches[-1].strip()

    return None

def extract_final_answer(output_text):
    think_split = output_text.split('</think>')
    after_think = think_split[-1] if len(think_split) > 1 else output_text
    return extract_boxed_answer(after_think)

def compare_answers(pred, gold):
    if pred is None or gold is None:
        return False
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    print(f"norm_pred:{norm_pred}, norm_gold:{norm_gold}")
    try:
        if hasattr(norm_pred, 'equals') and hasattr(norm_gold, 'equals'):
            return bool(norm_pred.equals(norm_gold))
        else:
            return str(norm_pred) == str(norm_gold)
    except Exception:
        return str(pred).strip() == str(gold).strip()

def sanitize_model_name(model_name):
    return os.path.basename(model_name.rstrip('/'))

# ---------- 数据并行评测函数 ----------

def prepare_batches(questions, tokenizer, batch_size=8):
    batches = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]
    tokenized_batches = []
    tokenizer.padding_side = "left"
    for batch in batches:
        tokenized_batches.append(
            tokenizer(batch, return_tensors="pt", padding="longest", truncation=True).to("cuda")
        )
    tokenizer.padding_side = "right"
    return tokenized_batches

def evaluate_dataset_parallel(model, tokenizer, jsonl_path, accelerator, max_new_tokens=512, batch_size=8):
    # 读取 jsonl
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    questions, gold_answers = [], []
    for line in lines:
        item = json.loads(line)
        question = item.get("question", "") or item.get("Problem", "") or item.get("problem", "")
        gold_answer = item.get("Answer", "") or item.get("answer", "")
        prompt = prompt_template.render(content=question)
        questions.append(prompt)
        gold_answers.append(gold_answer)

    # 分配到每个进程/GPU
    with accelerator.split_between_processes(questions) as questions_subset:
        results = dict(outputs=[], num_tokens=0, correct=0, total=0)
        token_batches = prepare_batches(questions_subset, tokenizer, batch_size=batch_size)

        for batch_tokenized in token_batches:
            with torch.no_grad():
                outputs_tokenized = model.generate(
                    **batch_tokenized,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            # remove prompt from output
            outputs_tokenized = [
                out[len(inp):] for inp, out in zip(batch_tokenized["input_ids"], outputs_tokenized)
            ]
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            results["outputs"].extend(outputs)
            results["num_tokens"] += sum([len(t) for t in outputs_tokenized])

            # 比较答案
            for i, out_text in enumerate(outputs):
                pred_ans = extract_final_answer(out_text)
                if compare_answers(str(pred_ans), str(gold_answers[i])):
                    results["correct"] += 1
                results["total"] += 1

    # gather all GPU results
    results_gathered = gather_object([results])
    if accelerator.is_main_process:
        total_tokens = sum(r["num_tokens"] for r in results_gathered)
        total_correct = sum(r["correct"] for r in results_gathered)
        total_samples = sum(r["total"] for r in results_gathered)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {
            "dataset": os.path.basename(jsonl_path),
            "size": total_samples,
            "accuracy": accuracy,
            "total_tokens": total_tokens
        }
    return None

# ---------- main ----------

import argparse
import os
import time
import json
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM


def sanitize_model_name(name: str):
    """移除路径符号，生成安全的输出文件名"""
    return os.path.basename(name).replace('/', '_').replace(':', '_')


def write_json(path, data):
    """安全写入 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, required=False, default='')
    parser.add_argument(
        '--dataset_name',
        type=str,
        nargs='+',  # ✅ 支持多个数据集名
        required=False,
        default=['aime_2024', 'aime_2025'],
        help='要处理的数据集名称列表（不含后缀），例如 gsm8k truthfulqa mmlu'
    )
    parser.add_argument('--max_new_tokens', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    model.to(accelerator.device)

    results_all = []
    start_time = time.time()

    # ✅ 支持多个数据集循环
    for dataset_name in args.dataset_name:
        dataset_path = os.path.join(args.data_dir, f"{dataset_name}.jsonl")
        if not os.path.exists(dataset_path):
            if accelerator.is_main_process:
                print(f"⚠️ 数据集不存在: {dataset_path}, 跳过。")
            continue

        # 评测单个数据集
        result = evaluate_dataset_parallel(
            model, tokenizer, dataset_path, accelerator,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )

        if accelerator.is_main_process and result is not None:
            results_all.append(result)
            print(f"✅ Dataset: {result['dataset']}")
            print(f"   Size: {result['size']}")
            print(f"   Accuracy: {result['accuracy']*100:.2f}%")
            print(f"   Total tokens: {result['total_tokens']}")
            print("-" * 40)

            # 每个数据集处理完立即写出结果
            output_file = f"{sanitize_model_name(args.model_name_or_path)}_results.json"
            write_json(output_file, results_all)
            print(f"💾 Results updated -> {output_file}")

    if accelerator.is_main_process:
        print("🎯 All datasets processed.")
        print("Total elapsed time:", time.time() - start_time)

if __name__ == "__main__":
    main()