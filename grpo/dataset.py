from datasets import load_dataset, load_from_disk
import json


template0 = "You are a helpful AI assistant. When presented with questions, think step by step to reach conclusions. "
template1 = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."


def convert_to_grpo_format(batch):
    # 从数据中提取问题、解答、答案
    problems = batch["problem"]
    solutions = batch["solution"]
    answers = batch["answer"]
    
    # 标准格式prompt
    prompts = [template0 + "\n" + problem + "\n" + template1 for problem in problems ]
    
    # 将格式化后的数据添加到批量数据中并返回 
    batch["ground_truth"] = answers
    batch["prompt"] = prompts
    return batch


def convert2json():
    # 加载数据集
    dataset = load_from_disk('./dataset_math500_grpo')

    # 指定保存路径
    output_dir = "./dataset_math500_grpo_json"

    # 将数据集转换为 JSON 格式
    dataset.to_json(f"{output_dir}/train.json", orient="records", lines=True)


def format_example():
    # 加载 Hugging Face 数据集
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # 使用 map 函数将整个数据集转换为所需格式
    formatted_data = dataset.map(convert_to_grpo_format, batched=True)

    formatted_data.save_to_disk('./dataset_math500_grpo')
    print("数据转换完成并已保存到 './dataset_math500_grpo' 目录。")
    
    print(formatted_data[0])
    
    
if __name__ == "__main__":
    format_example()
