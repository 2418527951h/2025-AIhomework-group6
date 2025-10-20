from datasets import load_dataset
from datasets import load_from_disk
import sys 
sys.path.append("..") 
from eval.eval import extract_final_answer, compare_answers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import torch

# Reward function
def reward_func(completions, ground_truth, **kwargs):
    pred_ans = [extract_final_answer(completion) for completion in completions]
    return [1.0 if compare_answers(str(pre), str(ans)) else 0.0 for pre, ans in zip(pred_ans, ground_truth)]

# 配置QLoRA参数
def get_peft_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

# 初始化GRPOConfig
def get_grpo_config():
    # 初始化GRPOConfig，通过peft_config传入QLoRA配置
    grpo_config = GRPOConfig(
        output_dir="./grpo_qlora_output",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True, 
        num_generations=2,  # 每个prompt生成样本数
        max_prompt_length=512,
        max_completion_length=1024,
    )
    return grpo_config


if __name__ == "__main__":
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained("/home/manager/.cache/modelscope/hub/models/Qwen/Qwen3-8B", 
                                                quantization_config=quantization_config)
    
    peft_config = get_peft_config()
    model = get_peft_model (model, peft_config)
    
    ds = load_from_disk("./dataset_math500_grpo")
    grpo_config = get_grpo_config()
    trainer = GRPOTrainer(
        model=model,
        args = grpo_config,
        reward_funcs=reward_func,
        train_dataset=ds,
    )
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    output_dir = "./grpo_qlora_model"
    # Save
    trainer.save_model(output_dir)
    trainer.accelerator.print(f"Model saved to {output_dir}.")

    # prompts = ['You are a helpful AI assistant. When presented with questions, think step by step to reach conclusions. \nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.']
    # completions = [r"<think>To convert a point from rectangular coordinates to polar coordinates, we use the following formulas:1. The radial distance ( r ) is given by the formula ( r = \sqrt{x^2 + y^2} ), where ( x ) and ( y ) are the rectangular coordinates.2. The angle ( \theta ) is given by the formula ( \theta = \tan^{-1} \left( \frac{y}{x} \right) ), but we must ensure that the angle is in the correct quadrant.For the point ( (0, 3) ), the coordinates are ( x = 0 ) and ( y = 3 ).* First, calculate ( r ):[r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3.]* Now, we need to find ( \theta ). Since ( x = 0 ), the point lies on the ( y )-axis. The angle corresponding to a point on the positive ( y )-axis is ( \frac{\pi}{2} ) radians.Thus, the polar coordinates of the point ( (0, 3) ) are ( (r, \theta) = (3, \frac{\pi}{2}) ). </think>The final answer is ( \boxed{(3, \frac{\pi}{2})} )."]
    # ground_truth = ["(3, \\frac{\\pi}{2})"]
    # print (reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth))