from datasets import load_dataset
from datasets import load_from_disk
from trl import GRPOTrainer
import sys 
sys.path.append("..") 
from eval.eval import extract_final_answer, compare_answers


ds = load_from_disk("./dataset_math500_grpo")
output_dir = '/saves'

# Reward function
def reward_func(completions, ground_truth, **kwargs):
    pred_ans = [extract_final_answer(completion) for completion in completions]
    return [1.0 if compare_answers(str(pre), str(ans)) else 0.0 for pre, ans in zip(pred_ans, ground_truth)]

trainer = GRPOTrainer(
    model="/home/manager/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    reward_funcs=reward_func,
    train_dataset=ds,
)
trainer.train()

# Log training complete
trainer.accelerator.print("✅ Training completed.")

# # Save
# trainer.save_model(output_dir)
# trainer.accelerator.print(f"💾 Model saved to {output_dir}.")

# prompts = ['You are a helpful AI assistant. When presented with questions, think step by step to reach conclusions. \nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.']
# completions = [r"<think>To convert a point from rectangular coordinates to polar coordinates, we use the following formulas:1. The radial distance ( r ) is given by the formula ( r = \sqrt{x^2 + y^2} ), where ( x ) and ( y ) are the rectangular coordinates.2. The angle ( \theta ) is given by the formula ( \theta = \tan^{-1} \left( \frac{y}{x} \right) ), but we must ensure that the angle is in the correct quadrant.For the point ( (0, 3) ), the coordinates are ( x = 0 ) and ( y = 3 ).* First, calculate ( r ):[r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3.]* Now, we need to find ( \theta ). Since ( x = 0 ), the point lies on the ( y )-axis. The angle corresponding to a point on the positive ( y )-axis is ( \frac{\pi}{2} ) radians.Thus, the polar coordinates of the point ( (0, 3) ) are ( (r, \theta) = (3, \frac{\pi}{2}) ). </think>The final answer is ( \boxed{(3, \frac{\pi}{2})} )."]
# ground_truth = ["(3, \\frac{\\pi}{2})"]
# print (reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth))