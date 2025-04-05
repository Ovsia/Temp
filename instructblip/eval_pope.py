import json
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm

# 路径设置
data_file = '/d/mqy/program2/LLaVA-main/pope/llava_pope_test.jsonl'
image_dir = '/s/COCO2014/val2014/'
output_file = '/d/mqy/program2/LAVIS-main/projects/instructblip/13b-pope.jsonl'

# 加载模型和处理器
model = InstructBlipForConditionalGeneration.from_pretrained("/d/mqy/pretrained/instructblip-vicuna-13b", torch_dtype=torch.float16)
processor = InstructBlipProcessor.from_pretrained("/d/mqy/pretrained/instructblip-vicuna-13b")

# 设置设备
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# 读取数据文件
with open(data_file, 'r') as f:
    data = [json.loads(line) for line in f]

# 处理每一行数据
results = []
for entry in tqdm(data):
    question_id = entry['question_id']
    image_name = entry['image']
    prompt = entry['text']

    # 构建图像路径并加载图像
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB")

    # 处理输入
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # 生成输出
    outputs = model.module.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    # 保存结果
    result = {
        "question_id": question_id,
        "prompt": prompt,
        "text": generated_text,
        "model_id": "instructblip-13b",
        "metadata": {}
    }
    results.append(result)

# 将结果保存到 JSONL 文件
with open(output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print("Evaluation completed and results saved to", output_file)
