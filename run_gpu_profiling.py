import os
import time
import json
import torch
import thop
# import psutil
# import tracemalloc
# from huggingface_hub import login
from PIL import Image
# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from transformers.image_utils import load_image
# from config import model_name, dataset, dataset_json, models_directory
import csv, argparse
from pynvml import *
import argparse
import builtins
from typing import List
from model_config import model_config
import torchvision.models
import timeit
import onnxruntime as ort

# Workaround for broken MINICPM code that doesn't import List
builtins.List = List

# ============ Parse CLI Arguments ============
def parse_args():
    parser = argparse.ArgumentParser(description='Model Profiling on GPU')
    parser.add_argument("-m", "--model_id", type=str, default=None, help="Model ID to run")
    parser.add_argument("-o", "--output_file", type=str, default="profiles.csv", help="Output CSV file")
    return parser.parse_args()

# ============ Setup ============
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(device)


# ============ Metrics ============

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # assuming single-GPU
    device_name = nvmlDeviceGetName(handle)
    return handle, device_name

def get_gpu_memory_and_util(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_mb": mem_info.used / 1024**2,
        "gpu_util_percent": util.gpu
    }

# ============ Load Model ============
def load_model(model_variant, weights, input_shape, handle):
    
    model = getattr(torchvision.models, model_variant)(weights=weights).eval()

    onnx_path = os.path.join("onnx_models", f"{model_variant}.onnx")
    if not os.path.exists(onnx_path):
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=12
        )

    file_size_mb = os.path.getsize(onnx_path) / (1024**2)

    mem_before_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
    try:
        start_time = timeit.default_timer()
        session = ort.InferenceSession(
            onnx_path, 
            providers=["CUDAExecutionProvider"]
        )
        load_duration_ms = (timeit.default_timer() - start_time) * 1000
        memory_after_load_mb = get_gpu_memory_and_util(handle)["gpu_mem_used_mb"]
        print(f"Mem before load: {mem_before_load_mb}, Mem after load: {memory_after_load_mb}")
        mem_usage_mb = memory_after_load_mb - mem_before_load_mb
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_variant}: {e}")
 
    return model, load_duration_ms, mem_usage_mb, file_size_mb


def get_flops(model, input_shape):
    dummy_input = torch.randn(input_shape)
    flops, params = thop.profile(model, inputs=(dummy_input,))
    return flops/1e9, params/1e6 # GFLOPs, Millions of parameters

# ============ Load Dataset ============
def load_dataset(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    return list(raw_data.values())

# def run_inference(model, processor, data, handle):
    model.eval()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_latency = 0
    correct = 0
    evaluated = 0
    total_gpu_util = 0
    total_gpu_mem = 0
    total_cpu_util = 0
    total_cpu_mem = 0

    tracemalloc.start()
    gpu_before = get_gpu_memory_and_util(handle)
    # print("GPU Before: ", gpu_before)
    cpu_before = get_cpu_memory_and_util()
    start_time = time.time()

    is_moondream = "moondream" in model_name
    is_qwen = "qwen" in model_name.lower()
    is_molmo = "molmo" in model_name.lower()
    is_phi = "phi" in model_name.lower()
    is_smolvlm = "smolvlm" in model_name.lower()
    is_llama = "llama-3.2" in model_name.lower()
    is_minicpm = "minicpm" in model_name.lower()

    for i, item in enumerate(data):
        print("\nRunning inference")
        question = item["question"]
        gt_answer = item.get("answer", "").strip().lower()
        image_id = item["image_id"]
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join("dataset", "val2014", image_name)

        if not os.path.exists(image_path):
            print(f"Skipping [{i}] - Image not found: {image_path, image_name}")
            continue
        print(f"✅ Using [{i}] - Found image: {image_name}")

        if not is_smolvlm:
            image = Image.open(image_path).convert("RGB")
        else:
            image = load_image(image_path).resize((512, 512), Image.LANCZOS)

        # conversation = [{"role": "user", "content": question}]
        # prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        start = time.time()
        
        with torch.no_grad():
            if is_moondream:
                # image_embeds = model.encode_image(image)
                # answer = model.answer_question(image_embeds=image_embeds, question=question+ " Please answer in one word.")
                answer = model.query(image, question + " Please answer in one word.")["answer"]
                input_tokens = len(question.split())
                generated_tokens = len(answer.split())
            elif is_qwen:
                from qwen_vl_utils import process_vision_info
                # Build messages for vision + text input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question+" Please answer in one word."},
                        ],}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=20)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
                answer = response.split()[0] if response else ""
                input_tokens = inputs.input_ids.shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            elif is_molmo:
                processed = processor.process(images=[image], text=question+ " Please answer in one word.")
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in processed.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                generated_token_ids = torch.cat([inputs["input_ids"], next_token_id[:, None]], dim=1)
                answer = processor.tokenizer.decode(generated_token_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = 1
            elif is_phi:
                # Add image placeholder to match expected format
                prompt = f"<|image_1|>\n{question} Please answer in one word."
                messages = [{"role": "user", "content": prompt}]
                
                # Create chat-style prompt with special tokens
                text_prompt = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Tokenize + process image(s) together
                inputs = processor(text_prompt, [image], return_tensors="pt").to(model.device)

                # Run inference
                outputs = model.generate(
                    **inputs,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    temperature=0.0,
                    do_sample=False,
                )

                # Strip off prompt tokens
                generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                answer = response.split()[0] if response else ""

                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = generated_ids.shape[1]
            elif is_smolvlm:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question + " Please answer in one word."}
                        ]
                    }
                ]
                # print(f"[DEBUG] Original image size: {image.size}")
                # image = image.copy().resize((512, 512), Image.LANCZOS)
                # print(f"[DEBUG] Resized image size: {image.size}")
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=20)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                import re
                match = re.search(r"assistant\s*:?[\s\n]*(\w+)", response, flags=re.IGNORECASE)
                answer = match.group(1) if match else response.strip().split()[0]

                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            elif is_llama:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question + " Please answer in one word."}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                generate_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                outputs = model.generate(**generate_inputs, max_new_tokens=20)
                response = processor.decode(outputs[0], skip_special_tokens=True).strip()
                import re
                match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", response, re.IGNORECASE | re.DOTALL)
                if match:
                    answer_chunk = match.group(1).strip()
                    answer = answer_chunk.split()[0] if answer_chunk else ""
                else:
                    answer = response.strip().split()[0]
                input_tokens = inputs["input_ids"].shape[1]
                generated_tokens = outputs.shape[1] - input_tokens
            elif is_minicpm:
                question_prompt = question + " Please answer in one word."
                msgs = [{"role": "user", "content": [image, question_prompt]}]
                
                with torch.no_grad():
                    response = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=processor,  # tokenizer = processor in this case
                    )
                
                answer = response.strip().split()[0]
                input_tokens = len(question_prompt.split())
                generated_tokens = len(answer.split())
            else:
                prompt = f"USER: <image>\n{question}\nPlease answer in one word.\nASSISTANT:"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                input_tokens = inputs["input_ids"].shape[1]
                outputs = model.generate(**inputs, max_new_tokens=10)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                answer = response.split("ASSISTANT:")[-1].strip().split()[0]
                generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        latency = time.time() - start

        total_prompt_tokens += input_tokens
        total_generated_tokens += generated_tokens
        total_latency += latency

        gpu_after = get_gpu_memory_and_util(handle)
        cpu_after = get_cpu_memory_and_util()
        gpu_mem_delta = gpu_after["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
        gpu_util_delta = gpu_after["gpu_util_percent"] - gpu_before["gpu_util_percent"]
        cpu_mem_delta = cpu_after["cpu_mem_used_mb"] - cpu_before["cpu_mem_used_mb"]
        cpu_util_delta = cpu_after["cpu_util_percent"] - cpu_before["cpu_util_percent"]
        print(f"[{i}] GPU util: {gpu_util_delta}%, Mem used: Δ{gpu_mem_delta:.2f} MB")
        total_gpu_util += gpu_after["gpu_util_percent"]
        total_gpu_mem += gpu_mem_delta
        total_cpu_util += cpu_after["cpu_util_percent"]
        total_cpu_mem += cpu_mem_delta

        evaluated += 1
        if gt_answer.lower() in answer.lower():
            correct += 1
        print(f"[{i}] Q: {question}")
        print(f"    GT: {gt_answer.lower()}")
        print(f"    →  Predicted: {answer.lower()}")
        print(f"[{i}] Accuracy so far: {correct}/{evaluated} = {correct / evaluated:.2f}")
        # print("GPU Before: ", gpu_before)
        # print("GPU After: ", gpu_after)
        # print("GPU Delta: ", gpu_util_delta)
        # print("GPU Total: ", total_gpu_util)

    total_inference_time = time.time() - start_time
    # current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = correct / len(data) if data else 0
    avg_gpu_util = total_gpu_util / len(data)
    avg_gpu_mem = total_gpu_mem / len(data)
    avg_cpu_util = total_cpu_util / len(data)
    avg_cpu_mem = total_cpu_mem / len(data)

    # print(total_gpu_util, avg_gpu_util)
    # print(total_cpu_util, avg_cpu_util)

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "ttft_ms": 0,
        "avg_latency_ms": (total_latency / len(data)) * 1000,
        "throughput_tps": total_generated_tokens / total_inference_time,
        "accuracy": accuracy,
        "total_time": total_inference_time,
        "cpu_mem": avg_cpu_mem,
        "cpu_util": avg_cpu_util,
        "gpu_util": avg_gpu_util,
        "gpu_mem": avg_gpu_mem,
        "num_samples": len(data),
    }

# ============ Main ============
def main():
    
    args = parse_args()
    print(args)
    
    handle, device_name = init_nvml()
    model_family, model_variant, weights, input_shape = model_config[args.model_id]
    model, load_duration_ms, mem_usage_mb, file_size_mb = load_model(model_variant, weights, input_shape, handle)
    gflops, mparams = get_flops(model, input_shape)
    

    metrics = {
        "model_family": model_family,
        "model_variant": model_variant,
        "model_input_shape": input_shape,  # (batch_size, channels, height, width)
        "device": device_name,
        "onnx_file_size_mb": file_size_mb,  # MB (ONNX)
        "onnx_load_duration_ms": load_duration_ms,  # ms
        "onnx_gpu_load_memory_mb": mem_usage_mb,  # MB (GPU)
        "onnx_gflops": gflops,  # GFLOPs
        "onnx_mparams": mparams,  # Millions of parameters
    }

    csv_path = args.output_file
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()