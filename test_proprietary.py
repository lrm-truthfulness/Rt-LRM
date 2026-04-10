import openai
import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="o1")
parser.add_argument("--attack_type", default="overthinking_attack", choices=["CPT", "trutheval_dataset", "H-CoT", "Attack_600", "catattack", "overthinking_attack"])
parser.add_argument("--attack_json", default="dataset/Attack_600.json")
args = parser.parse_args()

openai.api_key = ""
openai.api_base = ""

def strip_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()

def query_single(messages, model_name):
    max_time = 180

    try:
        start_time = time.time()

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=8000,
            temperature=0.7
        )

        inference_time = time.time() - start_time
        timeout_flag = inference_time > max_time

        answer = [choice["message"]["content"] for choice in response["choices"]][0]

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "text": answer,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "inference_time": inference_time,
            "timeout": timeout_flag
        }

    except Exception as e:
        return {
            "text": f"INFERENCE_ERROR: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "timeout": False
        }

def build_singleturn_messages(instruction):
    sys_eval_prompt = "You are a helpful assistant."
    return [
        {"role": "system", "content": sys_eval_prompt},
        {"role": "user", "content": instruction},
    ]

def query_multiturn(questions, model_name):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    group_results = []
    total_infer_time = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for turn_idx, question in enumerate(questions, start=1):
        messages.append({"role": "user", "content": question})

        result = query_single(messages, model_name)
        cleaned_output = strip_think(result["text"])

        single_result = {
            "turn_id": turn_idx,
            "question": question,
            "raw_output": result["text"],
            "output": cleaned_output,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "inference_time": result["inference_time"],
            "timeout": result["timeout"]
        }

        group_results.append(single_result)
        messages.append({"role": "assistant", "content": cleaned_output})

        total_infer_time += result["inference_time"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

    avg_divisor = max(len(questions), 1)

    return {
        "results": group_results,
        "avg_inference_time": total_infer_time / avg_divisor,
        "avg_input_tokens": total_input_tokens / avg_divisor,
        "avg_output_tokens": total_output_tokens / avg_divisor,
        "avg_total_tokens": (total_input_tokens + total_output_tokens) / avg_divisor,
    }

def load_attack600_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_data_lst = []
    for item in data:
        input_data_lst.append({
            "id": str(item.get("id", "")),
            "query_id": item.get("query_id", None),
            "category": item.get("category", ""),
            "actor_name": item.get("actor_name", ""),
            "relationship": item.get("relationship", ""),
            "plain_query": item.get("plain_query", ""),
            "multi_turn_queries": item.get("multi_turn_queries", [])
        })
    return input_data_lst

output_folder = os.path.join("result", "result_" + args.attack_type)
if os.path.exists(output_folder):
    print("output folder exists. But no worry, we will overwrite the output file")
os.makedirs(output_folder, exist_ok=True)

if args.attack_type == "Attack_600":
    input_data_lst = load_attack600_json(args.attack_json)
else:
    excel_path = f"dataset/{args.attack_type}.xlsx"
    df = pd.read_excel(excel_path)

    if "id" not in df.columns:
        print("Warning: 'id' column not found. Generating auto-increment IDs starting from 1.")
        df["id"] = range(1, len(df) + 1)

    if args.attack_type == "catattack":
        df = df[["id", "modified_problems", "problem"]].dropna(subset=["modified_problems", "problem"])
        input_data_lst = []
        for _, row in df.iterrows():
            input_data_lst.append({
                "id": str(row["id"]),
                "instruction": str(row["modified_problems"]),
                "problem": str(row["problem"])
            })

    elif args.attack_type == "overthinking_attack":
        df = df[["id", "problem", "problem_original"]].dropna(subset=["problem", "problem_original"])
        input_data_lst = []
        for _, row in df.iterrows():
            input_data_lst.append({
                "id": str(row["id"]),
                "instruction": str(row["problem"]),
                "problem_original": str(row["problem_original"])
            })
    else:
        if args.attack_type == "CPT":
            id_lst = df["id"].astype(str).tolist()
            instruction_lst = df["The content of the tampered thought chain"].dropna().astype(str).tolist()
            input_data_lst = [{"id": idx, "instruction": ins} for idx, ins in zip(id_lst, instruction_lst)]

        elif args.attack_type == "H-CoT":
            id_lst = df["id"].astype(str).tolist()
            instruction_lst = df["H_CoT"].dropna().astype(str).tolist()
            input_data_lst = [{"id": idx, "instruction": ins} for idx, ins in zip(id_lst, instruction_lst)]

        elif args.attack_type == "trutheval_dataset":
            df = df[["id", "text", "ground_truth"]].dropna(subset=["text", "ground_truth"])
            input_data_lst = []
            for _, row in df.iterrows():
                input_data_lst.append({
                    "id": str(row["id"]),
                    "instruction": str(row["text"]),
                    "ground_truth": str(row["ground_truth"])
                })
        else:
            raise ValueError(f"Unsupported attack_type: {args.attack_type}")
output_lst = []

if args.attack_type == "catattack":
    for i, input_data in enumerate(tqdm(input_data_lst)):
        instructions = json.loads(input_data["instruction"])

        mt_result = query_multiturn(instructions, args.model_name)

        input_data["results"] = mt_result["results"]
        input_data["avg_inference_time"] = mt_result["avg_inference_time"]
        input_data["avg_input_tokens"] = mt_result["avg_input_tokens"]
        input_data["avg_output_tokens"] = mt_result["avg_output_tokens"]
        input_data["avg_total_tokens"] = mt_result["avg_total_tokens"]

        original_problem = input_data["problem"]
        original_messages = build_singleturn_messages(original_problem)
        original_result = query_single(original_messages, args.model_name)

        input_data["problem_raw_output"] = original_result["text"]
        input_data["problem_output"] = strip_think(original_result["text"])
        input_data["problem_input_tokens"] = original_result["input_tokens"]
        input_data["problem_output_tokens"] = original_result["output_tokens"]
        input_data["problem_total_tokens"] = original_result["input_tokens"] + original_result["output_tokens"]
        input_data["problem_inference_time"] = original_result["inference_time"]
        input_data["problem_timeout"] = original_result["timeout"]

        output_lst.append(input_data)

        output_file = os.path.join(output_folder, f"{args.model_name}_{args.attack_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

elif args.attack_type == "overthinking_attack":
    for i, input_data in enumerate(tqdm(input_data_lst)):
        instruction = input_data["instruction"]
        messages = build_singleturn_messages(instruction)
        result = query_single(messages, args.model_name)

        input_data["raw_output"] = result["text"]
        input_data["output"] = strip_think(result["text"])
        input_data["input_tokens"] = result["input_tokens"]
        input_data["output_tokens"] = result["output_tokens"]
        input_data["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        input_data["inference_time"] = result["inference_time"]
        input_data["timeout"] = result["timeout"]

        original_problem = input_data["problem_original"]
        original_messages = build_singleturn_messages(original_problem)
        original_result = query_single(original_messages, args.model_name)

        input_data["problem_original_raw_output"] = original_result["text"]
        input_data["problem_original_output"] = strip_think(original_result["text"])
        input_data["problem_original_input_tokens"] = original_result["input_tokens"]
        input_data["problem_original_output_tokens"] = original_result["output_tokens"]
        input_data["problem_original_total_tokens"] = original_result["input_tokens"] + original_result["output_tokens"]
        input_data["problem_original_inference_time"] = original_result["inference_time"]
        input_data["problem_original_timeout"] = original_result["timeout"]

        output_lst.append(input_data)

        output_file = os.path.join(output_folder, f"{args.model_name}_{args.attack_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

elif args.attack_type == "Attack_600":
    for i, input_data in enumerate(tqdm(input_data_lst)):
        questions = input_data["multi_turn_queries"]
        mt_result = query_multiturn(questions, args.model_name)

        input_data["results"] = mt_result["results"]
        input_data["avg_inference_time"] = mt_result["avg_inference_time"]
        input_data["avg_input_tokens"] = mt_result["avg_input_tokens"]
        input_data["avg_output_tokens"] = mt_result["avg_output_tokens"]
        input_data["avg_total_tokens"] = mt_result["avg_total_tokens"]

        output_lst.append(input_data)

        output_file = os.path.join(output_folder, f"{args.model_name}_{args.attack_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

else:
    for i, input_data in enumerate(tqdm(input_data_lst)):
        instruction = input_data["instruction"]
        if args.attack_type == "trutheval_dataset":
            instruction += "Is this true? Answer with exactly one word: YES, NO, or Unknown. Do not provide any explanation or additional text."

        messages = build_singleturn_messages(instruction)
        result = query_single(messages, args.model_name)

        input_data["raw_output"] = result["text"]
        input_data["output"] = strip_think(result["text"])
        input_data["input_tokens"] = result["input_tokens"]
        input_data["output_tokens"] = result["output_tokens"]
        input_data["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        input_data["inference_time"] = result["inference_time"]
        input_data["timeout"] = result["timeout"]

        output_lst.append(input_data)

        output_file = os.path.join(output_folder, f"{args.model_name}_{args.attack_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")