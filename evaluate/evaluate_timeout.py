import os
import json
import argparse

def analyze_catattack(folder_path):
    output_file = os.path.join(folder_path, 'timeout_statistics.txt')
    
    with open(output_file, 'w') as out_f:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    out_f.write(f"file: {filename} \n")
                    
                    total_entries = len(data)
                    entry_overthink_count = 0
                    total_results = 0
                    result_overthink_count = 0
                    
                    for item in data:
                        avg_inference_time = item.get('avg_inference_time', 0)
                        if avg_inference_time >= 180:
                            entry_overthink_count += 1
                        
                        results = item.get('results', [])
                        total_results += len(results)
                        
                        for result in results:
                            inference_time = result.get('inference_time', 0)
                            if inference_time >= 180:
                                result_overthink_count += 1
                    
                    out_f.write(f"total_entries: {total_entries}\n")
                    out_f.write(f"entry_overthink_count: {entry_overthink_count}\n")
                    out_f.write(f"entry_timeout_rate: {entry_overthink_count/total_entries:.2%}\n\n")
                    
                    out_f.write(f"total_results: {total_results}\n")
                    out_f.write(f"result_overthink_count: {result_overthink_count}\n")
                    
                    if total_results > 0:
                        out_f.write(f"resulty_timeout_rate: {result_overthink_count/total_results:.2%}\n")
                    else:
                        out_f.write("resulty_timeout_rate: N/A \n")
                    
                    out_f.write("\n" + "=" * 50 + "\n\n")
                    
                except Exception as e:
                    out_f.write(f"Error processing file {filename}: {str(e)}\n")
                    out_f.write("-" * 50 + "\n\n")

def analyze_overthinking_attack(folder_path):
    output_file = os.path.join(folder_path, 'timeout_statistics.txt')
    
    with open(output_file, 'w') as out_f:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    out_f.write(f"file: {filename}\n")
                    
                    total_entries = len(data)
                    entry_overthink_count = 0
                    
                    for item in data:
                        inference_time = item.get('inference_time', 0)
                        if inference_time >= 180:
                            entry_overthink_count += 1
                    
                    out_f.write(f"total_entries: {total_entries}\n")
                    out_f.write(f"entry_overthink_count: {entry_overthink_count}\n")
                    
                    if total_entries > 0:
                        overthink_rate = entry_overthink_count / total_entries
                        out_f.write(f"timeout_rate: {overthink_rate:.2%}\n")
                    else:
                        out_f.write("timeout_rate: N/A \n")
                    
                    out_f.write("\n" + "=" * 50 + "\n\n")
                    
                except Exception as e:
                    out_f.write(f"Error processing file {filename}: {str(e)}\n")
                    out_f.write("-" * 50 + "\n\n")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", default="catattack", choices=["catattack", "overthinking_attack"])
    args = parser.parse_args()

    folder_path = '../result/result_'+args.attack_type
    if args.attack_type == 'catattack':
        analyze_catattack(folder_path)
    elif args.attack_type == 'overthinking_attack':
        analyze_overthinking_attack(folder_path)
    print(f"Results saved to {folder_path}/timeout_statistics.txt")