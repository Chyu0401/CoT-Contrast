import os
import json
import torch
import logging
import time
from typing import Dict, List, Tuple
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["DASHSCOPE_API_KEY"] = "sk-096d9d2e3aac472aad1d3c0d34c3f29f"
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

API_CALL_DELAY = 1.0  
MAX_RETRIES = 3    

def load_path_data(file_path: str) -> Dict:
    logger.info(f"Loading path data: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            path_data = json.load(f)
    else:
        path_data = torch.load(file_path, weights_only=False)
    
    return path_data

def load_dataset_data(dataset_name: str) -> Dict:
    dataset_file = f"datasets/{dataset_name}.pt"
    if not os.path.exists(dataset_file):
        logger.error(f"Dataset file not found: {dataset_file}")
        return {}
    
    logger.info(f"Loading dataset: {dataset_file}")
    dataset_data = torch.load(dataset_file, weights_only=False)
    
    if 'raw_texts' not in dataset_data:
        logger.error(f"Dataset {dataset_name} does not contain 'raw_texts' key")
        return {}
    
    logger.info(f"Successfully loaded dataset {dataset_name} with {len(dataset_data['raw_texts'])} nodes")
    return dataset_data

def save_json(file_name: str, data: Dict):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def api_call_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=800,  
                temperature=0.7   
            )
            response = completion.choices[0].message.content
            time.sleep(API_CALL_DELAY)
            return response.strip()
        except Exception as e:
            logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(API_CALL_DELAY * (attempt + 1))
            else:
                raise e

def generate_path_explanation_parallel(path_info: Tuple, raw_texts: List[str], diversity_scores: List[float]) -> Dict:
    """并行处理单个路径解释的函数"""
    index, path_data = path_info
    
    try:
        if isinstance(path_data, dict):
            anchor_id = path_data.get('anchor_id')
            target_id = path_data.get('target_node')  
            path = path_data.get('path', [])
            diversity_score = diversity_scores[index] if index < len(diversity_scores) else 0.0
            labels = path_data.get('labels', None)  
        else:
            if len(path_data) == 4:
                anchor_id, target_id, path, diversity_score = path_data
                labels = None
            elif len(path_data) == 5:
                anchor_id, target_id, path, diversity_score, label = path_data
                labels = [label] if label is not None else None
            else:
                logger.warning(f"Unexpected path data format: {len(path_data)} elements")
                return {"error": "Invalid path data format"}
        
        try:
            anchor_id = int(anchor_id)
            target_id = int(target_id)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid ID type in path: anchor_id={anchor_id}, target_id={target_id}, error: {e}")
            return {"error": f"Invalid ID type: {e}"}
        
        path_text = " → ".join(map(str, path))
        
        path_node_texts = []
        for node_id in path:
            try:
                node_id_int = int(node_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid node_id type: {node_id}, type: {type(node_id)}")
                path_node_texts.append(f"Node {node_id}: [Invalid node ID]")
                continue
                
            if 0 <= node_id_int < len(raw_texts):
                node_text = raw_texts[node_id_int]
                if len(node_text) > 200:
                    node_text = node_text[:200] + "..."
                path_node_texts.append(f"Node {node_id_int}: {node_text}")
            else:
                path_node_texts.append(f"Node {node_id_int}: [Text not available]")
        
        complete_path_prompt = f"""\
You will serve as an assistant to help me analyze a graph path and provide a structured explanation of why this path exists. I will provide you with information about the nodes in the path.

Path: {path_text}
This path goes from node {anchor_id} to node {target_id}.

Node information:
{chr(10).join(path_node_texts)}

Requirements:
1. Please provide your response in the strict JSON format, following this structure:
{{"path_description": "Briefly describe what this path represents in one sentence", "reasoning": "Explain your reasoning process for why this path exists step by step, showing the logical progression from one node to the next"}}

2. There are no word limits for the "reasoning".
3. Ensure that the "path_description" should be a concise summary of the overall progressionis and no longer than 100 words.
4. Do not provide any other information outside the JSON string.
5. Focus only on content in the actual text and avoid making false associations.
6. The reasoning should explain each hop in sequence, showing why each transition makes sense.
"""
        
        response = api_call_with_retry(complete_path_prompt)
        
        try:
            # 清理响应，移除可能的markdown代码块标记
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            explanation_data = json.loads(cleaned_response)
            path_description = explanation_data.get('path_description', '')
            reasoning = explanation_data.get('reasoning', '')
            
            # 组合explanation（详细解释）
            explanation = reasoning
            
            # 组合summary（总结：只包含路径描述）
            summary = path_description
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}, using raw response")
            # 如果JSON解析失败，手动拆分原始响应
            full_explanation = response.strip()
            
            sentences = full_explanation.split('. ')
            if len(sentences) >= 2:
                # 第一句作为路径描述
                path_description = sentences[0] + '.'
                # 中间部分作为详细解释
                explanation = '. '.join(sentences[1:]) + '.'
                summary = path_description
            else:
                # 如果无法拆分，使用原始响应
                explanation = full_explanation
                summary = f"Error: Unable to generate structured explanation for path {path_text}"
        
        result = {
            "path": path,
            "diversity_score": diversity_score,
            "explanation": explanation,
            "summary": summary
        }
        
        if labels is not None:
            result['labels'] = labels
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating explanation for path {index}: {str(e)}")
        return {
            "error": f"Error generating explanation: {str(e)}",
            "path": path if 'path' in locals() else [],
            "diversity_score": diversity_score if 'diversity_score' in locals() else 0.0,
            "explanation": f"Error: Unable to generate explanation for path",
            "summary": f"Error: Unable to generate summary for path"
        }

def process_paths_for_dataset_parallel(dataset_name: str, diversity_type: str, max_workers: int = 8) -> List[Dict]:
    dataset_data = load_dataset_data(dataset_name)
    if not dataset_data:
        logger.error(f"Failed to load dataset {dataset_name}")
        return []
    
    raw_texts = dataset_data['raw_texts']
    
    path_file_json = f"path_selection/{dataset_name}_{diversity_type}_diversity.json"
    path_file_pt = f"path_selection/{dataset_name}_{diversity_type}_diversity.pt"
    
    if os.path.exists(path_file_json):
        path_file = path_file_json
    elif os.path.exists(path_file_pt):
        path_file = path_file_pt
    else:
        logger.error(f"Path file not found: {path_file_json} or {path_file_pt}")
        return []
    
    path_data = load_path_data(path_file)
    
    if 'paths' in path_data:
        paths = path_data['paths']
        diversity_scores = path_data.get('diversity_scores', [])
    else:
        logger.error(f"Path data does not contain 'paths' key")
        return []
    
    # 检查是否存在已处理的文件
    output_file = f"explanation/{dataset_name}_{diversity_type}_diversity_explanations.json"
    os.makedirs("explanation", exist_ok=True)
    
    if os.path.exists(output_file):
        logger.info(f"[恢复] 检测到未完成文件: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        processed_paths = existing_data.get('results', [])
        processed_count = len(processed_paths)
        logger.info(f"已处理 {processed_count} 条路径，继续处理剩余 {len(paths) - processed_count} 条")
    else:
        logger.info(f"[开始] 新文件处理: {path_file}")
        processed_paths = []
        processed_count = 0
    
    # 准备剩余需要处理的路径
    remaining_paths = paths[processed_count:]
    remaining_diversity_scores = diversity_scores[processed_count:] if len(diversity_scores) > processed_count else []
    
    logger.info(f"Processing {len(remaining_paths)} {diversity_type} diversity paths for {dataset_name} with {max_workers} workers")
    
    # 准备任务数据
    tasks = [(processed_count + i, path_info) for i, path_info in enumerate(remaining_paths)]
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建future到索引的映射
        future_to_index = {executor.submit(generate_path_explanation_parallel, task, raw_texts, remaining_diversity_scores): task[0] 
                          for task in tasks}
        
        indexed_results = {}
        
        # 使用tqdm显示进度条，处理完成的任务
        for future in tqdm(as_completed(future_to_index), 
                          total=len(tasks), 
                          desc=f'Processing {dataset_name} {diversity_type} paths'):
            index = future_to_index[future]
            
            try:
                result = future.result()
                indexed_results[index] = result
            except Exception as exc:
                logger.error(f'Path at index {index} generated an exception: {exc}')
                indexed_results[index] = {
                    "error": f"Exception: {exc}",
                    "path": [],
                    "diversity_score": 0.0,
                    "explanation": f"Error: Exception occurred during processing",
                    "summary": f"Error: Exception occurred during processing"
                }
    
    # 按原始顺序重组结果
    for i in range(len(tasks)):
        current_index = processed_count + i
        if current_index in indexed_results:
            processed_paths.append(indexed_results[current_index])
        else:
            # 处理缺失的情况
            processed_paths.append({
                "error": "Missing result",
                "path": [],
                "diversity_score": 0.0,
                "explanation": "Error: Missing result",
                "summary": "Error: Missing result"
            })
    
    final_data = {
        "dataset": dataset_name,
        "diversity_type": diversity_type,
        "total_paths": len(paths),
        "processed_count": len(processed_paths),
        "results": processed_paths
    }
    save_json(output_file, final_data)
    logger.info(f"[完成] {dataset_name} {diversity_type} 处理完毕，有效处理 {len(processed_paths)}/{len(paths)} 条")
    
    return processed_paths

def main():
    # 只处理指定的三份文件
    datasets_and_types = [
        ("cora", "low"),
        ("citeseer", "low"), 
        ("citeseer", "high")
    ]
    max_workers = 8  # 可以根据需要调整线程数
    
    os.makedirs("explanation", exist_ok=True)
    
    logger.info("Starting parallel path explanation generation")
    logger.info(f"API call delay: {API_CALL_DELAY}s, Max retries: {MAX_RETRIES}, Max workers: {max_workers}")
    logger.info(f"Processing files: cora_low_diversity.json, citeseer_low_diversity.json, citeseer_high_diversity.json")
    
    total_start_time = time.time()
    
    for dataset, diversity_type in datasets_and_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {dataset} {diversity_type} diversity paths")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            results = process_paths_for_dataset_parallel(dataset, diversity_type, max_workers)
            logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
                
        except Exception as e:
            logger.error(f"Error processing {dataset} {diversity_type}: {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"All parallel path explanations generation completed!")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Generated files:")
    logger.info(f"  - explanation/cora_low_diversity_explanations.json")
    logger.info(f"  - explanation/citeseer_low_diversity_explanations.json") 
    logger.info(f"  - explanation/citeseer_high_diversity_explanations.json")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
