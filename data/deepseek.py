import openai
import pandas as pd
import numpy as np
import time
import os
import csv
import requests
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from requests.exceptions import RequestException

# ✅ DeepSeek 配置
DEEPSEEK_API_BASE = "https://api.probex.top/v1"  # DeepSeek API地址
DEEPSEEK_API_KEY = ""  # 替换为你的DeepSeek API密钥

# 配置参数
BATCH_SIZE = 500  # 每批次处理的数据量
MAX_RETRIES = 5  # 最大重试次数
INITIAL_DELAY = 1  # 初始重试延迟(秒)
BACKOFF_FACTOR = 2  # 指数退避因子
RESULT_CSV = "deepseek_test_predictions.csv"  # 使用CSV格式保存结果
LOG_FILE = "processing_log.txt"
PROGRESS_FILE = "progress_checkpoint.txt"
MODEL_NAME = "deepseek-v3"  # DeepSeek模型名称

# 加载数据
train_df = pd.read_excel("train-new.xlsx", header=None, names=["text", "label"])
test_df = pd.read_excel("test-new.xlsx", header=None, names=["text", "label"])

# 添加索引列，便于跟踪进度
test_df = test_df.reset_index(drop=False).rename(columns={'index': 'orig_index'})


# 构造few-shot示例，自动适应不同的数据集标签
def create_few_shot_messages(train_df, num_samples_per_label=10):
    # 获取数据集中的所有标签
    unique_labels = train_df['label'].unique()

    # 构造系统提示
    system_prompt = (
            "请对以下短文本进行分类，类别包括："
            + ", ".join([str(label) for label in unique_labels]) + "。\n"
                                                                   "注意：请仅返回类别标签，不要包含其他字符或解释。\n"
    )

    # 构造 few-shot 示例
    few_shot_messages = [{"role": "system", "content": system_prompt}]

    # 为每个标签选择若干个示例
    examples = train_df.groupby("label").apply(
        lambda x: x.sample(n=num_samples_per_label, random_state=42)).reset_index(drop=True)

    for _, row in examples.iterrows():
        few_shot_messages.append({"role": "user", "content": row['text']})
        few_shot_messages.append({"role": "assistant", "content": str(row['label'])})  # 确保标签是字符串

    return few_shot_messages


# 使用上面定义的函数构造消息
few_shot_messages = create_few_shot_messages(train_df, num_samples_per_label=10)


# DeepSeek API调用函数（带重试机制）
def deepseek_api_call(messages):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 10,
        "stream": False
    }

    try:
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()  # 如果请求失败则抛出异常
        return response.json()
    except RequestException as e:
        raise Exception(f"API请求失败: {str(e)}")


# 增强的分类函数（带重试机制）
def classify_with_retry(text, few_shot_messages):
    messages = few_shot_messages + [{"role": "user", "content": text}]

    for attempt in range(MAX_RETRIES):
        try:
            # 使用DeepSeek API
            response = deepseek_api_call(messages)

            # 获取回复内容并清理
            reply = response['choices'][0]['message']['content'].strip()
            return reply

        except Exception as e:
            error_msg = str(e)

            # 检查是否为敏感内容错误
            if "敏感内容" in error_msg or "不安全" in error_msg or "safety" in error_msg.lower():
                print(f"\n🚫 Content safety violation for record: {error_msg[:150]}")
                return "violation_error"

            # 检查是否为速率限制错误
            elif "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                delay = INITIAL_DELAY * (BACKOFF_FACTOR ** attempt)
                print(f"\n⚠️ Rate limit exceeded (Attempt {attempt + 1}/{MAX_RETRIES}): {error_msg[:150]}...")
                print(f"🕒 Retrying in {delay} seconds...")
                time.sleep(delay)

            # 其他可重试错误
            elif any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "server", "api"]):
                delay = INITIAL_DELAY * (BACKOFF_FACTOR ** attempt)
                print(f"\n⚠️ Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg[:150]}...")
                print(f"🕒 Retrying in {delay} seconds...")
                time.sleep(delay)

            # 不可恢复的错误
            else:
                print(f"\n❌ Unrecoverable error: {error_msg[:150]}")
                return "error"

    print(f"🚨 Request failed after {MAX_RETRIES} attempts")
    return "error"


# 初始化结果CSV文件
def init_csv_file():
    if not os.path.exists(RESULT_CSV):
        try:
            with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['orig_index', 'text', 'label', 'pred', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"✅ Created new result file: {RESULT_CSV}")
        except PermissionError as e:
            print(f"❌ Permission denied when creating file: {e}")
            print("⚠️ Please close any programs that may be using this file and restart the script")
            exit(1)
    else:
        print(f"✅ Using existing result file: {RESULT_CSV}")


# 保存单条结果到CSV
def save_single_result(orig_index, text, label, pred):
    max_retries = 5  # 最大重试次数
    retry_delay = 1  # 重试延迟(秒)

    for attempt in range(max_retries):
        try:
            # 尝试打开文件并写入
            with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['orig_index', 'text', 'label', 'pred', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'orig_index': orig_index,
                    'text': text,
                    'label': label,
                    'pred': pred,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
            return  # 写入成功，退出函数

        except PermissionError as e:
            # 权限错误处理
            if attempt < max_retries - 1:
                print(
                    f"⚠️ Permission denied when saving record {orig_index}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print(f"❌ Failed to save record {orig_index} after {max_retries} attempts: {e}")
                # 记录到错误日志
                with open("save_errors.log", "a") as error_log:
                    error_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to save record {orig_index}: {e}\n")

        except Exception as e:
            print(f"❌ Unexpected error when saving record {orig_index}: {e}")
            # 记录到错误日志
            with open("save_errors.log", "a") as error_log:
                error_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error saving record {orig_index}: {e}\n")
            break  # 非权限错误，立即退出


# 获取已处理的数据索引
def get_processed_indices():
    processed_indices = set()
    if os.path.exists(RESULT_CSV):
        try:
            with open(RESULT_CSV, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['orig_index'] and row['orig_index'].isdigit():
                        processed_indices.add(int(row['orig_index']))
        except Exception as e:
            print(f"⚠️ Error reading result file: {e}")
    return processed_indices


# 保存进度检查点
def save_progress_checkpoint(last_index):
    try:
        with open(PROGRESS_FILE, 'w') as f:
            f.write(str(last_index))
    except Exception as e:
        print(f"⚠️ Error saving progress checkpoint: {e}")


# 加载进度检查点
def load_progress_checkpoint():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


# 初始化结果文件
init_csv_file()

# 获取已处理的数据索引
processed_indices = get_processed_indices()
print(f"ℹ️ Found {len(processed_indices)} processed records")

# 加载进度检查点
start_index = load_progress_checkpoint()
print(f"ℹ️ Starting from index: {start_index}")

# 分批处理数据
total_samples = len(test_df)
num_batches = int(np.ceil(total_samples / BATCH_SIZE))

print(f"\n🚀 Starting processing of {total_samples} records")
print(f"📦 Batch size: {BATCH_SIZE}, Total batches: {num_batches}")
print(f"🧠 Using model: {MODEL_NAME}")

# 创建总进度条
pbar_total = tqdm(total=total_samples, desc="Overall Progress", position=0)

# 设置已处理的进度
pbar_total.update(len(processed_indices))

for batch_idx in range(int(np.ceil(start_index / BATCH_SIZE)), num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min((batch_idx + 1) * BATCH_SIZE, total_samples)

    # 创建批次进度条
    pbar_batch = tqdm(total=batch_end - batch_start, desc=f"Batch {batch_idx + 1}/{num_batches}", position=1)

    print(f"\n🔧 Processing batch {batch_idx + 1}/{num_batches} (records {batch_start}-{batch_end - 1})")

    # 处理当前批次
    for idx in range(batch_start, batch_end):
        # 跳过已处理的记录
        if idx in processed_indices:
            pbar_batch.update(1)
            pbar_total.update(1)
            continue

        row = test_df.iloc[idx]
        orig_index = row['orig_index']
        text = row['text']
        label = row['label']

        # 进行分类
        pred = classify_with_retry(text, few_shot_messages)

        # 如果预测是"error"或"violation_error"，保存为"error"
        if pred in ["error", "violation_error"]:
            print(f"⚠️ Sensitive content or error detected for record {orig_index}. Saving 'error'.")
            pred = "error"

        # 保存结果
        save_single_result(orig_index, text, label, pred)

        # 更新进度
        pbar_batch.update(1)
        pbar_total.update(1)

        # 更新检查点（每10条保存一次）
        if idx % 10 == 0:
            save_progress_checkpoint(idx)

    # 关闭批次进度条
    pbar_batch.close()

    # 保存批次结束检查点
    save_progress_checkpoint(batch_end)

    # 记录进度
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"Batch {batch_idx + 1}: Processed {batch_start}-{batch_end - 1} at {time.ctime()}\n")
    except Exception as e:
        print(f"⚠️ Error writing to log file: {e}")

    print(f"💾 Saved results for batch {batch_idx + 1}")

# 关闭总进度条
pbar_total.close()

# 删除进度检查点文件
if os.path.exists(PROGRESS_FILE):
    try:
        os.remove(PROGRESS_FILE)
        print("✅ Removed progress checkpoint file")
    except Exception as e:
        print(f"⚠️ Error removing progress file: {e}")

# 最终评估
print("\n📊 All data processed, starting evaluation...")

# 加载完整结果
try:
    result_df = pd.read_csv(RESULT_CSV)
except FileNotFoundError:
    print("❌ Result file not found. Evaluation skipped.")
    exit(1)

# 过滤掉错误结果
valid_df = result_df[(result_df['pred'] != 'error') & (result_df['pred'] != 'violation_error')]
error_count = (result_df['pred'] == 'error').sum()
violation_count = (result_df['pred'] == 'violation_error').sum()

if len(valid_df) > 0:
    # 确保所有标签都是字符串类型
    y_true = valid_df["label"].astype(str).str.strip().str.lower()
    y_pred = valid_df["pred"].astype(str).str.strip().str.lower()

    # 创建标签映射（确保数字标签和字符串标签能正确匹配）
    unique_labels = sorted(set(y_true) | set(y_pred))

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\n🎯 Final Evaluation Results:")
    print(f"✅ Valid samples: {len(valid_df)}/{len(result_df)}")
    print(f"❌ Error/failed samples: {error_count}")
    print(f"🚫 Sensitive content violations: {violation_count}")
    print("🎯 Accuracy:", acc)
    print("🎯 Macro F1 Score:", f1)
    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_labels))

    # 保存最终结果到Excel
    try:
        result_file = f"{MODEL_NAME.replace('-', '_')}_predictions_final.xlsx"
        result_df.to_excel(result_file, index=False)
        print(f"\n✅ Final results saved to {result_file}")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
else:
    print("❌ No valid results available for evaluation")

# 保存详细报告
detailed_report = f"{MODEL_NAME.replace('-', '_')}_classification_report.txt"
try:
    with open(detailed_report, 'w') as f:
        f.write(f"Dataset size: {len(result_df)}\n")
        f.write(f"Valid samples: {len(valid_df)}\n")
        f.write(f"Error samples: {error_count}\n")
        f.write(f"Sensitive content violations: {violation_count}\n\n")
        if len(valid_df) > 0:
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, labels=unique_labels))
    print(f"📝 Detailed report saved to {detailed_report}")
except Exception as e:
    print(f"❌ Error saving detailed report: {e}")

print("\n✅ Processing complete!")