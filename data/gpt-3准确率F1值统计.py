import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os


def calculate_metrics(file_path):
    # 根据文件扩展名选择读取方式
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=None)
        elif file_ext == '.csv':
            # 尝试自动检测分隔符
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    # 常见分隔符检测
                    if ';' in first_line:
                        sep = ';'
                    elif '\t' in first_line:
                        sep = '\t'
                    elif '|' in first_line:
                        sep = '|'
                    else:
                        sep = ','  # 默认逗号分隔
            except:
                sep = ','  # 出错时使用默认逗号分隔

            df = pd.read_csv(file_path, header=None, sep=sep, encoding='utf-8', engine='python')
        else:
            print(f"❌ 不支持的文件格式: {file_ext}")
            return
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return

    # 检查必要的列是否存在
    required_columns = {2: '真实标签', 3: '预测标签'}
    if len(df.columns) < 4:
        print(f"❌ 文件列数不足: 需要至少4列，实际只有{len(df.columns)}列")
        return

    # 重命名列以便处理
    column_mapping = {}
    for idx, name in required_columns.items():
        if idx < len(df.columns):
            column_mapping[df.columns[idx]] = name
        else:
            print(f"❌ 缺少第{idx + 1}列")
            return

    df = df.rename(columns=column_mapping)

    # 过滤无效行
    initial_count = len(df)
    df = df.dropna(subset=['真实标签', '预测标签'])
    df = df[df['真实标签'].notna() & df['预测标签'].notna()]

    if len(df) == 0:
        print("❌ 没有有效数据可用于计算")
        return

    # 统计被过滤的行数
    filtered_count = initial_count - len(df)

    # 转换为字符串类型以确保一致性
    df['真实标签'] = df['真实标签'].astype(str).str.strip().str.lower()
    df['预测标签'] = df['预测标签'].astype(str).str.strip().str.lower()

    # 获取所有真实标签的类别
    true_labels = df['真实标签'].unique()

    # 识别不在真实标签类别中的预测
    invalid_mask = ~df['预测标签'].isin(true_labels)
    invalid_count = invalid_mask.sum()

    # 处理无效预测：设为特殊值"invalid"
    df.loc[invalid_mask, '预测标签'] = 'invalid'

    # 计算准确率（只考虑有效预测）
    valid_mask = ~invalid_mask
    valid_accuracy = accuracy_score(
        df.loc[valid_mask, '真实标签'],
        df.loc[valid_mask, '预测标签']
    ) if valid_mask.any() else 0

    # 计算整体准确率（包括无效预测）
    overall_accuracy = accuracy_score(
        df['真实标签'],
        df['预测标签'].where(~invalid_mask, 'invalid')
    )

    # 计算F1分数（宏平均和微平均）
    # 只考虑真实标签中存在的类别
    labels = [label for label in true_labels if label != 'invalid']

    macro_f1 = f1_score(
        df['真实标签'],
        df['预测标签'],
        average='macro',
        labels=labels,
        zero_division=0
    )

    micro_f1 = f1_score(
        df['真实标签'],
        df['预测标签'],
        average='micro',
        labels=labels,
        zero_division=0
    )

    # 生成分类报告
    report = classification_report(
        df['真实标签'],
        df['预测标签'],
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    # 打印结果
    print("\n" + "=" * 50)
    print(f"📊 数据集分析: {file_path}")
    print("=" * 50)
    print(f"📝 总样本数: {initial_count}")
    print(f"🚫 被过滤的无效样本: {filtered_count}")
    print(f"✅ 有效样本数: {len(df)}")
    print(f"⚠️ 预测标签不在真实类别中的样本数: {invalid_count}")
    print("\n" + "-" * 50)
    print(f"🎯 准确率 (仅有效预测): {valid_accuracy:.4f}")
    print(f"🎯 整体准确率 (包含无效预测): {overall_accuracy:.4f}")
    print(f"🎯 宏平均F1分数: {macro_f1:.4f}")
    print(f"🎯 微平均F1分数: {micro_f1:.4f}")
    print("\n" + "-" * 50)
    print("📈 分类报告:")
    print(classification_report(
        df['真实标签'],
        df['预测标签'],
        labels=labels,
        zero_division=0
    ))

    # 保存详细结果
    result_df = pd.DataFrame({
        '真实标签': df['真实标签'],
        '预测标签': df['预测标签'],
        '是否正确': df['真实标签'] == df['预测标签'],
        '是否有效预测': ~invalid_mask
    })

    # 根据输入文件类型决定输出格式
    output_file = file_path.replace(file_ext, '_analysis.xlsx')
    result_df.to_excel(output_file, index=False)
    print(f"\n💾 详细分析结果已保存到: {output_file}")

    return {
        'total_samples': initial_count,
        'filtered_samples': filtered_count,
        'valid_samples': len(df),
        'invalid_predictions': invalid_count,
        'valid_accuracy': valid_accuracy,
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'classification_report': report
    }


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = "gpt35_test_predictions.csv"

    # 计算指标
    metrics = calculate_metrics(file_path)

    # 如果需要，可以在这里访问具体的指标值
    if metrics:
        print("\n" + "=" * 50)
        print(f"宏平均F1分数: {metrics['macro_f1']:.4f}")
        print(f"整体准确率: {metrics['overall_accuracy']:.4f}")