import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 加载数据
data = pd.read_csv('runs/models/UT-1/final_joint_no_rel_att_s1/summary.csv')

base_path = 'figure'
os.makedirs(base_path)
# 提取不同模型的评估指标数据
models = data['fold']
accuracy = data['accuracy']
recall = data['recall_m']
precision = data['precision_m']
f1 = data['f1_m']

# 设置图表的大小
plt.figure(figsize=(12, 8))

# 绘制柱状图
plt.subplot(2, 2, 1)
plt.bar(models, accuracy, color='b', alpha=0.7)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy by Fold')

plt.subplot(2, 2, 2)
plt.bar(models, recall, color='g', alpha=0.7)
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.title('Recall by Fold')

plt.subplot(2, 2, 3)
plt.bar(models, precision, color='r', alpha=0.7)
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.title('Precision by Fold')

plt.subplot(2, 2, 4)
plt.bar(models, f1, color='orange', alpha=0.7)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('F1 Score by Fold')

plt.tight_layout()
save_path = base_path + '/' + "F1_score.png"
plt.savefig(save_path)
plt.show()

# 绘制折线图
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(models, accuracy, marker='o', color='b')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy by Fold')

plt.subplot(2, 2, 2)
plt.plot(models, recall, marker='o', color='g')
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.title('Recall by Fold')

plt.subplot(2, 2, 3)
plt.plot(models, precision, marker='o', color='r')
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.title('Precision by Fold')

plt.subplot(2, 2, 4)
plt.plot(models, f1, marker='o', color='orange')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('F1 Score by Fold')

plt.tight_layout()
save_path = base_path + '/' + "/pr.png"
plt.savefig(save_path)
plt.show()

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(data[['loss', 'accuracy', 'recall_m', 'precision_m', 'f1_m']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Evaluation Metrics')
save_path = base_path + '/' + "cm_matrix.png"
plt.savefig(save_path)
plt.show()
