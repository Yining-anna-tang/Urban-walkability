import pandas as pd

# 原始数据路径
input_path = r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv'
output_path = r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/2分类Y（Y2=心理专注）.csv'

df = pd.read_csv(input_path, encoding="GBK")

# 二分类：1,2 -> 0 ; 3,4,5 -> 1
df['Y_binary'] = df['Y'].apply(lambda x: 1 if x >= 3 else 0)

# 保存新文件
df.to_csv(output_path, index=False, encoding="GBK")

print("✅ 二分类完成：1/2=0 , 3/4/5=1")
print("📂 文件保存到：", output_path)
print(df['Y_binary'].value_counts())
