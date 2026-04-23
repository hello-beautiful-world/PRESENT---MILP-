# PRESENT MILP Differential Characteristic Search

本项目为本科毕业设计实验代码，课题：**《不同MILP编码策略对PRESENT差分特征搜索效率的影响研究》**

---

## 📌项目简介
本项目实现了 **PRESENT-80 轻量分组密码**的差分特征自动搜索工具，通过三种不同的 MILP（混合整数线性规划）编码策略，对比它们在求解效率、变量数、约束数上的差异。

## ✨ 核心功能
- 实现三种 MILP 编码策略：
  1.  DDT 直接编码
  2.  凸包不等式编码（H-representation）
  3.  CDP 条件差分传播编码
- 支持 4~8 轮 PRESENT-80 的最小活跃 S 盒数搜索
- 自动统计并输出：变量数、约束数、求解时间、最小活跃 S 盒数

## 🛠️ 运行环境
- Python 3.7+
- Gurobi 9.1+
- `gurobipy` Python 包

## 🚀 运行方式
```bash
# 1. 单策略运行示例
python present_milp_search.py --strategy ddt --rounds 4
python present_milp_search.py --strategy hull --rounds 5
python present_milp_search.py --strategy cdp  --rounds 6

# 2. 批量运行所有策略
python present_milp_search.py --all
