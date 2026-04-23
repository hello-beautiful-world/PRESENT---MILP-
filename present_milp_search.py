# -*- coding: utf-8 -*-
"""
PRESENT-80 差分特征自动搜索实验代码
================================================
本代码实现三种不同的 MILP 编码策略,用于搜索 PRESENT-80 在
4-8 轮下的最小差分活跃 S 盒数目,并给出求解时间、变量数量、
约束数量三个指标的对比结果。

三种编码策略:
  Strategy 1: DDT 直接编码 (Direct DDT Encoding)
              通过引入 0-1 指示变量, 枚举差分分布表的每一个非零
              条目, 给出最严格但规模最大的约束。
  Strategy 2: 传统凸包不等式编码 (Convex Hull H-representation)
              将 S 盒的所有可能差分模式视作 8 维空间中的点集,
              计算其凸包, 并用贪心算法从凸包不等式中挑选若干条
              作为约束; 这是 Sun 等人提出的主流编码方式。
              为便于实验复现, 本实现预先给出已挑选好的若干条
              不等式, 具体见 CONVEX_HULL_INEQUALITIES 列表。
  Strategy 3: CDP 条件差分传播编码 (Conditional Differential
              Propagation) 基于 PRESENT S 盒的 undisturbed bits
              等条件差分性质, 用少量线性不等式实现剪枝。

使用方法:
  python present_milp_search.py --strategy ddt --rounds 4
  python present_milp_search.py --strategy hull --rounds 5
  python present_milp_search.py --strategy cdp  --rounds 6
  python present_milp_search.py --all

需要的第三方库:
  gurobipy (Gurobi 官方 Python 接口, 须自行安装 Gurobi 并激活许可证)
"""

import argparse                       #解析命令行参数,通过 --strategy、--rounds 等参数控制实验。
import time                           #记录模型构建时间和求解时间
from typing import List, Tuple, Dict  #类型标注，列表、元组、字典

try:
    import gurobipy as gp              #导入 Gurobi的 Python 接口
    from gurobipy import GRB           #GRB提供常量,如GRB.BINARY、GRB.MINIMIZE、GRB.OPTIMAL
except ImportError:         
    print("[警告] 未检测到 gurobipy, 请先安装 Gurobi 并配置许可证")
    raise


# =========================================================
# 1. PRESENT-80 常量:S盒、置换层、轮数
# =========================================================
# 4x4 S盒
PRESENT_SBOX = [
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
]

# 置换层: 第i位被搬到位置 P_LAYER[i]，例如新一轮的x[P_LAYER[i]]<——上一轮x[i]
P_LAYER = [
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
    4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
    8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
    12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
]

N_BITS = 64          # 分组长度
N_SBOX = 16          # 每轮S盒数量
SBOX_SIZE = 4        # S盒输入长度


# =========================================================
# 2. 差分分布表 (DDT) 的计算
# =========================================================
def compute_ddt(sbox: List[int]) -> List[List[int]]:
    """构造 4x4 S 盒的差分分布表, 返回 16x16 的二维列表"""
    n = len(sbox)
    ddt = [[0] * n for _ in range(n)]
    for x in range(n):                      #遍历所有明文x
        for dx in range(n):                 #遍历所有输入差分dx
            dy = sbox[x] ^ sbox[x ^ dx]     #计算输出差分dy
            ddt[dx][dy] += 1                #统计每个(dx,dy)对出现的次数
    return ddt                              #返回16×16的二维列表


def enumerate_valid_transitions(ddt: List[List[int]]) -> List[Tuple[int, int]]:
    """从DDT中筛选出所有合法的非零差分转移对(dx,dy)"""
    transitions = []                            #用来存放所有合法非零差分转移对(dx,dy)
    for dx in range(1, 16):                     #从1开始循坏，dx=0表示相同明文差分为0，无研究意义。
        for dy in range(16):                    #present的S盒双射，dx≠0⇒dy≠0，dy可从1开始循环，保证代码通用也可从0开始。
            if ddt[dx][dy] > 0:
                transitions.append((dx, dy))
    return transitions


# =========================================================
# 3. 凸包方法预先挑选出的不等式
# -----------------------------------------------------------
#这些不等式来源于: 对PRESENT S盒所有可能差分模式，(共97个8维0-1向量)计算凸包的H表示,再用贪心算法。
#(详见文献 Sun et al., ASIACRYPT 2014) 从中挑选出若干条。
#凸包策略使用的 21 条线性不等式，每行 9 个数，前 8 个是系数（对应输入差分 4 位 + 输出差分 4 位），最后 1 个是常数项。
#每一行形如 (c0,c1,c2,c3,c4,c5,c6,c7, c),对应约束: c0*x0 + c1*x1 + ... + c7*y3 + c >= 0
#其中 (x0,x1,x2,x3) 为 S 盒输入差分的 4 比特,(y0,y1,y2,y3) 为 S 盒输出差分的 4 比特。
#这 21 条不等式可覆盖所有不可能差分模式。
# =========================================================
CONVEX_HULL_INEQUALITIES = [
    (-2,  1,  1,  3,  1, -1,  1,  2,  0),
    ( 1, -2, -3, -2,  1, -4,  3, -3, 10),
    ( 2, -2,  3, -4, -1, -4, -4,  1, 11),
    (-1, -2, -2, -1, -1,  2, -1,  0,  6),
    (-2,  1, -2, -1,  1, -1, -2,  0,  6),
    ( 2,  1,  1, -3,  1,  2,  1,  2,  0),
    (-1,  1,  1,  1,  0,  0,  0,  1,  0),
    ( 1,  1,  1, -1,  0,  0,  0,  1,  0),
    ( 0,  0,  0,  1,  1,  1,  1, -1,  0),
    ( 0,  0,  0,  1,  1, -1,  1,  1,  0),
    ( 0,  0,  0, -1,  1, -1,  1, -1,  2),
    (-1,  1,  1, -1,  0,  0,  0, -1,  2),
    ( 1, -1,  1, -1,  1,  0,  1,  0,  0),
    ( 1,  1,  1,  1,  0,  1,  0,  1,  0),
    ( 0,  1,  0,  1, -1,  0, -1,  0,  1),
    ( 1,  0,  1,  0, -1, -1,  0,  0,  1),
    ( 0,  1,  1,  0,  1,  1,  1,  0,  0),
    ( 1,  1,  0,  1,  0,  1,  1,  1,  0),
    ( 1,  0,  1,  1,  1,  1,  0,  1,  0),
    ( 0,  0,  1,  1, -1,  0,  1, -1,  1),
    ( 1,  1, -1,  0, -1,  1,  0,  0,  1),
]


# =========================================================
# 4. CDP(条件差分传播)不等式
# -----------------------------------------------------------
# PRESENT S盒的undisturbed bits性质（6条）:（某些特定输入差分模式下，输出的某些比特位是确定的）
#   1001 -> ***0(输入为 0x9, 则输出最低位恒为 0)
#   0001 -> ***1(输入为 0x1, 则输出最低位恒为 1)
#   1000 -> ***1(输入为 0x8, 则输出最低位恒为 1)
#   ***1 -> 0001(输出为 0x1, 则输入最低位恒为 1)
#   ***1 -> 0100(输出为 0x4, 则输入最低位恒为 1)
#   ***0 -> 0101(输出为 0x5, 则输入最低位恒为 0)
# 按 Sun et al. Inscrypt 2013 的推导, 每条规则可写成一条线性不等式。
# 这里 (x0,x1,x2,x3)中x3为LSB（最低有效位）,(y0,y1,y2,y3)中y3为LSB。
# =========================================================
CDP_INEQUALITIES = [
    (-1,  1,  1, -1,  0,  0,  0, -1,  2),   # 1001 -> ***0
    ( 1,  1,  1, -1,  0,  0,  0,  1,  0),   # 0001 -> ***1
    (-1,  1,  1,  1,  0,  0,  0,  1,  0),   # 1000 -> ***1
    ( 0,  0,  0,  1,  1,  1,  1, -1,  0),   # ***1 -> 0001
    ( 0,  0,  0,  1,  1, -1,  1,  1,  0),   # ***1 -> 0100
    ( 0,  0,  0, -1,  1, -1,  1, -1,  2),   # ***0 -> 0101
]


# =========================================================
# 5. S盒基础约束（三种策略共用）
# -----------------------------------------------------------
# 基础约束用于刻画非零输入 <=> 非零输出和活跃S盒指示约束
#   1. A = 1  当且仅当输入差分非零
#   2. 输入非零则输出非零 (S 盒为双射)
# =========================================================
def add_sbox_basic_constraints(model, x_in, y_out, A_var):
    """添加 S 盒的基础结构性约束 (所有策略共用)"""
    #约束1，A_var=1，当且仅当输入差分非零，即x_in有非零位
    for xi in x_in:
        model.addConstr(xi <= A_var)            #每个输入比特都不超过活跃指示变量A，即若A=0则所有输入比特必须为0。
    model.addConstr(gp.quicksum(x_in) >=A_var)  #输入比特之和≥A，即若A=1则至少有一个输入比特为1。

    # 约束2，输入输出同为零或同为非零 (双射)
    model.addConstr(4 * gp.quicksum(y_out) - gp.quicksum(x_in) >= 0)
    model.addConstr(4 * gp.quicksum(x_in) - gp.quicksum(y_out) >= 0)


# =========================================================
# 6. 策略 1: DDT 直接编码
# -----------------------------------------------------------
# 核心思想: 对每个S盒引入若干0-1指示变量t_{dx, dy},
# 其中(dx, dy)遍历DDT中所有非零转移。
# 约束要求"所有t变量之和等于1" (如果S盒活跃),并把 t 变量与输入/输出比特绑定。
# 这种编码最贴近 DDT 的真实刻画, 但变量数明显多于其他两种。
# =========================================================
def build_model_ddt(rounds: int):
    """使用 DDT 直接编码构建 rounds 轮 PRESENT-80 差分搜索模型"""
    model = gp.Model("PRESENT_DDT")                             #创建名为PRESENT_DDT的Gurobi模型
    model.Params.OutputFlag = 0                                 #关闭 Gurobi 求解器的日志输出

    ddt = compute_ddt(PRESENT_SBOX)                             #计算S盒的DDT
    transitions = enumerate_valid_transitions(ddt)              #从DDT表里筛选出所有合法的非零差分转移

    # 创建状态比特变量: x[r][i]表示第r轮输入的第i比特差分
    x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{i}")      #这是循环里的函数，下面
          for i in range(N_BITS)] for r in range(rounds + 1)]    #！！！注意要+1，多出的一轮用于存储最后一轮S盒的输出

    A_vars = []  # 活跃 S 盒指示变量, 用于构造目标函数
    for r in range(rounds):
        for s in range(N_SBOX):
            # 取出第 r 轮第 s 个 S 盒的输入 4 比特
            xin = [x[r][4 * s + b] for b in range(SBOX_SIZE)]
            # S 盒输出 4 比特 (临时变量, 随后经 pLayer 得到下一轮输入)
            yout = [model.addVar(vtype=GRB.BINARY, name=f"y_{r}_{s}_{b}")
                    for b in range(SBOX_SIZE)]
            A = model.addVar(vtype=GRB.BINARY, name=f"A_{r}_{s}")#为每个S盒创建1个活跃指示变量 A，A=1表示该S盒被激活（输入差分非零）。
            A_vars.append(A)

            add_sbox_basic_constraints(model, xin, yout, A)#添加S盒的基础约束

            # DDT 直接编码的核心: 为每个合法转移引入0-1指示变量t，t=1表示该S盒选择了这个具体的差分转移
            t_vars = []
            for (dx, dy) in transitions:
                t = model.addVar(vtype=GRB.BINARY,
                                 name=f"t_{r}_{s}_{dx}_{dy}")
                t_vars.append((t, dx, dy))

            # 如果 S 盒活跃, 则恰好选中一个合法转移。所有t变量之和等于A，若S盒活跃（A=1），则恰好选中一个合法转移；若不活跃（A=0），则所有t均为 0。
            model.addConstr(gp.quicksum(t for (t, _, _) in t_vars) == A)

            # 把 t 变量与具体的输入输出比特值绑定
            # 规则: 若 t_{dx,dy} = 1, 则 (xin, yout) 必须与 (dx, dy) 一致
            for bit in range(SBOX_SIZE):
                # 第 bit 位 (从高位 bit=0 到低位 bit=3)
                coeff_x = gp.LinExpr()
                coeff_y = gp.LinExpr()
                for (t, dx, dy) in t_vars:
                    xb = (dx >> (SBOX_SIZE - 1 - bit)) & 1
                    yb = (dy >> (SBOX_SIZE - 1 - bit)) & 1
                    if xb == 1:
                        coeff_x.add(t)
                    if yb == 1:
                        coeff_y.add(t)
                # 若活跃, 第 bit 位的差分 = 对应 t 变量之和
                model.addConstr(xin[bit] == coeff_x)
                model.addConstr(yout[bit] == coeff_y)

            # 经 pLayer 更新至下一轮输入
            # bit在当前轮的全局位置: 4*s + bit
            # pLayer 把位置 i 搬到 P[i]
            for bit in range(SBOX_SIZE):
                src_pos = 4 * s + bit
                dst_pos = P_LAYER[src_pos]
                model.addConstr(x[r + 1][dst_pos] == yout[bit])

    # 目标函数: 最小化活跃S盒总数
    model.setObjective(gp.quicksum(A_vars), GRB.MINIMIZE)
    # 排除平凡解，要求第0轮至少有一个输入差分位为1，排除全零的平凡解。！！！注意！
    model.addConstr(gp.quicksum(x[0]) >= 1)

    return model, A_vars


# =========================================================
# 7. 策略 2: 凸包不等式编码
# -----------------------------------------------------------
# 该编码直接用预先挑选好的若干条线性不等式描述 S 盒的差分行为,
# 不引入额外的指示变量 t。
# =========================================================
def add_inequalities(model, xin, yout, ineqs):
    """将一组形如 (c0..c7, c) 的不等式施加到 xin, yout 上"""
    for row in ineqs:
        c = row[:-1]            #前8个系数
        rhs = -row[-1]          #注意右端常数取负！
        model.addConstr(
            c[0] * xin[0] + c[1] * xin[1] + c[2] * xin[2] + c[3] * xin[3] +
            c[4] * yout[0] + c[5] * yout[1] + c[6] * yout[2] + c[7] * yout[3]
            >= rhs
        )


def build_model_hull(rounds: int):
    """使用凸包不等式编码构建 rounds 轮模型"""
    model = gp.Model("PRESENT_HULL")
    model.Params.OutputFlag = 0

    x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{i}")
          for i in range(N_BITS)] for r in range(rounds + 1)]

    A_vars = []
    for r in range(rounds):
        for s in range(N_SBOX):
            xin = [x[r][4 * s + b] for b in range(SBOX_SIZE)]
            yout = [model.addVar(vtype=GRB.BINARY, name=f"y_{r}_{s}_{b}")
                    for b in range(SBOX_SIZE)]
            A = model.addVar(vtype=GRB.BINARY, name=f"A_{r}_{s}")
            A_vars.append(A)

            add_sbox_basic_constraints(model, xin, yout, A)
            add_inequalities(model, xin, yout, CONVEX_HULL_INEQUALITIES)

            for bit in range(SBOX_SIZE):
                src_pos = 4 * s + bit
                dst_pos = P_LAYER[src_pos]
                model.addConstr(x[r + 1][dst_pos] == yout[bit])

    model.setObjective(gp.quicksum(A_vars), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x[0]) >= 1)
    return model, A_vars


# =========================================================
# 8. 策略 3:CDP条件差分传播编码
# -----------------------------------------------------------
# 仅利用6条从undisturbed bits推导出的CDP不等式进行剪枝。
# 由于不等式数量极少, 得到的可行域会比凸包法稍宽, 但模型规模是最小的, 因此对于轮数较高的实例往往更具效率优势。
# =========================================================
def build_model_cdp(rounds: int):
    """使用 CDP 条件差分传播编码构建 rounds 轮模型"""
    model = gp.Model("PRESENT_CDP")
    model.Params.OutputFlag = 0

    x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{i}")
          for i in range(N_BITS)] for r in range(rounds + 1)]

    A_vars = []
    for r in range(rounds):
        for s in range(N_SBOX):
            xin = [x[r][4 * s + b] for b in range(SBOX_SIZE)]
            yout = [model.addVar(vtype=GRB.BINARY, name=f"y_{r}_{s}_{b}")
                    for b in range(SBOX_SIZE)]
            A = model.addVar(vtype=GRB.BINARY, name=f"A_{r}_{s}")
            A_vars.append(A)

            add_sbox_basic_constraints(model, xin, yout, A)
            add_inequalities(model, xin, yout, CDP_INEQUALITIES)

            for bit in range(SBOX_SIZE):
                src_pos = 4 * s + bit
                dst_pos = P_LAYER[src_pos]
                model.addConstr(x[r + 1][dst_pos] == yout[bit])

    model.setObjective(gp.quicksum(A_vars), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x[0]) >= 1)
    return model, A_vars


# =========================================================
# 9. 主求解流程: 统计求解时间、变量数、约束数、最小活跃 S 盒
# =========================================================
STRATEGY_BUILDERS = {
    "ddt":  build_model_ddt,
    "hull": build_model_hull,
    "cdp":  build_model_cdp,
}

STRATEGY_NAMES = {
    "ddt":  "DDT 直接编码",
    "hull": "凸包不等式编码",
    "cdp":  "CDP 条件差分传播编码",
}


def solve_single(strategy: str, rounds: int, time_limit: int = 3600) -> Dict:
    """对指定策略、指定轮数执行一次求解并返回统计信息"""
    builder = STRATEGY_BUILDERS[strategy]
    t0 = time.time()
    model, A_vars = builder(rounds)#构建模型
    build_time = time.time() - t0

    # Gurobi 设置: 允许设置求解时间上限, 输出关闭
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0

    model.update()                  # ！！加这一行，强制刷新模型缓冲区，是必须的，因为 Gurobi 采用懒更新机制，addVar/addConstr 的结果不会立即反映到 NumVars/NumConstrs，必须显式刷新。
    n_vars = model.NumVars          #读取变量数
    n_constrs = model.NumConstrs    #读取约束数

    t0 = time.time()
    model.optimize()                #执行求解
    solve_time = time.time() - t0

    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        active = int(round(model.ObjVal)) if model.SolCount > 0 else -1
    else:
        active = -1

    return {
        "strategy":   strategy,
        "rounds":     rounds,
        "n_vars":     n_vars,
        "n_constrs":  n_constrs,
        "build_time": build_time,
        "solve_time": solve_time,
        "active":     active,
        "status":     model.Status,
    }


def print_row(r: Dict):
    """格式化打印一行结果"""
    print(f"  策略: {STRATEGY_NAMES[r['strategy']]:<20s} "
          f"轮数: {r['rounds']:>2d}  "
          f"变量: {r['n_vars']:>6d}  "
          f"约束: {r['n_constrs']:>6d}  "
          f"最小活跃 S 盒: {r['active']:>3d}  "
          f"求解时间: {r['solve_time']:>8.3f} s")


def run_all(rounds_range=(4, 5, 6, 7, 8)):
    """对三种策略、4-8 轮 PRESENT-80 全部跑一遍, 打印对比表"""
    print("=" * 80)
    print("PRESENT-80 三种 MILP 编码策略对比实验")
    print("=" * 80)
    all_results = []
    for strategy in ("ddt", "hull", "cdp"):
        print(f"\n--- 策略: {STRATEGY_NAMES[strategy]} ---")
        for R in rounds_range:
            res = solve_single(strategy, R)
            print_row(res)
            all_results.append(res)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="PRESENT-80 三种 MILP 编码策略的差分特征搜索实验")
    parser.add_argument("--strategy", choices=["ddt", "hull", "cdp"],
                        default=None,
                        help="指定单次实验的编码策略")
    parser.add_argument("--rounds", type=int, default=None,
                        help="指定单次实验的轮数 (4-8)")
    parser.add_argument("--all", action="store_true",
                        help="跑完所有策略 x 所有轮数")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="单次求解的时间上限, 秒 (默认 3600)")
    args = parser.parse_args()

    if args.all or args.strategy is None:
        run_all()
    else:
        res = solve_single(args.strategy, args.rounds, args.time_limit)
        print_row(res)


if __name__ == "__main__":
    main()
