"""Research implementation adapted from 标准列生成.py. See docs/provenance.md."""

import gurobipy as grb
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Union


class ColumnGenerationOptimizer:
    """标准列生成算法优化器"""

    def __init__(self, roll_width: int, time_limit: int = 300):
        """
        初始化

        参数:
            roll_width: 母卷宽度(mm)
            time_limit: MIP求解时间限制(秒)
        """
        self.roll_width = roll_width
        self.time_limit = time_limit

    def solve(self, width_demands: Union[Dict[int, int], str]) -> Dict:
        """
        求解切割问题

        参数:
            width_demands: {宽度: 需求量} 或 "730:60, 880:155" 格式

        返回:
            优化结果字典
        """
        # 解析输入
        if isinstance(width_demands, str):
            width_demands = self._parse_input(width_demands)

        widths = np.array(sorted(width_demands.keys()))
        demands = np.array([width_demands[w] for w in widths])

        print(f"\n{'=' * 80}")
        print(f"一维切割优化 - 标准列生成算法")
        print(f"{'=' * 80}")
        print(f"母卷宽度: {self.roll_width}mm")
        print(f"产品种类: {len(widths)}")
        print(f"总需求: {np.sum(demands)}")
        print(f"需求明细: {dict(zip(widths, demands))}")
        print(f"{'=' * 80}\n")

        # 列生成求解
        patterns, iterations = self._column_generation(widths, demands)

        # 求解整数MIP
        result = self._solve_integer_mip(patterns, widths, demands, iterations)

        return result

    def _parse_input(self, input_str: str) -> Dict[int, int]:
        """解析字符串输入 "730:60, 880:155" """
        width_demands = {}
        for pair in input_str.split(','):
            if ':' in pair:
                width, demand = pair.strip().split(':')
                width_demands[int(width)] = int(demand)
        return width_demands

    def _column_generation(self, widths: np.ndarray, demands: np.ndarray) -> Tuple[List[np.ndarray], int]:
        """
        列生成主循环

        返回:
            (切割模式列表, 迭代次数)
        """
        n_items = len(widths)

        # 初始化：单位矩阵（每个产品一个模式，每次只切1个）
        patterns = []
        for i in range(n_items):
            pattern = np.zeros(n_items, dtype=int)
            pattern[i] = 1  # 只切1个！
            patterns.append(pattern)

        print("开始列生成迭代...")
        iteration = 0

        while iteration < 1000:
            iteration += 1

            # 求解线性松弛主问题
            dual_prices = self._solve_master_lp(patterns, widths, demands)

            # 求解背包子问题
            new_pattern, reduced_cost = self._solve_knapsack(widths, dual_prices)

            # 检查收敛
            if reduced_cost >= -1e-6:
                print(f"迭代 {iteration}: 收敛 (归约成本={reduced_cost:.6f})")
                break

            # 添加新模式
            patterns.append(new_pattern)

            if iteration % 5 == 0 or iteration <= 3:
                print(f"迭代 {iteration}: 归约成本={reduced_cost:.6f}, 模式总数={len(patterns)}")

        print(f"\n列生成完成: {iteration}次迭代, 生成{len(patterns)}个模式\n")

        return patterns, iteration

    def _solve_master_lp(self, patterns: List[np.ndarray],
                         widths: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """
        求解线性松弛主问题，返回对偶价格

        主问题:
            min  Σ x_j
            s.t. Σ a_ij * x_j >= d_i  (对偶变量: π_i)
                 x_j >= 0
        """
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("MasterLP")
        model.setParam('OutputFlag', 0)
        model.setParam('Method', 1)  # 对偶单纯形法

        # 变量：每个模式的使用次数
        x = model.addVars(n_patterns, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        # 目标：最小化母卷数
        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        # 约束：满足需求
        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i],
                name=f"demand_{i}"
            )

        model.optimize()

        # 提取对偶价格
        dual_prices = np.array([model.getConstrByName(f"demand_{i}").Pi
                                for i in range(n_items)])

        return dual_prices

    def _solve_knapsack(self, widths: np.ndarray, dual_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        求解背包子问题，生成新的切割模式

        子问题:
            max  Σ π_i * a_i
            s.t. Σ w_i * a_i <= W
                 a_i >= 0, integer

        归约成本 = 1 - max(Σ π_i * a_i)

        返回:
            (新模式, 归约成本)
        """
        n_items = len(widths)

        model = grb.Model("Knapsack")
        model.setParam('OutputFlag', 0)

        # 变量：每种产品切多少个
        a = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, name="a")

        # 目标：最大化对偶价格加权和
        model.setObjective(
            grb.quicksum(dual_prices[i] * a[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )

        # 约束：不超过母卷宽度
        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n_items)) <= self.roll_width,
            name="width"
        )

        model.optimize()

        # 提取解
        new_pattern = np.array([a[i].X for i in range(n_items)], dtype=int)
        reduced_cost = 1.0 - model.ObjVal  # 注意：是 1 - obj，不是 obj - 1

        return new_pattern, reduced_cost

    def _solve_integer_mip(self, patterns: List[np.ndarray], widths: np.ndarray,
                           demands: np.ndarray, iterations: int) -> Dict:
        """
        求解整数MIP，得到最终切割方案

        整数主问题:
            min  Σ x_j
            s.t. Σ a_ij * x_j >= d_i
                 x_j >= 0, integer
        """
        print("求解整数MIP...")
        mip_start = time.time()

        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("MasterMIP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)

        # 变量：整数
        x = model.addVars(n_patterns, lb=0, vtype=GRB.INTEGER, name="x")

        # 目标
        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        # 约束
        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i],
                name=f"demand_{i}"
            )

        model.optimize()

        mip_time = time.time() - mip_start

        # 提取解
        optimal_rolls = int(round(model.ObjVal))

        used_patterns = []
        used_counts = []

        for j in range(n_patterns):
            if x[j].X > 0.5:
                used_patterns.append(patterns[j])
                used_counts.append(int(round(x[j].X)))

        # 计算统计
        total_waste = 0
        pattern_details = []

        for pattern, count in zip(used_patterns, used_counts):
            pattern_width = np.sum(pattern * widths)
            waste = self.roll_width - pattern_width

            total_waste += waste * count

            # 构建描述
            desc_parts = []
            for i, qty in enumerate(pattern):
                if qty > 0:
                    desc_parts.append(f"{qty}×{widths[i]}mm")

            pattern_details.append({
                '切割方案': ' + '.join(desc_parts),
                '使用次数': count,
                '切割宽度': int(pattern_width),
                '剩余宽度': int(waste),
                '纸损率': f"{waste / self.roll_width * 100:.2f}%"
            })

        total_waste_rate = total_waste / (optimal_rolls * self.roll_width) if optimal_rolls > 0 else 0

        # 构建结果
        result = {
            'optimal_rolls': optimal_rolls,
            'n_patterns_used': len(used_patterns),
            'n_patterns_generated': n_patterns,
            'iterations': iterations,
            'total_waste': int(total_waste),
            'waste_rate': total_waste_rate,
            'mip_time': mip_time,
            'pattern_details': pattern_details
        }

        # 打印结果
        self._print_result(result, widths, demands)

        return result

    def _print_result(self, result: Dict, widths: np.ndarray, demands: np.ndarray):
        """打印结果"""
        print(f"{'=' * 80}")
        print(f"优化结果")
        print(f"{'=' * 80}")
        print(f"最优母卷数: {result['optimal_rolls']}")
        print(f"使用模式数: {result['n_patterns_used']} (生成了{result['n_patterns_generated']}个)")
        print(f"总纸损: {result['total_waste']}mm ({result['waste_rate']:.2%})")
        print(f"MIP求解时间: {result['mip_time']:.2f}秒")
        print(f"\n{'-' * 80}")
        print(f"{'编号':<6} {'切割方案':<45} {'次数':<8} {'切割宽':<10} {'剩余':<10} {'纸损率':<10}")
        print(f"{'-' * 80}")

        for i, detail in enumerate(result['pattern_details'], 1):
            print(f"#{i:<5} {detail['切割方案']:<45} {detail['使用次数']:<8} "
                  f"{detail['切割宽度']:<10} {detail['剩余宽度']:<10} {detail['纸损率']:<10}")

        print(f"{'-' * 80}")
        print(f"总计: {result['optimal_rolls']} 母卷, 纸损 {result['total_waste']}mm ({result['waste_rate']:.2%})")
        print(f"{'=' * 80}\n")

    def save_report(self, result: Dict, filename: str = 'cutting_plan.xlsx'):
        """保存结果到Excel"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 摘要
                summary = pd.DataFrame([{
                    '最优母卷数': result['optimal_rolls'],
                    '使用模式数': result['n_patterns_used'],
                    '生成模式数': result['n_patterns_generated'],
                    '迭代次数': result['iterations'],
                    '总纸损(mm)': result['total_waste'],
                    '纸损率(%)': result['waste_rate'] * 100,
                    'MIP时间(秒)': result['mip_time']
                }])
                summary.to_excel(writer, sheet_name='摘要', index=False)

                # 切割方案
                details = pd.DataFrame(result['pattern_details'])
                details.to_excel(writer, sheet_name='切割方案', index=False)

            print(f"报告已保存: {filename}\n")
        except Exception as e:
            print(f"保存失败: {e}\n")


# ============================================================================
# 使用示例
# ============================================================================

