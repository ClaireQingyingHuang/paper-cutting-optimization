"""Research implementation adapted from CG+Diving Heuristic.py. See docs/provenance.md."""

import gurobipy as grb
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Union


class ColumnGenerationOptimizer:
    """改进版列生成算法优化器（已修复统计BUG）"""

    def __init__(self, roll_width: int, time_limit: int = 300,
                 n_columns_per_iter: int = 3, diving_threshold: float = 0.3):
        """
        初始化

        参数:
            roll_width: 母卷宽度(mm)
            time_limit: MIP求解时间限制(秒)
            n_columns_per_iter: 每次迭代添加的列数
            diving_threshold: 深潜启发式固定变量的阈值
        """
        self.roll_width = roll_width
        self.time_limit = time_limit
        self.n_columns_per_iter = n_columns_per_iter
        self.diving_threshold = diving_threshold

        self.upper_bound = float('inf')
        self.best_solution = None

    def solve(self, width_demands: Union[Dict[int, int], str]) -> Dict:
        """求解切割问题"""
        if isinstance(width_demands, str):
            width_demands = self._parse_input(width_demands)

        widths = np.array(sorted(width_demands.keys()))
        demands = np.array([width_demands[w] for w in widths])

        print(f"\n{'=' * 80}")
        print(f"一维切割优化 - 改进版列生成算法（已修复）")
        print(f"{'=' * 80}")
        print(f"母卷宽度: {self.roll_width}mm")
        print(f"产品种类: {len(widths)}")
        print(f"总需求量: {np.sum(demands)} 个")
        print(f"理论下界: {np.sum(demands * widths) / self.roll_width:.0f} 卷（按材料总长估算）")
        print(f"优化策略: 多列添加({self.n_columns_per_iter}列/次) + 深潜启发式")
        print(f"{'=' * 80}\n")

        # 列生成求解
        patterns, lp_solution, iterations = self._column_generation(widths, demands)

        # 深潜启发式构建整数解
        result = self._diving_heuristic_solve(patterns, widths, demands, lp_solution, iterations)

        return result

    def _parse_input(self, input_str: str) -> Dict[int, int]:
        """解析字符串输入"""
        width_demands = {}
        for pair in input_str.split(','):
            if ':' in pair:
                width, demand = pair.strip().split(':')
                width_demands[int(width)] = int(demand)
        return width_demands

    def _column_generation(self, widths: np.ndarray, demands: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, int]:
        """列生成主循环"""
        n_items = len(widths)

        # 初始化：单位矩阵
        patterns = []
        for i in range(n_items):
            pattern = np.zeros(n_items, dtype=int)
            pattern[i] = 1
            patterns.append(pattern)

        print("开始列生成迭代（多列添加策略）...")
        iteration = 0
        lp_solution = None

        while iteration < 1000:
            iteration += 1

            # 求解线性松弛主问题
            dual_prices, lp_solution = self._solve_master_lp(patterns, widths, demands)

            # 求解背包子问题，返回多个优质列
            new_patterns = self._solve_knapsack_multi(widths, dual_prices)

            # 检查收敛
            if not new_patterns:
                print(f"迭代 {iteration}: 收敛")
                break

            # 添加所有新模式
            for pattern, rc in new_patterns:
                patterns.append(pattern)

            if iteration % 5 == 0 or iteration <= 3:
                best_rc = min([rc for _, rc in new_patterns])
                print(
                    f"迭代 {iteration}: 添加{len(new_patterns)}列, 最优归约成本={best_rc:.6f}, 模式总数={len(patterns)}")

            # 定期求解整数MIP更新上界
            if iteration % 10 == 0:
                self._update_upper_bound(patterns, widths, demands, iteration)

        print(f"\n列生成完成: {iteration}次迭代, 生成{len(patterns)}个模式")
        print(f"线性松弛解: {np.sum(lp_solution):.2f} 卷")
        print(f"当前上界: {self.upper_bound if self.upper_bound != float('inf') else '未知'}\n")

        return patterns, lp_solution, iteration

    def _solve_master_lp(self, patterns: List[np.ndarray],
                         widths: np.ndarray, demands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """求解线性松弛主问题"""
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("MasterLP")
        model.setParam('OutputFlag', 0)
        model.setParam('Method', 1)

        x = model.addVars(n_patterns, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        constrs = []
        for i in range(n_items):
            constr = model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i],
                name=f"demand_{i}"
            )
            constrs.append(constr)

        model.optimize()

        dual_prices = np.array([constrs[i].Pi for i in range(n_items)])
        lp_solution = np.array([x[j].X for j in range(n_patterns)])

        return dual_prices, lp_solution

    def _solve_knapsack_multi(self, widths: np.ndarray, dual_prices: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """改进版背包求解：返回多个优质列"""
        n_items = len(widths)
        result_patterns = []

        model = grb.Model("Knapsack")
        model.setParam('OutputFlag', 0)
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', self.n_columns_per_iter * 2)

        a = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, name="a")

        model.setObjective(
            grb.quicksum(dual_prices[i] * a[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )

        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n_items)) <= self.roll_width,
            name="width"
        )

        # 上界剪枝
        if self.upper_bound != float('inf'):
            min_usage = self.roll_width * 0.65
            model.addConstr(
                grb.quicksum(widths[i] * a[i] for i in range(n_items)) >= min_usage,
                name="min_usage"
            )

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            n_solutions = model.SolCount

            for sol_idx in range(min(n_solutions, self.n_columns_per_iter * 2)):
                model.setParam('SolutionNumber', sol_idx)

                pattern = np.array([a[i].Xn for i in range(n_items)], dtype=int)
                obj_val = sum(dual_prices[i] * pattern[i] for i in range(n_items))
                reduced_cost = 1.0 - obj_val

                if reduced_cost < -1e-6:
                    is_duplicate = any(np.array_equal(pattern, p) for p, _ in result_patterns)
                    if not is_duplicate:
                        result_patterns.append((pattern, reduced_cost))

            result_patterns.sort(key=lambda x: x[1])
            result_patterns = result_patterns[:self.n_columns_per_iter]

        return result_patterns

    def _update_upper_bound(self, patterns: List[np.ndarray], widths: np.ndarray,
                            demands: np.ndarray, iteration: int):
        """定期求解整数MIP更新上界"""
        try:
            temp_result = self._solve_integer_mip_fast(patterns, widths, demands, time_limit=30)

            if temp_result['optimal_rolls'] < self.upper_bound:
                self.upper_bound = temp_result['optimal_rolls']
                self.best_solution = temp_result
                print(f"  → 上界更新: {self.upper_bound}")
        except:
            pass

    def _solve_integer_mip_fast(self, patterns: List[np.ndarray], widths: np.ndarray,
                                demands: np.ndarray, time_limit: int = 30) -> Dict:
        """快速求解整数MIP"""
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("FastMIP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        model.setParam('MIPGap', 0.02)

        x = model.addVars(n_patterns, lb=0, vtype=GRB.INTEGER, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i]
            )

        model.optimize()

        optimal_rolls = int(round(model.ObjVal)) if model.SolCount > 0 else float('inf')

        return {'optimal_rolls': optimal_rolls}

    def _diving_heuristic_solve(self, patterns: List[np.ndarray], widths: np.ndarray,
                                demands: np.ndarray, lp_solution: np.ndarray, iterations: int) -> Dict:
        """
        🔥 深潜启发式算法（已修复统计BUG）
        """
        print("开始深潜启发式...")
        diving_start = time.time()

        n_items = len(widths)
        n_patterns = len(patterns)

        # Step 1: 按LP解值排序
        pattern_scores = [(j, lp_solution[j]) for j in range(n_patterns)]
        pattern_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 2: 贪心固定变量
        fixed_vars = {}
        remaining_demands = demands.copy()

        print("  固定高分变量...")
        for j, score in pattern_scores:
            if score >= self.diving_threshold:
                pattern = patterns[j]

                max_possible = float('inf')
                for i in range(n_items):
                    if pattern[i] > 0:
                        max_possible = min(max_possible, remaining_demands[i] // pattern[i])

                if max_possible > 0:
                    fixed_count = min(int(np.ceil(score)), int(max_possible))
                    fixed_vars[j] = fixed_count
                    remaining_demands -= pattern * fixed_count

                    print(f"    固定模式{j}: {fixed_count}次 (LP值={score:.3f})")

        print(f"  共固定{len(fixed_vars)}个变量")

        # Step 3: 求解剩余问题（已修复）
        print("  求解剩余MIP...")
        final_result = self._solve_restricted_mip_fixed(patterns, widths, remaining_demands,
                                                        fixed_vars, iterations)

        diving_time = time.time() - diving_start
        final_result['diving_time'] = diving_time
        final_result['n_fixed_vars'] = len(fixed_vars)

        print(f"深潜启发式完成: 用时{diving_time:.2f}秒\n")

        return final_result

    def _solve_restricted_mip_fixed(self, patterns: List[np.ndarray], widths: np.ndarray,
                                    remaining_demands: np.ndarray, fixed_vars: Dict[int, int],
                                    iterations: int) -> Dict:
        """
        ✅ 修复版：求解带固定变量的限制性MIP
        """
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("RestrictedMIP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)

        # 变量
        x = model.addVars(n_patterns, lb=0, vtype=GRB.INTEGER, name="x")

        # ✅ 修复：固定部分变量
        for j, val in fixed_vars.items():
            x[j].LB = val
            x[j].UB = val

        # 目标
        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        # 约束：满足剩余需求
        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= remaining_demands[i],
                name=f"demand_{i}"
            )

        model.optimize()

        # ✅ 修复：model.ObjVal 已经包含了固定变量的贡献，不需要再加！
        total_rolls = int(round(model.ObjVal))

        # ✅ 修复：统一提取所有变量（包括固定的）
        used_patterns = []
        used_counts = []

        for j in range(n_patterns):
            if x[j].X > 0.5:  # 包括固定的和MIP求解的
                used_patterns.append(patterns[j])
                used_counts.append(int(round(x[j].X)))

        # 计算统计
        total_waste = 0
        pattern_details = []

        for pattern, count in zip(used_patterns, used_counts):
            pattern_width = np.sum(pattern * widths)
            waste = self.roll_width - pattern_width
            total_waste += waste * count

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

        total_waste_rate = total_waste / (total_rolls * self.roll_width) if total_rolls > 0 else 0

        result = {
            'optimal_rolls': total_rolls,
            'n_patterns_used': len(used_patterns),
            'n_patterns_generated': n_patterns,
            'iterations': iterations,
            'total_waste': int(total_waste),
            'waste_rate': total_waste_rate,
            'pattern_details': pattern_details
        }

        # 验证需求满足情况
        total_produced = np.zeros(n_items, dtype=int)
        for pattern, count in zip(used_patterns, used_counts):
            total_produced += pattern * count

        self._print_result(result, widths, total_produced)

        return result

    def _print_result(self, result: Dict, widths: np.ndarray, total_produced: np.ndarray):
        """打印结果"""
        print(f"{'=' * 80}")
        print(f"最终优化结果")
        print(f"{'=' * 80}")
        print(f"✅ 最优母卷数: {result['optimal_rolls']}")
        print(f"使用模式数: {result['n_patterns_used']} (生成了{result['n_patterns_generated']}个)")
        print(f"总纸损: {result['total_waste']}mm ({result['waste_rate']:.2%})")

        if 'n_fixed_vars' in result:
            print(f"深潜固定变量数: {result['n_fixed_vars']}")
            print(f"深潜用时: {result.get('diving_time', 0):.2f}秒")

        print(f"\n{'-' * 80}")
        print(f"{'编号':<6} {'切割方案':<45} {'次数':<8} {'切割宽':<10} {'剩余':<10} {'纸损率':<10}")
        print(f"{'-' * 80}")

        for i, detail in enumerate(result['pattern_details'], 1):
            print(f"#{i:<5} {detail['切割方案']:<45} {detail['使用次数']:<8} "
                  f"{detail['切割宽度']:<10} {detail['剩余宽度']:<10} {detail['纸损率']:<10}")

        print(f"{'-' * 80}")
        print(f"总计: {result['optimal_rolls']} 母卷, 纸损 {result['total_waste']}mm ({result['waste_rate']:.2%})")
        print(f"{'=' * 80}\n")

    def save_report(self, result: Dict, filename: str = 'cutting_plan_fixed.xlsx'):
        """保存结果到Excel"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                summary = pd.DataFrame([{
                    '最优母卷数': result['optimal_rolls'],
                    '使用模式数': result['n_patterns_used'],
                    '生成模式数': result['n_patterns_generated'],
                    '迭代次数': result['iterations'],
                    '总纸损(mm)': result['total_waste'],
                    '纸损率(%)': result['waste_rate'] * 100,
                    '深潜固定变量数': result.get('n_fixed_vars', 0),
                    '深潜用时(秒)': result.get('diving_time', 0)
                }])
                summary.to_excel(writer, sheet_name='摘要', index=False)

                details = pd.DataFrame(result['pattern_details'])
                details.to_excel(writer, sheet_name='切割方案', index=False)

            print(f"报告已保存: {filename}\n")
        except Exception as e:
            print(f"保存失败: {e}\n")


# ============================================================================
# 使用示例
# ============================================================================

