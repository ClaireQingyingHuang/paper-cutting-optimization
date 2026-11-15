"""Research implementation adapted from 3种算法测试.py. See docs/provenance.md."""

import gurobipy as grb
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Union


class HeuristicOptimizer:
    """启发式算法优化器（FFD & Residual）"""

    def __init__(self, roll_width: int):
        """
        初始化

        参数:
            roll_width: 母卷宽度(mm)
        """
        self.roll_width = roll_width

    def _parse_input(self, width_demands: Union[Dict[int, int], str]) -> Tuple[np.ndarray, np.ndarray]:
        """解析输入"""
        if isinstance(width_demands, str):
            wd = {}
            for pair in width_demands.split(','):
                if ':' in pair:
                    width, demand = pair.strip().split(':')
                    wd[int(width)] = int(demand)
            width_demands = wd

        widths = np.array(sorted(width_demands.keys()))
        demands = np.array([width_demands[w] for w in widths])
        return widths, demands

    # ========================================================================
    # FFD算法（First Fit Decreasing）
    # ========================================================================

    def solve_ffd(self, width_demands: Union[Dict[int, int], str]) -> Dict:
        """
        FFD算法：先降序排序，然后First Fit

        参数:
            width_demands: {宽度: 需求量}

        返回:
            优化结果字典
        """
        widths, demands = self._parse_input(width_demands)

        print(f"\n{'=' * 80}")
        print(f"FFD算法（First Fit Decreasing）")
        print(f"{'=' * 80}")
        print(f"母卷宽度: {self.roll_width}mm")
        print(f"产品种类: {len(widths)}")
        print(f"总需求: {np.sum(demands)}")
        print(f"{'=' * 80}\n")

        start = time.time()

        # 生成所有物品（按宽度降序）
        items = []
        for w, d in zip(widths, demands):
            items.extend([w] * d)
        items.sort(reverse=True)

        # FFD装箱
        bins = []  # 每个bin是一个list，记录宽度
        for item in items:
            placed = False
            for bin in bins:
                if sum(bin) + item <= self.roll_width:
                    bin.append(item)
                    placed = True
                    break
            if not placed:
                bins.append([item])

        solve_time = time.time() - start

        # 统计结果
        result = self._calculate_result(bins, widths, demands, solve_time, "FFD")
        self._print_result(result)

        return result

    # ========================================================================
    # Residual FFD（列生成 + FFD取整）
    # ========================================================================

    def solve_residual_ffd(self, width_demands: Union[Dict[int, int], str]) -> Dict:
        """
        Residual FFD：列生成松弛 → 向下取整 → FFD解决残差

        参数:
            width_demands: {宽度: 需求量}

        返回:
            优化结果字典
        """
        widths, demands = self._parse_input(width_demands)

        print(f"\n{'=' * 80}")
        print(f"Residual FFD算法（Column Generation + FFD）")
        print(f"{'=' * 80}")
        print(f"母卷宽度: {self.roll_width}mm")
        print(f"产品种类: {len(widths)}")
        print(f"总需求: {np.sum(demands)}")
        print(f"{'=' * 80}\n")

        start = time.time()

        # 第一步：列生成求松弛解
        print("第1步：列生成求松弛解...")
        patterns, pattern_freqs = self._column_generation_lp(widths, demands)

        # 第二步：向下取整
        print("第2步：向下取整...")
        bins = []
        remaining_demands = demands.copy()

        for pattern, freq in zip(patterns, pattern_freqs):
            freq_int = int(np.floor(freq))
            if freq_int > 0:
                for _ in range(freq_int):
                    bins.append(self._pattern_to_list(pattern, widths))
                remaining_demands -= pattern * freq_int

        print(f"  取整后使用 {len(bins)} 母卷，剩余需求 {np.sum(remaining_demands)}")

        # 第三步：FFD解决残差
        print("第3步：FFD解决残差...")
        if np.sum(remaining_demands) > 0:
            items = []
            for w, d in zip(widths, remaining_demands):
                items.extend([w] * int(d))
            items.sort(reverse=True)

            for item in items:
                placed = False
                for bin in bins:
                    if sum(bin) + item <= self.roll_width:
                        bin.append(item)
                        placed = True
                        break
                if not placed:
                    bins.append([item])

        solve_time = time.time() - start

        # 统计结果
        result = self._calculate_result(bins, widths, demands, solve_time, "Residual-FFD")
        self._print_result(result)

        return result

    # ========================================================================
    # Residual Greedy（列生成 + Greedy取整）
    # ========================================================================

    def solve_residual_greedy(self, width_demands: Union[Dict[int, int], str]) -> Dict:
        """
        Residual Greedy：列生成松弛 → 向下取整 → Greedy解决残差

        参数:
            width_demands: {宽度: 需求量}

        返回:
            优化结果字典
        """
        widths, demands = self._parse_input(width_demands)

        print(f"\n{'=' * 80}")
        print(f"Residual Greedy算法（Column Generation + Greedy）")
        print(f"{'=' * 80}")
        print(f"母卷宽度: {self.roll_width}mm")
        print(f"产品种类: {len(widths)}")
        print(f"总需求: {np.sum(demands)}")
        print(f"{'=' * 80}\n")

        start = time.time()

        # 第一步：列生成求松弛解
        print("第1步：列生成求松弛解...")
        patterns, pattern_freqs = self._column_generation_lp(widths, demands)

        # 第二步：向下取整
        print("第2步：向下取整...")
        bins = []
        remaining_demands = demands.copy()

        for pattern, freq in zip(patterns, pattern_freqs):
            freq_int = int(np.floor(freq))
            if freq_int > 0:
                for _ in range(freq_int):
                    bins.append(self._pattern_to_list(pattern, widths))
                remaining_demands -= pattern * freq_int

        print(f"  取整后使用 {len(bins)} 母卷，剩余需求 {np.sum(remaining_demands)}")

        # 第三步：Greedy解决残差
        print("第3步：Greedy解决残差...")
        while np.sum(remaining_demands) > 0:
            # 求解背包问题生成一个模式
            pattern = self._solve_knapsack_for_demands(widths, remaining_demands)

            # 计算可以使用多少次（不超过需求）
            max_freq = self.roll_width  # 理论最大值
            for i, qty in enumerate(pattern):
                if qty > 0:
                    max_freq = min(max_freq, remaining_demands[i] // qty)

            if max_freq > 0:
                bins.append(self._pattern_to_list(pattern, widths))
                remaining_demands -= pattern
            else:
                break

        solve_time = time.time() - start

        # 统计结果
        result = self._calculate_result(bins, widths, demands, solve_time, "Residual-Greedy")
        self._print_result(result)

        return result

    # ========================================================================
    # 辅助函数
    # ========================================================================

    def _column_generation_lp(self, widths: np.ndarray, demands: np.ndarray) -> Tuple[List, List]:
        """列生成求解LP松弛，返回(模式列表, 频率列表)"""
        n_items = len(widths)

        # 初始化单位矩阵
        patterns = [np.zeros(n_items, dtype=int) for _ in range(n_items)]
        for i in range(n_items):
            patterns[i][i] = 1

        iteration = 0
        while iteration < 1000:
            iteration += 1

            # 求解主问题
            dual_prices = self._solve_master_lp(patterns, widths, demands)

            # 求解背包
            new_pattern, reduced_cost = self._solve_knapsack(widths, dual_prices)

            if reduced_cost >= -1e-6:
                break

            patterns.append(new_pattern)

        # 求解最终LP获取频率
        pattern_freqs = self._get_lp_frequencies(patterns, widths, demands)

        return patterns, pattern_freqs

    def _solve_master_lp(self, patterns: List, widths: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """求解主问题LP，返回对偶价格"""
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("MasterLP")
        model.setParam('OutputFlag', 0)

        x = model.addVars(n_patterns, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i],
                name=f"d_{i}"
            )

        model.optimize()

        dual_prices = np.array([model.getConstrByName(f"d_{i}").Pi for i in range(n_items)])
        return dual_prices

    def _solve_knapsack(self, widths: np.ndarray, dual_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """求解背包子问题"""
        n_items = len(widths)

        model = grb.Model("KP")
        model.setParam('OutputFlag', 0)

        a = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, name="a")

        model.setObjective(
            grb.quicksum(dual_prices[i] * a[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )

        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n_items)) <= self.roll_width
        )

        model.optimize()

        new_pattern = np.array([int(a[i].X) for i in range(n_items)], dtype=int)
        reduced_cost = 1.0 - model.ObjVal

        return new_pattern, reduced_cost

    def _solve_knapsack_for_demands(self, widths: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """求解背包问题（直接最大化物品总长度）"""
        n_items = len(widths)

        model = grb.Model("KP")
        model.setParam('OutputFlag', 0)

        a = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, name="a")

        # 目标：最大化切割长度
        model.setObjective(
            grb.quicksum(widths[i] * a[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )

        # 约束：不超过母卷宽度
        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n_items)) <= self.roll_width
        )

        # 约束：不超过需求
        for i in range(n_items):
            model.addConstr(a[i] <= demands[i])

        model.optimize()

        pattern = np.array([int(a[i].X) for i in range(n_items)], dtype=int)
        return pattern

    def _get_lp_frequencies(self, patterns: List, widths: np.ndarray, demands: np.ndarray) -> List:
        """求解LP获取模式频率"""
        n_items = len(widths)
        n_patterns = len(patterns)

        model = grb.Model("LP")
        model.setParam('OutputFlag', 0)

        x = model.addVars(n_patterns, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)

        for i in range(n_items):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) >= demands[i]
            )

        model.optimize()

        freqs = [x[j].X for j in range(n_patterns)]
        return freqs

    def _pattern_to_list(self, pattern: np.ndarray, widths: np.ndarray) -> List[int]:
        """将模式数组转为宽度列表"""
        result = []
        for i, qty in enumerate(pattern):
            result.extend([widths[i]] * int(qty))
        return result

    def _calculate_result(self, bins: List, widths: np.ndarray,
                          demands: np.ndarray, solve_time: float, method: str) -> Dict:
        """计算统计结果"""
        optimal_rolls = len(bins)
        total_waste = 0
        pattern_dict = {}

        # 统计每种模式
        for bin in bins:
            bin_sorted = tuple(sorted(bin, reverse=True))
            pattern_dict[bin_sorted] = pattern_dict.get(bin_sorted, 0) + 1

        # 构建详情
        pattern_details = []
        for pattern, count in sorted(pattern_dict.items(), key=lambda x: -x[1]):
            pattern_width = sum(pattern)
            waste = self.roll_width - pattern_width
            total_waste += waste * count

            # 统计每种宽度数量
            width_counts = {}
            for w in pattern:
                width_counts[w] = width_counts.get(w, 0) + 1

            desc_parts = []
            for w in sorted(width_counts.keys(), reverse=True):
                desc_parts.append(f"{width_counts[w]}×{w}mm")

            pattern_details.append({
                '切割方案': ' + '.join(desc_parts),
                '使用次数': count,
                '切割宽度': int(pattern_width),
                '剩余宽度': int(waste),
                '纸损率': f"{waste / self.roll_width * 100:.2f}%"
            })

        waste_rate = total_waste / (optimal_rolls * self.roll_width) if optimal_rolls > 0 else 0

        return {
            'method': method,
            'optimal_rolls': optimal_rolls,
            'n_patterns_used': len(pattern_dict),
            'total_waste': int(total_waste),
            'waste_rate': waste_rate,
            'solve_time': solve_time,
            'pattern_details': pattern_details
        }

    def _print_result(self, result: Dict):
        """打印结果"""
        print(f"{'=' * 80}")
        print(f"优化结果 - {result['method']}")
        print(f"{'=' * 80}")
        print(f"最优母卷数: {result['optimal_rolls']}")
        print(f"使用模式数: {result['n_patterns_used']}")
        print(f"总纸损: {result['total_waste']}mm ({result['waste_rate']:.2%})")
        print(f"求解时间: {result['solve_time']:.3f}秒")
        print(f"\n{'-' * 80}")
        print(f"{'编号':<6} {'切割方案':<45} {'次数':<8} {'切割宽':<10} {'剩余':<10} {'纸损率':<10}")
        print(f"{'-' * 80}")

        for i, detail in enumerate(result['pattern_details'], 1):
            print(f"#{i:<5} {detail['切割方案']:<45} {detail['使用次数']:<8} "
                  f"{detail['切割宽度']:<10} {detail['剩余宽度']:<10} {detail['纸损率']:<10}")

        print(f"{'-' * 80}")
        print(f"总计: {result['optimal_rolls']} 母卷, 纸损 {result['total_waste']}mm ({result['waste_rate']:.2%})")
        print(f"{'=' * 80}\n")

    def save_comparison(self, results: List[Dict], filename: str = '算法对比.xlsx'):
        """保存多个算法对比结果"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 对比摘要
                summary_data = []
                for r in results:
                    summary_data.append({
                        '算法': r['method'],
                        '母卷数': r['optimal_rolls'],
                        '模式数': r['n_patterns_used'],
                        '总纸损(mm)': r['total_waste'],
                        '纸损率(%)': r['waste_rate'] * 100,
                        '求解时间(秒)': r['solve_time']
                    })
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='算法对比', index=False)

                # 每个算法的详细方案
                for r in results:
                    details = pd.DataFrame(r['pattern_details'])
                    sheet_name = r['method'][:30]  # Excel sheet名称限制
                    details.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"对比报告已保存: {filename}\n")
        except Exception as e:
            print(f"保存失败: {e}\n")


# ============================================================================
# 使用示例
# ============================================================================

