"""Research implementation adapted from four_algorithms.py. See docs/provenance.md."""

import numpy as np
from time import time as get_time  # 修复：正确导入time
from typing import Dict, List, Tuple, Optional
try:
    import gurobipy as grb
    from gurobipy import GRB
except ImportError:
    grb = None
    GRB = None

# 尝试导入PuLP用于MSHP算法
try:
    from pulp import *

    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class CuttingStockAlgorithms:
    """5种切割算法的统一接口"""

    def __init__(self, roll_width: int = 1000):
        """
        初始化

        参数:
            roll_width: 母卷宽度
        """
        self.roll_width = roll_width

    def _parse_input(self, width_demands: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """解析输入为numpy数组"""
        widths = np.array(sorted(width_demands.keys(), reverse=True))  # 降序排序
        demands = np.array([width_demands[w] for w in widths])
        return widths, demands

    def _calculate_theoretical_lb(self, widths: np.ndarray, demands: np.ndarray) -> int:
        """
        计算理论下界（改进版）

        LB = max(L0, L1)
        - L0: 连续下界（总长度÷母卷宽度）
        - L1: 基于最大物料的下界（考虑每种物料单独需要的最少母卷数）
        """
        # L0: 连续下界
        total_length = np.sum(widths * demands)
        L0 = int(np.ceil(total_length / self.roll_width))

        # L1: 基于每种物料的下界
        L1 = 0
        for i in range(len(widths)):
            items_per_roll = self.roll_width // widths[i]
            if items_per_roll > 0:
                min_rolls_for_item = int(np.ceil(demands[i] / items_per_roll))
                L1 = max(L1, min_rolls_for_item)

        # 理论下界取两者最大值
        return max(L0, L1)

    # ========================================================================
    # 算法 1: FFD (First Fit Decreasing)
    # ========================================================================

    def solve_ffd(self, width_demands: Dict[int, int], verbose: bool = False) -> Dict:
        """FFD算法：物品按宽度降序排序，然后First Fit装箱"""
        widths, demands = self._parse_input(width_demands)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"FFD算法 (First Fit Decreasing)")
            print(f"{'=' * 60}")

        start_time = get_time()

        # 生成所有物品（已按宽度降序）
        items = []
        for w, d in zip(widths, demands):
            items.extend([w] * d)

        # FFD装箱
        bins = []
        for item in items:
            placed = False
            for bin in bins:
                if sum(bin) + item <= self.roll_width:
                    bin.append(item)
                    placed = True
                    break
            if not placed:
                bins.append([item])

        solve_time = get_time() - start_time

        return self._build_result(bins, widths, demands, solve_time, "FFD", verbose)

    # ========================================================================
    # 算法 2: Haessler's SHP (Sequential Heuristic Procedure)
    # ========================================================================

    def solve_haessler_shp(self, width_demands: Dict[int, int], verbose: bool = False) -> Dict:
        """
        Haessler (1975) 的序贯启发式算法
        原始的Sequential Heuristic Procedure
        """
        widths, demands = self._parse_input(width_demands)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Haessler's SHP (1975)")
            print(f"{'=' * 60}")
            print(f"母卷宽度: {self.roll_width}mm")
            print(f"需求规格数: {len(widths)}")
            print(f"总需求量: {sum(demands)}")

        start_time = get_time()

        # 按长度降序排序
        sorted_items = sorted(width_demands.items(), key=lambda x: x[0], reverse=True)
        residual = {w: d for w, d in sorted_items}
        bins = []

        # 计算初始指标
        I1 = self._calculate_I1(residual)
        I2 = self._calculate_I2(residual, I1)

        # 设置抱负水平参数
        MINR = max(1, int(I2) - 1)  # 最小物品数
        MAXR = 100  # 最大物品数（可根据切刀数调整）
        MAXTL = int(0.03 * self.roll_width)  # 最大允许浪费

        alpha = 0.7  # MINU = alpha * I1

        pattern_id = 0

        while sum(residual.values()) > 0:
            pattern_id += 1

            # 生成切割模式（字典序搜索）
            pattern, waste = self._generate_pattern_lexicographic(
                residual, MINR, MAXR, MAXTL, alpha, I1
            )

            if not pattern:
                break

            # 计算使用频率
            frequency = min(residual[w] // c for w, c in pattern.items() if c > 0)
            frequency = max(1, frequency)

            # 更新剩余需求
            for width, count in pattern.items():
                residual[width] -= frequency * count
                if residual[width] < 0:
                    residual[width] = 0

            # 保存模式
            for _ in range(frequency):
                bin_pattern = []
                for width, count in pattern.items():
                    bin_pattern.extend([width] * count)
                bins.append(bin_pattern)

            if verbose:
                cuts_str = ' + '.join([f"{c}×{w}mm" for w, c in sorted(pattern.items(), reverse=True)])
                print(f"模式 {pattern_id}: {cuts_str}")
                print(f"  频率: {frequency}, 浪费: {waste}mm ({waste / self.roll_width * 100:.1f}%)")
                remaining_items = {k: v for k, v in residual.items() if v > 0}
                if remaining_items:
                    print(f"  剩余: {remaining_items}")

            # 重新计算I1和I2
            if sum(residual.values()) > 0:
                I1 = self._calculate_I1(residual)
                I2 = self._calculate_I2(residual, I1)

        solve_time = get_time() - start_time

        return self._build_result(bins, widths, demands, solve_time, "Haessler-SHP", verbose)

    def _calculate_I1(self, residual: Dict[int, int]) -> float:
        """计算I1：满足剩余需求所需的母卷数估计"""
        total = sum(w * d for w, d in residual.items())
        return total / self.roll_width

    def _calculate_I2(self, residual: Dict[int, int], I1: float) -> float:
        """计算I2：每个母卷平均应获得的物品数"""
        total_items = sum(residual.values())
        return total_items / max(I1, 1)

    def _generate_pattern_lexicographic(self, residual: Dict[int, int],
                                        MINR: int, MAXR: int, MAXTL: int,
                                        alpha: float, I1: float) -> Tuple[Dict[int, int], int]:
        """
        使用字典序搜索生成切割模式

        Returns:
            (pattern, waste): 切割模式和浪费
        """
        MINU = max(1, int(alpha * I1))

        best_pattern = None
        best_waste = float('inf')

        # 逐渐降低MINU直到找到可行模式
        while MINU >= 1:
            # 字典序搜索：按降序尝试组合
            pattern = self._lexicographic_search(residual, MINU, MINR, MAXR, MAXTL)

            if pattern:
                waste = self.roll_width - sum(w * c for w, c in pattern.items())
                if waste < best_waste:
                    best_pattern = pattern
                    best_waste = waste
                break

            MINU -= 1

        # 如果没找到满足条件的，返回浪费最小的
        if best_pattern is None:
            best_pattern = self._generate_min_waste_pattern(residual)
            best_waste = self.roll_width - sum(w * c for w, c in best_pattern.items())

        return best_pattern, best_waste

    def _lexicographic_search(self, residual: Dict[int, int],
                              MINU: int, MINR: int, MAXR: int,
                              MAXTL: int) -> Optional[Dict[int, int]]:
        """字典序搜索切割模式"""
        pattern = {}
        remaining = self.roll_width

        # 按长度降序处理
        for width in sorted(residual.keys(), reverse=True):
            if residual[width] == 0:
                continue

            # 计算最大可放入数量
            max_fit = remaining // width
            # 考虑MINU约束
            max_demand = residual[width] // MINU if MINU > 0 else residual[width]

            count = min(max_fit, max_demand, residual[width])

            if count > 0:
                pattern[width] = count
                remaining -= width * count

        # 检查约束
        total_items = sum(pattern.values())
        waste = remaining

        if (MINR <= total_items <= MAXR and
                waste <= MAXTL and
                all(residual[w] >= MINU * c for w, c in pattern.items())):
            return pattern

        return None

    def _generate_min_waste_pattern(self, residual: Dict[int, int]) -> Dict[int, int]:
        """生成浪费最小的模式（贪婪填充）"""
        pattern = {}
        remaining = self.roll_width

        for width in sorted(residual.keys(), reverse=True):
            if residual[width] > 0:
                count = min(remaining // width, residual[width])
                if count > 0:
                    pattern[width] = count
                    remaining -= width * count

        return pattern

    # ========================================================================
    # 算法 3: Modified SHP (MSHP)
    # ========================================================================

    def solve_modified_shp(self, width_demands: Dict[int, int],
                          alpha_range: List[float] = None,
                          beta_range: List[float] = None,
                          verbose: bool = False) -> Dict:
        """
        Modified Sequential Heuristic Procedure (MSHP)
        论文提出的改进算法，使用整数背包问题生成切割模式
        """
        if not HAS_PULP:
            raise ImportError("MSHP需要PuLP库支持，请安装: pip install pulp")

        widths, demands = self._parse_input(width_demands)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Modified Sequential Heuristic Procedure (MSHP)")
            print(f"{'=' * 60}")
            print(f"母卷宽度: {self.roll_width}mm")
            print(f"需求规格数: {len(widths)}")
            print(f"总需求量: {sum(demands)}")

        start_time = get_time()

        # 默认参数范围
        if alpha_range is None:
            alpha_range = [0.15 + 0.15 * i for i in range(4)]  # [0.15, 0.30, 0.45, 0.60]
        if beta_range is None:
            beta_range = [0.3 + 0.025 * i + 0.6 * (i // 18) for i in range(18)]

        # Phase 1: 生成多个切割方案
        all_patterns = []
        best_plan = None
        best_cost = float('inf')

        if verbose:
            print(f"\nPhase 1: 生成切割方案")
            print(f"  Alpha范围: {alpha_range}")
            print(f"  Beta范围: {[f'{b:.2f}' for b in beta_range[:5]]}...")

        # 将demands转为字典
        demands_dict = {int(w): int(d) for w, d in zip(widths, demands)}

        plan_count = 0
        for alpha_g in alpha_range:
            for beta_g in beta_range:
                plan_count += 1
                patterns = self._generate_cutting_plan_mshp(demands_dict, alpha_g, beta_g)

                # 评估方案
                total_stocks = len(patterns)  # Count rolls, not individual pieces.
                total_setups = len(set(tuple(sorted(bin)) for bin in patterns))
                cost = total_stocks * self.roll_width + total_setups * 100

                if cost < best_cost:
                    best_cost = cost
                    best_plan = patterns

                # 收集所有不同的模式
                for p in patterns:
                    pattern_tuple = tuple(sorted(p))
                    if pattern_tuple not in [tuple(sorted(pp)) for pp in all_patterns]:
                        all_patterns.append(p)

        if verbose:
            print(f"  生成了 {plan_count} 个方案")
            print(f"  收集了 {len(all_patterns)} 个不同的切割模式")
            print(f"  最佳方案: {len(best_plan)} 母卷")

        # Phase 2: 使用整数规划优化（可选）
        if verbose:
            print(f"\nPhase 2: 整数规划优化")

        try:
            optimized_patterns = self._optimize_with_ip_mshp(demands_dict, all_patterns, best_plan)
            final_bins = optimized_patterns
        except:
            if verbose:
                print("  优化失败，使用Phase 1最佳方案")
            final_bins = best_plan

        solve_time = get_time() - start_time

        return self._build_result(final_bins, widths, demands, solve_time, "Modified-SHP", verbose)

    def _generate_cutting_plan_mshp(self, demands: Dict[int, int],
                                   alpha_g: float, beta_g: float) -> List[List[int]]:
        """使用给定参数生成一个切割方案"""
        bins = []
        residual = demands.copy()

        MINR = 1
        MAXR = 100

        while sum(residual.values()) > 0:
            # 计算I1
            I1 = sum(w * r for w, r in residual.items()) / self.roll_width

            # 计算ui（Algorithm 2）
            u_bounds = self._calculate_u_bounds_mshp(residual, alpha_g, beta_g, I1)

            # 生成模式（使用整数背包问题，Model 14-20）
            pattern, waste = self._generate_pattern_knapsack_mshp(
                residual, u_bounds, MINR, MAXR
            )

            if not pattern:
                # 回退到贪婪
                pattern = self._greedy_pattern_mshp(residual)
                waste = self.roll_width - sum(w * c for w, c in pattern.items())

            # 计算频率
            frequency = min(residual[w] // c for w, c in pattern.items() if c > 0)
            frequency = max(1, frequency)

            # 更新剩余需求
            for width, count in pattern.items():
                residual[width] -= frequency * count
                residual[width] = max(0, residual[width])

            # 保存模式
            for _ in range(frequency):
                bin_pattern = []
                for width, count in pattern.items():
                    bin_pattern.extend([width] * count)
                bins.append(bin_pattern)

        return bins

    def _calculate_u_bounds_mshp(self, residual: Dict[int, int],
                                alpha_g: float, beta_g: float, I1: float) -> Dict[int, int]:
        """
        计算ui上界（Algorithm 2）

        Returns:
            {width: upper_bound}
        """
        MINU = max(1, int(alpha_g * I1))
        u_bounds = {}

        while MINU >= 1:
            for width, demand in residual.items():
                # ui = min(W/wi, ri/MINU)
                geometric_bound = self.roll_width // width
                demand_bound = demand // MINU if MINU > 0 else demand
                u_bounds[width] = min(geometric_bound, demand_bound)

            # 检查停止条件
            utilization = sum(u_bounds[w] * w for w in residual.keys()) / self.roll_width

            if utilization >= beta_g or MINU <= 1:
                break

            MINU -= 1

        return u_bounds

    def _generate_pattern_knapsack_mshp(self, residual: Dict[int, int],
                                       u_bounds: Dict[int, int],
                                       MINR: int, MAXR: int) -> Tuple[Dict[int, int], int]:
        """
        使用整数背包问题生成切割模式（Model 14-20）

        Min v + t
        s.t. sum(wi * ai) + t = W
             ri*v - ai >= 0, for all i
             MINR <= sum(ai) <= MAXR
             0 <= ai <= ui, for all i
        """
        try:
            # 创建问题
            prob = LpProblem("Pattern_Generation", LpMinimize)

            # 决策变量
            a_vars = {w: LpVariable(f"a_{w}", 0, u_bounds.get(w, residual[w]), LpInteger)
                      for w in residual.keys() if residual[w] > 0}
            t = LpVariable("t", 0, self.roll_width, LpContinuous)  # 浪费
            v = LpVariable("v", 0, None, LpContinuous)  # 辅助变量

            # 目标函数: Min v + t
            prob += v + t

            # 约束1: sum(wi * ai) + t = W
            prob += lpSum([w * a_vars[w] for w in a_vars]) + t == self.roll_width

            # 约束2: ri*v - ai >= 0
            for w in a_vars:
                prob += residual[w] * v - a_vars[w] >= 0

            # 约束3: MINR <= sum(ai) <= MAXR
            prob += lpSum([a_vars[w] for w in a_vars]) >= MINR
            prob += lpSum([a_vars[w] for w in a_vars]) <= MAXR

            # 求解
            prob.solve(PULP_CBC_CMD(msg=0))

            if prob.status == 1:  # Optimal
                pattern = {w: int(a_vars[w].varValue) for w in a_vars
                           if a_vars[w].varValue > 0.5}
                waste = int(t.varValue)
                return pattern, waste

        except Exception as e:
            pass

        return {}, 0

    def _greedy_pattern_mshp(self, residual: Dict[int, int]) -> Dict[int, int]:
        """贪婪生成模式"""
        pattern = {}
        remaining = self.roll_width

        for width in sorted(residual.keys(), reverse=True):
            if residual[width] > 0:
                count = min(remaining // width, residual[width])
                if count > 0:
                    pattern[width] = count
                    remaining -= width * count

        return pattern

    def _optimize_with_ip_mshp(self, demands: Dict[int, int],
                              all_patterns: List[List[int]],
                              initial_solution: List[List[int]]) -> List[List[int]]:
        """
        使用整数规划优化模式使用（Model 21-25）

        Min cr*sum(xj) + cs*sum(εj)
        s.t. sum(aij*xj) >= di
             xj <= Mj*εj
             xj >= 0, εj in {0,1}
        """
        # 将patterns转为字典格式
        pattern_dicts = []
        for p in all_patterns:
            pd = {}
            for w in p:
                pd[w] = pd.get(w, 0) + 1
            pattern_dicts.append(pd)

        # 创建问题
        prob = LpProblem("Cutting_Stock_Optimization", LpMinimize)

        # 决策变量
        n_patterns = len(pattern_dicts)
        x_vars = [LpVariable(f"x_{j}", 0, None, LpInteger) for j in range(n_patterns)]
        e_vars = [LpVariable(f"e_{j}", 0, 1, LpBinary) for j in range(n_patterns)]

        # 目标函数
        cr = self.roll_width  # 母卷成本
        cs = 100  # 换型成本
        prob += lpSum([cr * x_vars[j] + cs * e_vars[j] for j in range(n_patterns)])

        # 约束1: 满足需求
        for width, demand in demands.items():
            prob += lpSum([pattern_dicts[j].get(width, 0) * x_vars[j]
                           for j in range(n_patterns)]) >= demand

        # 约束2: xj <= Mj * εj
        for j in range(n_patterns):
            # Mj = max possible frequency
            Mj = 1000  # 设置一个足够大的值
            for w in pattern_dicts[j].keys():
                if pattern_dicts[j].get(w, 0) > 0:
                    Mj = min(Mj, demands[w] // pattern_dicts[j].get(w, 1))
            Mj = max(Mj, 1000)
            prob += x_vars[j] <= Mj * e_vars[j]

        # 求解（限制时间）
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10))

        # 提取解
        result_bins = []
        for j in range(n_patterns):
            freq = int(x_vars[j].varValue) if x_vars[j].varValue else 0
            if freq > 0:
                for _ in range(freq):
                    result_bins.append(all_patterns[j])

        return result_bins if result_bins else initial_solution

    # ========================================================================
    # 算法 4: Residual Greedy
    # ========================================================================

    def solve_residual_greedy(self, width_demands: Dict[int, int], verbose: bool = False) -> Dict:
        """Residual Greedy: 列生成松弛 → 向下取整 → Greedy解决残差"""
        widths, demands = self._parse_input(width_demands)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Residual Greedy (CG + Floor + Greedy)")
            print(f"{'=' * 60}")

        start_time = get_time()

        # Step 1: 列生成求松弛解
        patterns = self._column_generation(widths, demands, verbose=False)
        pattern_freqs = self._get_lp_frequencies(patterns, widths, demands)

        # Step 2: 向下取整
        bins = []
        remaining_demands = demands.copy()

        for pattern, freq in zip(patterns, pattern_freqs):
            freq_int = int(np.floor(freq))
            if freq_int > 0:
                for _ in range(freq_int):
                    bins.append(self._pattern_to_list(pattern, widths))
                remaining_demands = np.maximum(remaining_demands - pattern * freq_int, 0)

        if verbose:
            print(f"  取整后: {len(bins)}个母卷, 剩余需求={np.sum(remaining_demands)}")

        # Step 3: Greedy解决残差
        while np.sum(remaining_demands) > 0:
            pattern = self._solve_greedy_knapsack(widths, remaining_demands)

            # 计算这个模式最多能用几次
            max_freq = float('inf')
            for i, qty in enumerate(pattern):
                if qty > 0:
                    max_freq = min(max_freq, remaining_demands[i] // qty)

            max_freq = int(max_freq)

            if max_freq > 0:
                for _ in range(max_freq):
                    bins.append(self._pattern_to_list(pattern, widths))
                remaining_demands = np.maximum(remaining_demands - pattern * max_freq, 0)
            else:
                break

        solve_time = get_time() - start_time

        return self._build_result(bins, widths, demands, solve_time, "Residual-Greedy", verbose)

    def _solve_greedy_knapsack(self, widths: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """Greedy背包子问题：最大化切割长度"""
        n = len(widths)

        model = grb.Model("GreedyKP")
        model.setParam('OutputFlag', 0)

        a = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="a")

        # 目标：最大化切割长度
        model.setObjective(
            grb.quicksum(widths[i] * a[i] for i in range(n)),
            GRB.MAXIMIZE
        )

        # 约束1：不超过母卷宽度
        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n)) <= self.roll_width
        )

        # 约束2：不超过需求
        for i in range(n):
            model.addConstr(a[i] <= demands[i])

        model.optimize()

        pattern = np.array([int(a[i].X) for i in range(n)], dtype=int)
        return pattern

    # ========================================================================
    # 算法 5: Standard Column Generation
    # ========================================================================

    def solve_column_generation(self, width_demands: Dict[int, int],
                                time_limit: int = 300, verbose: bool = False) -> Dict:
        """标准列生成算法（求解整数MIP）"""
        widths, demands = self._parse_input(width_demands)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Standard Column Generation")
            print(f"{'=' * 60}")

        start_time = get_time()

        # 列生成
        patterns = self._column_generation(widths, demands, verbose)

        if verbose:
            print(f"  生成了{len(patterns)}个模式")

        # 求解整数MIP
        bins = self._solve_integer_mip(patterns, widths, demands, time_limit)

        solve_time = get_time() - start_time

        return self._build_result(bins, widths, demands, solve_time, "Column-Generation", verbose)

    # ========================================================================
    # 兼容性方法（用于旧测试脚本）
    # ========================================================================

    def solve_residual_ffd(self, width_demands: Dict[int, int], verbose: bool = False) -> Dict:
        """兼容旧测试脚本：现在调用Haessler's SHP，并使用新名称"""
        result = self.solve_haessler_shp(width_demands, verbose)
        # 使用新的算法名称
        result['method'] = 'Haessler-SHP'
        return result

    def solve_advanced_column_generation(self, width_demands: Dict[int, int],
                                         time_limit: int = 300, verbose: bool = False) -> Dict:
        """兼容旧测试脚本：现在调用Modified SHP，并使用新名称"""
        if HAS_PULP:
            result = self.solve_modified_shp(width_demands, verbose)
            # 使用新的算法名称
            result['method'] = 'Modified-SHP'
        else:
            # 如果没有PuLP，回退到Haessler's SHP
            result = self.solve_haessler_shp(width_demands, verbose)
            result['method'] = 'Haessler-SHP(fallback)'
        return result

    # ========================================================================
    # 列生成辅助方法
    # ========================================================================

    def _column_generation(self, widths: np.ndarray, demands: np.ndarray,
                           verbose: bool = False) -> List[np.ndarray]:
        """列生成主循环"""
        n = len(widths)

        # 初始模式：单位矩阵
        patterns = [np.eye(n, dtype=int)[i] for i in range(n)]

        iteration = 0
        while iteration < 1000:
            iteration += 1

            # 求解LP主问题
            dual_prices = self._solve_master_lp(patterns, widths, demands)

            # 求解背包子问题
            new_pattern, reduced_cost = self._solve_pricing_knapsack(widths, dual_prices)

            if reduced_cost >= -1e-6:
                break

            patterns.append(new_pattern)

        if verbose:
            print(f"  列生成: {iteration}次迭代, 生成{len(patterns)}个模式")

        return patterns

    def _solve_master_lp(self, patterns: List[np.ndarray],
                         widths: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """求解LP主问题，返回对偶价格"""
        n = len(widths)
        m = len(patterns)

        model = grb.Model("MasterLP")
        model.setParam('OutputFlag', 0)

        x = model.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)

        constrs = []
        for i in range(n):
            c = model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(m)) >= demands[i]
            )
            constrs.append(c)

        model.optimize()

        dual_prices = np.array([constrs[i].Pi for i in range(n)])
        return dual_prices

    def _solve_pricing_knapsack(self, widths: np.ndarray,
                                dual_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """求解定价子问题（背包）"""
        n = len(widths)

        model = grb.Model("PricingKP")
        model.setParam('OutputFlag', 0)

        a = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="a")

        model.setObjective(
            grb.quicksum(dual_prices[i] * a[i] for i in range(n)),
            GRB.MAXIMIZE
        )

        model.addConstr(
            grb.quicksum(widths[i] * a[i] for i in range(n)) <= self.roll_width
        )

        model.optimize()

        pattern = np.array([int(a[i].X) for i in range(n)], dtype=int)
        reduced_cost = 1.0 - model.ObjVal

        return pattern, reduced_cost

    def _solve_integer_mip(self, patterns: List[np.ndarray],
                           widths: np.ndarray, demands: np.ndarray,
                           time_limit: int) -> List[List[int]]:
        """求解整数MIP"""
        n = len(widths)
        m = len(patterns)

        model = grb.Model("MasterMIP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        x = model.addVars(m, lb=0, vtype=GRB.INTEGER, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)

        for i in range(n):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(m)) >= demands[i]
            )

        model.optimize()

        # 提取解
        bins = []
        for j in range(m):
            if x[j].X > 0.5:
                count = int(round(x[j].X))
                for _ in range(count):
                    bins.append(self._pattern_to_list(patterns[j], widths))

        return bins

    def _get_lp_frequencies(self, patterns: List[np.ndarray],
                            widths: np.ndarray, demands: np.ndarray) -> List[float]:
        """求解LP获取模式频率"""
        n = len(widths)
        m = len(patterns)

        model = grb.Model("LP")
        model.setParam('OutputFlag', 0)

        x = model.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

        model.setObjective(grb.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)

        for i in range(n):
            model.addConstr(
                grb.quicksum(patterns[j][i] * x[j] for j in range(m)) >= demands[i]
            )

        model.optimize()

        return [x[j].X for j in range(m)]

    # ========================================================================
    # 辅助函数
    # ========================================================================

    def _pattern_to_list(self, pattern: np.ndarray, widths: np.ndarray) -> List[int]:
        """将模式数组转为宽度列表"""
        result = []
        for i, qty in enumerate(pattern):
            result.extend([widths[i]] * int(qty))
        return result

    def _build_result(self, bins: List[List[int]], widths: np.ndarray,
                      demands: np.ndarray, solve_time: float,
                      method: str, verbose: bool) -> Dict:
        """构建结果字典"""
        n_rolls = len(bins)

        # 统计模式
        pattern_dict = {}
        for bin in bins:
            bin_sorted = tuple(sorted(bin, reverse=True))
            pattern_dict[bin_sorted] = pattern_dict.get(bin_sorted, 0) + 1

        # 计算损耗
        total_waste = sum(self.roll_width - sum(bin) for bin in bins)
        waste_rate = total_waste / (n_rolls * self.roll_width) if n_rolls > 0 else 0
        theoretical_lb = self._calculate_theoretical_lb(widths, demands)

        result = {
            'method': method,
            'n_rolls': n_rolls,
            'n_patterns': len(pattern_dict),
            'total_waste': int(total_waste),
            'waste_rate': waste_rate,
            'solve_time': solve_time,
            'theoretical_lb': theoretical_lb,
            'gap': n_rolls - theoretical_lb,
            'patterns': bins
        }

        if verbose:
            print(f"  母卷数: {n_rolls} (理论下界={theoretical_lb}, GAP={result['gap']})")
            print(f"  模式数: {len(pattern_dict)}")
            print(f"  损耗率: {waste_rate:.2%}")
            print(f"  时间: {solve_time:.3f}秒")

        return result


# ========================================================================
# 示例使用
# ========================================================================

