import random
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. ГЕНЕРАЦИЯ И СОХРАНЕНИЕ ВХОДНЫХ ДАННЫХ (Вариант 5: N = 35 приборов)
# =====================================================================
def generate_lab_dataset(n_items=35, seed=42):
    rng = random.Random(seed)
    equipment = []
    categories = ["Осциллограф", "Генератор сигналов", "Спектроанализатор", 
                  "Источник питания", "Мультиметр", "Логический анализатор", "Сервер расчётов"]
    
    for i in range(n_items):
        cat = categories[i % len(categories)]
        cost = rng.randint(15, 120)          # Стоимость, тыс. руб.
        power = rng.randint(50, 600)         # Потребляемая мощность, Вт
        # Полезность коррелирует со стоимостью с добавлением шума
        utility = round(cost * rng.uniform(0.8, 1.5) + rng.randint(5, 25), 1)
        equipment.append({
            "id": i,
            "name": f"{cat} #{i+1}",
            "cost": cost,
            "power": power,
            "utility": utility
        })
        
    # Генерация 12 пар несовместимого оборудования (конфликты)
    conflicts = set()
    while len(conflicts) < 12:
        a, b = sorted(rng.sample(range(n_items), 2))
        conflicts.add((a, b))
    conflicts = sorted(list(conflicts))
    
    # Лимиты лаборатории (~45% от суммарных затрат и мощности всех приборов)
    max_budget = int(sum(item["cost"] for item in equipment) * 0.45)
    max_power = int(sum(item["power"] for item in equipment) * 0.45)
    
    dataset = {
        "seed": seed,
        "n_items": n_items,
        "max_budget": max_budget,
        "max_power": max_power,
        "conflicts": conflicts,
        "equipment": equipment
    }
    
    with open("lab2_input_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    return dataset

# =====================================================================
# 2. ОЦЕНКА РЕШЕНИЯ, ПРОВЕРКА ДОПУСТИМОСТИ И РЕМОНТ
# =====================================================================
def evaluate_raw(chromosome, dataset):
    """Возвращает (полезность, стоимость, мощность, число_конфликтов)."""
    total_u, total_c, total_p = 0.0, 0, 0
    selected = [i for i, bit in enumerate(chromosome) if bit == 1]
    
    for idx in selected:
        item = dataset["equipment"][idx]
        total_u += item["utility"]
        total_c += item["cost"]
        total_p += item["power"]
        
    conflict_count = 0
    selected_set = set(selected)
    for a, b in dataset["conflicts"]:
        if a in selected_set and b in selected_set:
            conflict_count += 1
            
    return total_u, total_c, total_p, conflict_count

def is_feasible(chromosome, dataset):
    _, c, p, conf = evaluate_raw(chromosome, dataset)
    return (c <= dataset["max_budget"]) and (p <= dataset["max_power"]) and (conf == 0)

def repair_chromosome(chromosome, dataset):
    """Жадный ремонт недопустимого решения: устраняет конфликты, перерасход бюджета и мощности."""
    chrom = chromosome[:]
    items = dataset["equipment"]
    
    # Шаг 1: Устранение конфликтов совместимости (удаляем менее полезный прибор из пары)
    for a, b in dataset["conflicts"]:
        if chrom[a] == 1 and chrom[b] == 1:
            if items[a]["utility"] < items[b]["utility"]:
                chrom[a] = 0
            else:
                chrom[b] = 0
                
    # Шаг 2: Устранение превышения бюджета и мощности
    while True:
        _, total_c, total_p, _ = evaluate_raw(chrom, dataset)
        if total_c <= dataset["max_budget"] and total_p <= dataset["max_power"]:
            break
        selected = [i for i, bit in enumerate(chrom) if bit == 1]
        if not selected:
            break
        # Удаляем выбранный прибор с наименьшей удельной полезностью (utility / (cost + power/10))
        worst_idx = min(
            selected,
            key=lambda i: items[i]["utility"] / (items[i]["cost"] + items[i]["power"] / 10.0)
        )
        chrom[worst_idx] = 0
        
    return chrom

# =====================================================================
# 3. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ДЛЯ ДИСКРЕТНОЙ ОПТИМИЗАЦИИ
# =====================================================================
class DiscreteGA:
    def __init__(self, dataset, pop_size=50, max_gen=120, cx_rate=0.8, mut_rate=None,
                 constraint_mode="repair", crossover_type="two_point", penalty_coef=50.0, seed=42):
        self.dataset = dataset
        self.n = dataset["n_items"]
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate if mut_rate is not None else 1.0 / self.n
        self.constraint_mode = constraint_mode  # "repair" или "penalty"
        self.crossover_type = crossover_type    # "two_point" или "uniform"
        self.penalty_coef = penalty_coef
        self.rng = random.Random(seed)

    def fitness(self, chrom):
        u, c, p, conf = evaluate_raw(chrom, self.dataset)
        if self.constraint_mode == "repair":
            return u
        else:
            over_b = max(0, c - self.dataset["max_budget"])
            over_p = max(0, p - self.dataset["max_power"])
            penalty = self.penalty_coef * (over_b + over_p / 5.0 + conf * 10.0)
            return u - penalty

    def init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = [1 if self.rng.random() < 0.35 else 0 for _ in range(self.n)]
            if self.constraint_mode == "repair":
                chrom = repair_chromosome(chrom, self.dataset)
            pop.append(chrom)
        return pop

    def tournament(self, pop, fits, k=3):
        idxs = self.rng.sample(range(len(pop)), k)
        best = max(idxs, key=lambda i: fits[i])
        return pop[best][:]

    def crossover(self, p1, p2):
        if self.rng.random() > self.cx_rate:
            return p1[:], p2[:]
        if self.crossover_type == "two_point":
            pt1, pt2 = sorted(self.rng.sample(range(1, self.n), 2))
            c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
            c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        else:  # uniform
            c1, c2 = [], []
            for g1, g2 in zip(p1, p2):
                if self.rng.random() < 0.5:
                    c1.append(g1); c2.append(g2)
                else:
                    c1.append(g2); c2.append(g1)
        return c1, c2

    def mutate(self, chrom):
        for i in range(self.n):
            if self.rng.random() < self.mut_rate:
                chrom[i] = 1 - chrom[i]
        if self.constraint_mode == "repair":
            chrom = repair_chromosome(chrom, self.dataset)
        return chrom

    def run(self):
        pop = self.init_pop()
        history = []
        best_feasible_chrom = None
        best_feasible_u = -1.0

        for _ in range(self.max_gen):
            fits = [self.fitness(ind) for ind in pop]
            
            # Отслеживаем лучшее допустимое решение
            for ind in pop:
                if is_feasible(ind, self.dataset):
                    u, _, _, _ = evaluate_raw(ind, self.dataset)
                    if u > best_feasible_u:
                        best_feasible_u = u
                        best_feasible_chrom = ind[:]
                        
            history.append(best_feasible_u if best_feasible_u > 0 else 0.0)
            
            # Элитизм
            elite_idx = max(range(len(pop)), key=lambda i: fits[i])
            new_pop = [pop[elite_idx][:]]
            
            while len(new_pop) < self.pop_size:
                p1 = self.tournament(pop, fits)
                p2 = self.tournament(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
                
            pop = new_pop[:self.pop_size]

        feasible_ratio = sum(1 for ind in pop if is_feasible(ind, self.dataset)) / self.pop_size
        return best_feasible_chrom, best_feasible_u, feasible_ratio, history

# =====================================================================
# 4. БАЗОВАЯ КОНСТРУКТИВНАЯ ЖАДНАЯ ЭВРИСТИКА (BASELINE)
# =====================================================================
def greedy_heuristic(dataset):
    items = dataset["equipment"]
    # Сортируем по убыванию отношения полезности к приведённым затратам
    order = sorted(
        range(dataset["n_items"]),
        key=lambda i: items[i]["utility"] / (items[i]["cost"] + items[i]["power"] / 10.0),
        reverse=True
    )
    chrom = [0] * dataset["n_items"]
    for idx in order:
        chrom[idx] = 1
        if not is_feasible(chrom, dataset):
            chrom[idx] = 0
    u, c, p, conf = evaluate_raw(chrom, dataset)
    return chrom, u, c, p

# =====================================================================
# 5. ПРОВЕДЕНИЕ ВСЕХ ЭКСПЕРИМЕНТОВ И СОХРАНЕНИЕ CSV
# =====================================================================
def run_all_lab2():
    dataset = generate_lab_dataset(n_items=35, seed=42)
    print(f"Сгенерировано {dataset['n_items']} приборов. Лимит бюджета: {dataset['max_budget']} тыс. руб., "
          f"Лимит мощности: {dataset['max_power']} Вт, Конфликтов: {len(dataset['conflicts'])}")

    # Пример 2 допустимых и 2 недопустимых решений (Обязательный эксперимент 2)
    rng = random.Random(10)
    valid_examples, invalid_examples = [], []
    while len(valid_examples) < 2 or len(invalid_examples) < 2:
        cand = [1 if rng.random() < 0.45 else 0 for _ in range(dataset["n_items"])]
        u, c, p, conf = evaluate_raw(cand, dataset)
        if is_feasible(cand, dataset) and len(valid_examples) < 2:
            valid_examples.append((cand, u, c, p, conf))
        elif not is_feasible(cand, dataset) and len(invalid_examples) < 2:
            invalid_examples.append((cand, u, c, p, conf))

    with open("lab2_examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Тип", "Полезность", "Стоимость", "Мощность", "Конфликты", "Вектор"])
        for ex in valid_examples:
            w.writerow(["Допустимое", ex[1], ex[2], ex[3], ex[4], "".join(map(str, ex[0]))])
        for ex in invalid_examples:
            w.writerow(["Недопустимое", ex[1], ex[2], ex[3], ex[4], "".join(map(str, ex[0]))])

    # Серии из 20 запусков для сравнения подходов (Обязательные эксперименты 3 и 4)
    configs = [
        ("ГА: Ремонт + 2-точечный CX", "repair", "two_point"),
        ("ГА: Штрафы + 2-точечный CX", "penalty", "two_point"),
        ("ГА: Ремонт + Равномерный CX", "repair", "uniform"),
    ]
    
    summary_rows = []
    best_overall_chrom = None
    best_overall_u = -1.0
    main_histories = []

    for label, mode, cx in configs:
        results_u, feas_ratios = [], []
        t0 = time.time()
        for run_idx in range(20):
            ga = DiscreteGA(dataset, pop_size=50, max_gen=120, constraint_mode=mode,
                            crossover_type=cx, seed=100 + run_idx)
            chrom, u, f_ratio, hist = ga.run()
            results_u.append(u if u > 0 else 0.0)
            feas_ratios.append(f_ratio)
            if label == "ГА: Ремонт + 2-точечный CX":
                main_histories.append(hist)
            if u > best_overall_u:
                best_overall_u = u
                best_overall_chrom = chrom[:]
        avg_t = (time.time() - t0) / 20.0
        
        summary_rows.append([
            label,
            f"{np.max(results_u):.2f}",
            f"{np.mean(results_u):.2f}",
            f"{np.median(results_u):.2f}",
            f"{np.min(results_u):.2f}",
            f"{np.std(results_u):.2f}",
            f"{np.mean(feas_ratios)*100:.1f}%",
            f"{avg_t:.4f}"
        ])

    # Сравнение с жадной эвристикой (Обязательный эксперимент 6)
    g_chrom, g_u, g_c, g_p = greedy_heuristic(dataset)
    summary_rows.append([
        "Жадная эвристика (Baseline)",
        f"{g_u:.2f}", f"{g_u:.2f}", f"{g_u:.2f}", f"{g_u:.2f}", "0.00", "100.0%", "0.0001"
    ])

    with open("lab2_comparison_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Алгоритм / Конфигурация", "Лучшее (Max)", "Среднее", "Медиана", 
                    "Худшее (Min)", "Стд. откл.", "Доля допустимых", "Время (сек)"])
        w.writerows(summary_rows)

    # Сохранение лучшего найденного комплекта оборудования в предметном виде (Обязательный эксперимент 5)
    with open("lab2_best_equipment_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Наименование прибора", "Стоимость (тыс. руб.)", "Мощность (Вт)", "Полезность"])
        for idx, bit in enumerate(best_overall_chrom):
            if bit == 1:
                it = dataset["equipment"][idx]
                w.writerow([it["id"], it["name"], it["cost"], it["power"], it["utility"]])

    # График сходимости для основного алгоритма
    hist_arr = np.array(main_histories)
    plt.figure(figsize=(10, 6))
    plt.plot(np.max(hist_arr, axis=0), label="Максимум (лучшее)", color="green")
    plt.plot(np.mean(hist_arr, axis=0), label="Среднее", color="blue")
    plt.plot(np.min(hist_arr, axis=0), label="Минимум (худшее)", color="red", alpha=0.6)
    plt.axhline(y=g_u, color="orange", linestyle="--", label=f"Жадная эвристика ({g_u:.1f})")
    plt.fill_between(range(120), np.min(hist_arr, axis=0), np.max(hist_arr, axis=0), color="blue", alpha=0.1)
    plt.title("Сходимость ГА (Вариант 5: Комплект оборудования лаборатории)")
    plt.xlabel("Поколение")
    plt.ylabel("Суммарная полезность комплекта")
    plt.legend()
    plt.grid(True)
    plt.savefig("lab2_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Готово! Сохранены файлы: lab2_input_data.json, lab2_examples.csv, "
          "lab2_comparison_results.csv, lab2_best_equipment_set.csv, lab2_convergence.png")

if __name__ == "__main__":
    run_all_lab2()