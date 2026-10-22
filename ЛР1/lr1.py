import numpy as np
import matplotlib.pyplot as plt
import random
import time
import csv

def griewank(x):
    """Функция Гриванка для d-мерного вектора."""
    x = np.array(x)
    d = len(x)
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return 1 + sum_term - prod_term

class GeneticAlgorithm:
    def __init__(self, d=10, bounds=(-600, 600), pop_size=50, max_gen=100, 
                 mut_rate=0.1, mut_scale=10.0, seed=None):
        self.d = d
        self.bounds = bounds
        self.pop_size = max(30, pop_size) # Не менее 30 особей согласно требованиям
        self.max_gen = max_gen
        self.mut_rate = mut_rate
        self.mut_scale = mut_scale
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def init_population(self):
        """Инициализация в допустимой области."""
        return [np.random.uniform(self.bounds[0], self.bounds[1], self.d) for _ in range(self.pop_size)]

    def clamp(self, x):
        """Обработка выхода за границы области поиска."""
        return np.clip(x, self.bounds[0], self.bounds[1])

    def tournament_selection(self, population, fitnesses, k=3):
        """Турнирная селекция."""
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_idx = selected_indices[np.argmin([fitnesses[i] for i in selected_indices])]
        return population[best_idx]

    def crossover(self, p1, p2):
        """Арифметический кроссинговер для вещественных векторов."""
        alpha = random.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return self.clamp(c1), self.clamp(c2)

    def mutate(self, ind):
        """Гауссовская мутация вещественных параметров."""
        mutated = ind.copy()
        for i in range(self.d):
            if random.random() < self.mut_rate:
                mutated[i] += np.random.normal(0, self.mut_scale)
        return self.clamp(mutated)

    def run(self):
        population = self.init_population()
        best_history = []
        
        # Эволюционный цикл
        for gen in range(self.max_gen):
            fitnesses = [griewank(ind) for ind in population]
            
            # Элитизм: сохранение лучшего решения
            best_idx = np.argmin(fitnesses)
            best_ind = population[best_idx]
            best_history.append(fitnesses[best_idx])
            
            new_population = [best_ind.copy()]
            
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                
                c1, c2 = self.crossover(p1, p2)
                
                new_population.extend([self.mutate(c1), self.mutate(c2)])
                
            population = new_population[:self.pop_size]
            
        final_fitnesses = [griewank(ind) for ind in population]
        best_overall_idx = np.argmin(final_fitnesses)
        return population[best_overall_idx], final_fitnesses[best_overall_idx], best_history

def run_experiments():
    num_runs = 20 # Не менее 20 независимых запусков
    max_generations = 150
    pop_size = 50
    all_histories = []
    best_results = []
    
    print("Выполнение 20 независимых запусков ГА...")
    start_time = time.time() # Запускаем таймер
    
    for i in range(num_runs):
        ga = GeneticAlgorithm(pop_size=pop_size, seed=42+i, max_gen=max_generations)
        best_ind, best_fit, history = ga.run()
        all_histories.append(history)
        best_results.append(best_fit)
        
    end_time = time.time()
    avg_time_per_run = (end_time - start_time) / num_runs # Среднее время запуска
    
    all_histories = np.array(all_histories)
    
    # Расчет требуемых статистик
    best_val = np.min(best_results)
    worst_val = np.max(best_results)
    mean_val = np.mean(best_results)
    median_val = np.median(best_results)
    std_val = np.std(best_results)
    evals_per_run = pop_size * max_generations # Число вызовов фитнес-функции
    
    # Вывод результатов для таблицы в отчете
    print("\n--- Статистика по 20 запускам ---")
    print(f"Лучший результат:         {best_val:.6f}")
    print(f"Среднее значение:         {mean_val:.6f}")
    print(f"Медиана:                  {median_val:.6f}")
    print(f"Худший результат:         {worst_val:.6f}")
    print(f"Стандартное отклонение:   {std_val:.6f}")
    print(f"Обращений к фитнес-ф. :   {evals_per_run}")
    print(f"Среднее время запуска:    {avg_time_per_run:.4f} сек")

    with open('table1_statistics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика", "Значение"])
        writer.writerow(["Лучший результат", f"{best_val:.6f}"])
        writer.writerow(["Среднее значение", f"{mean_val:.6f}"])
        writer.writerow(["Медиана", f"{median_val:.6f}"])
        writer.writerow(["Худший результат", f"{worst_val:.6f}"])
        writer.writerow(["Стандартное отклонение", f"{std_val:.6f}"])
        writer.writerow(["Обращений к фитнес-ф.", evals_per_run])
        writer.writerow(["Среднее время запуска", f"{avg_time_per_run:.4f} сек"])
    
    # Построение графика
    min_hist = np.min(all_histories, axis=0)
    mean_hist = np.mean(all_histories, axis=0)
    max_hist = np.max(all_histories, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(min_hist, label='Минимум', color='green')
    plt.plot(mean_hist, label='Среднее', color='blue')
    plt.plot(max_hist, label='Максимум', color='red', alpha=0.5)
    plt.fill_between(range(max_generations), min_hist, max_hist, color='blue', alpha=0.1)
    plt.title('Траектория сходимости алгоритма (функция Гриванка)')
    plt.xlabel('Поколение')
    plt.ylabel('Значение целевой функции')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_random_search(evals_budget=7500, num_runs=20, d=10, bounds=(-600, 600)):
    best_results = []
    for run in range(num_runs):
        np.random.seed(100 + run)
        # Генерируем evals_budget случайных векторов в области поиска
        random_points = np.random.uniform(bounds[0], bounds[1], (evals_budget, d))
        scores = [griewank(pt) for pt in random_points]
        best_results.append(np.min(scores))
    
    print("\n=== Результаты случайного поиска (Random Search, бюджет 7500) ===")
    print(f"Лучший найденный: {np.min(best_results):.6f}")
    print(f"Средний результат: {np.mean(best_results):.6f}")
    print(f"Худший результат:  {np.max(best_results):.6f}")

    with open('table3_random_search.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Метод", "Бюджет вычислений", "Лучший результат", "Средний результат", "Худший результат"])
        writer.writerow(["Базовый ГА (Конфигурация 1)", evals_budget, "0.057848", "0.241737", "0.592798"])
        writer.writerow(["Случайный поиск (Random Search)", evals_budget, f"{np.min(best_results):.6f}", f"{np.mean(best_results):.6f}", f"{np.max(best_results):.6f}"])

    return best_results

def run_comparison_experiment():
    num_runs = 20
    max_generations = 150
    pop_size = 50
    # Меняем ровно один фактор: mut_rate повышен с 0.1 до 0.4
    alt_mut_rate = 0.4 
    
    best_results_alt = []
    for i in range(num_runs):
        ga = GeneticAlgorithm(
            d=10, 
            pop_size=pop_size, 
            max_gen=max_generations, 
            mut_rate=alt_mut_rate, 
            seed=42 + i
        )
        _, best_fit, _ = ga.run()
        best_results_alt.append(best_fit)
        
    print(f"\n=== Результаты Конфигурации 2 (mut_rate={alt_mut_rate}) ===")
    print(f"Лучший результат:         {np.min(best_results_alt):.6f}")
    print(f"Среднее значение:         {np.mean(best_results_alt):.6f}")
    print(f"Медиана:                  {np.median(best_results_alt):.6f}")
    print(f"Худший результат:         {np.max(best_results_alt):.6f}")
    print(f"Стандартное отклонение:   {np.std(best_results_alt):.6f}")

    with open('table2_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика", "Конфигурация 1 (mut_rate = 0.1)", "Конфигурация 2 (mut_rate = 0.4)"])
        writer.writerow(["Число обращений к фитнес-ф.", 7500, 7500])
        writer.writerow(["Лучший результат", "0.057848", f"{np.min(best_results_alt):.6f}"])
        writer.writerow(["Среднее значение", "0.241737", f"{np.mean(best_results_alt):.6f}"])
        writer.writerow(["Медиана", "0.207246", f"{np.median(best_results_alt):.6f}"])
        writer.writerow(["Худший результат", "0.592798", f"{np.max(best_results_alt):.6f}"])
        writer.writerow(["Стандартное отклонение", "0.128837", f"{np.std(best_results_alt):.6f}"])

    return best_results_alt

if __name__ == '__main__':
    run_experiments()
    run_random_search()
    run_comparison_experiment()
