import numpy as np

# 1. Problem Tanımı: Rastrigin Fonksiyonu (30 Boyut) [1, 3]
def fitness_rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 2. FSS Parametreleri [2, 4, 5]
n_fish = 30
dimensions = 30
iterations = 10000
w_scale = 5000  # [2]
step_ind_init = 0.1 * (5.12 - (-5.12)) # Arama alanının %10'u [2]
step_ind_final = 0.001 * (5.12 - (-5.12))
step_vol_init = 0.1 * (5.12 - (-5.12))
step_vol_final = 0.01 * (5.12 - (-5.12))
lower_bound, upper_bound = -5.12, 5.12

# 3. Başlatma (Initialization) [4, 6, 7]
# Balıklar optimumdan uzak bölgede başlatılır [4]
pos = np.random.uniform(2.56, 5.12, (n_fish, dimensions))
weights = np.ones(n_fish) * (w_scale / 2) # Tüm balıklar Wscale/2 ile doğar [6]
prev_fitness = np.array([fitness_rastrigin(p) for p in pos])

print(f"Başlangıç En İyi Fitness: {np.min(prev_fitness):.4f}")

# Ana Döngü (FSS Cycle) [7, 8]
for iteration in range(iterations):
    # Adım boyutlarının doğrusal azaltılması [9, 10]
    step_ind = step_ind_init - (step_ind_init - step_ind_final) * (iteration / iterations)
    step_vol = step_vol_init - (step_vol_init - step_vol_final) * (iteration / iterations)

    # --- A. BİREYSEL HAREKET (Individual Movement) [9, 11] ---
    # Rastgele yön seçilir ve step_ind ile çarpılır
    random_move = np.random.uniform(-1, 1, (n_fish, dimensions)) * step_ind
    new_pos = pos + random_move
    new_pos = np.clip(new_pos, lower_bound, upper_bound)
    
    new_fitness = np.array([fitness_rastrigin(p) for p in new_pos])
    
    # Sadece yiyecek miktarı artarsa hareket gerçekleşir [9, 11]
    improved_idx = new_fitness < prev_fitness
    pos_diff = np.zeros_like(pos)
    pos_diff[improved_idx] = new_pos[improved_idx] - pos[improved_idx]
    pos[improved_idx] = new_pos[improved_idx]
    
    # --- B. BESLENME OPERATÖRÜ (Feeding) [6, 12] ---
    # Formül (1): Ağırlık değişimi fitness farkına oranlanır
    fitness_diff = -(new_fitness - prev_fitness) # Minimizasyon için
    max_delta_f = np.max(np.abs(fitness_diff)) if np.max(np.abs(fitness_diff)) != 0 else 1
    weights = weights + (fitness_diff / max_delta_f)
    weights = np.clip(weights, 1, w_scale) # 1 ile Wscale arası sınır [6]

    # --- C. KOLEKTİF-İÇGÜDÜSEL HAREKET [13, 14] ---
    # Formül (2): Başarılı balıkların hareketlerinin ağırlıklı ortalaması
    sum_fitness_diff = np.sum(fitness_diff)
    if sum_fitness_diff != 0:
        instinctive_drift = np.sum(pos_diff * fitness_diff[:, np.newaxis], axis=0) / sum_fitness_diff
        pos += instinctive_drift

    # --- D. KOLEKTİF-İRADİ HAREKET [10, 15, 16] ---
    # Formül (3): Ağırlık merkezinin (Barycenter) hesaplanması
    barycenter = np.sum(pos * weights[:, np.newaxis], axis=0) / np.sum(weights)
    
    total_weight_change = np.sum(fitness_diff)
    for i in range(n_fish):
        dist_to_barycenter = pos[i] - barycenter
        # Başarı varsa DARAL (Formül 4), başarısızlık varsa GENİŞLE (Formül 5) [10]
        if total_weight_change > 0:
            pos[i] -= step_vol * np.random.rand() * dist_to_barycenter
        else:
            pos[i] += step_vol * np.random.rand() * dist_to_barycenter
            
    prev_fitness = np.array([fitness_rastrigin(p) for p in pos])

    if (iteration + 1) % 1000 == 0:
        print(f"İterasyon {iteration+1}/{iterations} - En İyi Fitness: {np.min(prev_fitness):.4f}")

print(f"\nFinal En İyi Fitness: {np.min(prev_fitness):.6f}")
