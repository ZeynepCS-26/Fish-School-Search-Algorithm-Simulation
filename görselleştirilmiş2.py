import numpy as np
import matplotlib.pyplot as plt
# 1. Rastrigin Fonksiyonu (Kaynak [5], Denklem 9)
def fitness_rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 2. Parametreler (Kaynak [4, 6])
n_fish = 30
dims = 2
iterations = 100
w_scale = 5000
lower_bound, upper_bound = -5.12, 5.12
step_ind_init = 0.1 * (upper_bound - lower_bound)
step_vol_init = 0.1 * (upper_bound - lower_bound)

# Balıklar optimumdan uzak bölgede başlatılır (Kaynak [4])
pos = np.random.uniform(2.56, 5.12, (n_fish, dims))
weights = np.ones(n_fish) * (w_scale / 2) # Başlangıç ağırlığı Wscale/2 (Kaynak [7])
prev_fitness = np.array([fitness_rastrigin(p) for p in pos])

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for t in range(iterations):
    # Adım boyutlarını azaltma (Kaynak [2, 3])
    step_ind = step_ind_init * (1 - t / iterations)
    step_vol = step_vol_init * (1 - t / iterations)
    
    # --- Bireysel Hareket ---
    new_pos = pos + np.random.uniform(-1, 1, (n_fish, dims)) * step_ind
    new_pos = np.clip(new_pos, lower_bound, upper_bound)
    new_fitness = np.array([fitness_rastrigin(p) for p in new_pos])
    
    improved = new_fitness < prev_fitness
    pos_diff = np.zeros_like(pos)
    pos_diff[improved] = new_pos[improved] - pos[improved]
    
    # --- Beslenme (Feeding) - Ağırlık Güncelleme (Kaynak [7], Denklem 1) ---
    delta_f = -(new_fitness - prev_fitness)
    max_delta_f = np.max(np.abs(delta_f)) if np.max(np.abs(delta_f)) != 0 else 1
    weights += (delta_f / max_delta_f)
    weights = np.clip(weights, 1, w_scale)
    
    pos[improved] = new_pos[improved]
    prev_fitness[improved] = new_fitness[improved]
    
    # --- Kolektif-İradi Hareket (Kaynak [3, 8]) ---
    barycenter = np.sum(pos * weights[:, np.newaxis], axis=0) / np.sum(weights)
    total_w_change = np.sum(delta_f)
    
    for i in range(n_fish):
        dist = pos[i] - barycenter
        # Başarı varsa daral, yoksa genişle (Denklem 4 & 5)
        if total_w_change > 0:
            pos[i] -= step_vol * np.random.rand() * dist
        else:
            pos[i] += step_vol * np.random.rand() * dist

    # Grafik Güncelleme
    ax.clear()
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    # Balıkların büyüklüğü ağırlıklarını temsil eder (Kaynak [9], Figür 2)
    ax.scatter(pos[:, 0], pos[:, 1], s=weights/5, c='blue', alpha=0.6, edgecolors='black')
    ax.scatter(0, 0, c='red', marker='X', s=200, label='Global Optimum (0,0)')
    ax.set_title(f"İterasyon: {t+1} | Durum: {'Daralma' if total_w_change > 0 else 'Genişleme'}\n3 saniye bekleniyor...")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.draw()
    plt.pause(1) # 1 saniyelik bekleme

plt.ioff()
plt.show()