import numpy as np
import matplotlib.pyplot as plt

# 1. Problemin Tanımı (Rastrigin Fonksiyonu) [1, 2]
def fitness_function(x):
    # Kaynaklarda n-boyutlu veriliyor, burada 2D görselleştirme için n=2
    return 10 * 2 + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 2. FSS Parametreleri [3, 4]
n_fish = 30
dimensions = 2
iterations = 100
w_scale = 5000 # Maksimum ağırlık [3]
step_ind_initial = 0.1 # Bireysel adım [4]
step_vol_initial = 0.01 # İradi adım [4]
aquarium_bound = 5.12 # Rastrigin için arama uzayı [2]

# 3. Başlatma (Initialization) [5]
pos = np.random.uniform(-aquarium_bound, aquarium_bound, (n_fish, dimensions))
weights = np.ones(n_fish) * (w_scale / 2) # Tüm balıklar w_scale/2 ile başlar [3]
prev_fitness = np.array([fitness_function(p) for p in pos])

# Görselleştirme hazırlığı
plt.ion()
fig, ax = plt.subplots()

for iteration in range(iterations):
    # Adım boyutlarını iterasyonla doğrusal azaltma [6, 7]
    step_ind = step_ind_initial * (1 - iteration / iterations)
    step_vol = step_vol_initial * (1 - iteration / iterations)
    
    # --- A. BİREYSEL HAREKET (Individual Movement) [6, 8] ---
    new_pos = pos + np.random.uniform(-1, 1, (n_fish, dimensions)) * step_ind
    # Sınır kontrolü
    new_pos = np.clip(new_pos, -aquarium_bound, aquarium_bound)
    
    new_fitness = np.array([fitness_function(p) for p in new_pos])
    
    # Sadece daha iyiyse hareket et
    improved_idx = new_fitness < prev_fitness
    pos[improved_idx] = new_pos[improved_idx]
    
    # --- B. BESLENME OPERATÖRÜ (Feeding) [3, 9] ---
    delta_f = -(new_fitness - prev_fitness) # Minimizasyon için tersini alıyoruz
    max_delta_f = np.max(np.abs(delta_f)) if np.max(np.abs(delta_f)) != 0 else 1
    weights = weights + (delta_f / max_delta_f)
    weights = np.clip(weights, 1, w_scale) # Ağırlık 1 ile Wscale arasında [3]
    
    # --- C. KOLEKTİF-İÇGÜDÜSEL HAREKET [10, 11] ---
    if np.sum(delta_f) != 0:
        instinctive_drift = np.sum((new_pos - pos) * delta_f[:, np.newaxis], axis=0) / np.sum(delta_f)
        pos += instinctive_drift
    
    # --- D. KOLEKTİF-İRADİ HAREKET [7, 12, 13] ---
    # Ağırlık merkezini (barycenter) hesapla
    barycenter = np.sum(pos * weights[:, np.newaxis], axis=0) / np.sum(weights)
    
    total_weight_change = np.sum(delta_f)
    for i in range(n_fish):
        dist_to_barycenter = pos[i] - barycenter
        if total_weight_change > 0: # Sürü kilo aldıysa DARAL [7]
            pos[i] -= step_vol * np.random.rand() * dist_to_barycenter
        else: # Sürü kilo kaybettiyse GENİŞLE [7]
            pos[i] += step_vol * np.random.rand() * dist_to_barycenter
            
    prev_fitness = np.array([fitness_function(p) for p in pos])

    # Görselleştirme Güncelleme
    ax.clear()
    ax.set_xlim(-aquarium_bound, aquarium_bound)
    ax.set_ylim(-aquarium_bound, aquarium_bound)
    ax.scatter(pos[:, 0], pos[:, 1], s=weights/50, c='blue', alpha=0.6) # Ağırlığa göre boyut
    ax.set_title(f"İterasyon: {iteration} - En İyi Skor: {np.min(prev_fitness):.4f}")
    plt.pause(0.5)

plt.ioff()
plt.show()


