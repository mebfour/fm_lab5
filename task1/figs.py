import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "task1/figures"

# Настройка шрифтов
plt.rcParams.update({
    "font.size": 16,          # базовый размер
    "axes.titlesize": 18,     # заголовок графика
    "axes.labelsize": 16,     # подписи осей
    "xtick.labelsize": 14,    # подписи X
    "ytick.labelsize": 14,    # подписи Y
    "legend.fontsize": 16,    
    "figure.titlesize": 18
})

T = 5
V = 10
dt = 0.08
dnu = 0.08


# Π(t)
def pi_func(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)

# аналитический Фурье-образ
def pi_hat_analytic(nu):
    return np.sinc(nu)  

# графики для пункта 1.2
t = np.linspace(-2, 2, 1000)
nu = np.linspace(-10, 10, 1000)

fig1 = plt.figure()
plt.plot(t, pi_func(t))
plt.title("Функция Π(t)")                     # Заголовок на русском
plt.xlabel("Время t")                         # Подпись оси X
plt.ylabel("Значение Π(t)")                   # Подпись оси Y
plt.grid()
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, "Pi_1.png"), dpi=300)
plt.close(fig1)

fig2 = plt.figure()
plt.plot(nu, pi_hat_analytic(nu))
plt.title("Аналитический")   
plt.xlabel("Частота ν")                               # Подпись оси X
plt.ylabel("Значение образа")            # Подпись оси Y
plt.grid()
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, "Pi_hat_1.png"), dpi=300)
plt.close(fig2)
# ----------------------------------------------------------------------------------------------------

# ПУНКТ 1.3
# прямое преобразование Фурье с помощью трапеций
def fourier_trapz(t, f, nu):
    F = []
    for w in nu:
        integrand = f * np.exp(-2j * np.pi * w * t)
        F.append(np.trapz(integrand, t))
    return np.array(F)

def inverse_fourier_trapz(nu, F, t):
    f_rec = []
    for ti in t:
        integrand = F * np.exp(2j * np.pi * nu * ti)
        f_rec.append(np.trapz(integrand, nu))
    return np.array(f_rec) 

# пример расчета


t = np.arange(-T, T + dt, dt)
nu = np.linspace(-V, V, int(2*V/dnu) + 1)

f = pi_func(t)

F_num = fourier_trapz(t, f, nu)
f_rec = inverse_fourier_trapz(nu, F_num, t)



N = len(t)
# dt = t[1] - t[0]
# dnu = 1 / (N * dt)



nu_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))


from itertools import product

T_values = [1, 5, 20]
dt_values = [0.0001, 0.1, 0.3]
V_values = [2, 10]
dnu_values = [0.01, 1]

param_sets = [
    {"T": T, "dt": dt, "V": V, "dnu": dnu}
    for T, dt, V, dnu in product(T_values, dt_values, V_values, dnu_values)
]


prefix = "1_3"

for i, p in enumerate(param_sets):
    break
    if i in [0, 3,5, 12,14, 15,17,18,19,24,25,27,29,30,31,34]:
        continue  # Выбираем 6 наборов для графиков
    t_fine = np.linspace(-T, T, 5000)
    f_fine = pi_func(t_fine)
    T, dt, V, dnu = p["T"], p["dt"], p["V"], p["dnu"]

    nu_dense = np.linspace(-V, V, 1000)
    F_analytic_dense = pi_hat_analytic(nu_dense)

    # сетки

    t = np.arange(-T, T + dt, dt)
    nu = np.arange(-V, V + dnu, dnu)
    f = pi_func(t)

    # прямое и обратное через trapz
    F_num = fourier_trapz(t, f, nu)
    f_rec = inverse_fourier_trapz(nu, F_num, t)


        # --------- ГРАФИК 1: спектр ----------
    fig1 = plt.figure(figsize=(14, 6))
 
    plt.plot(nu_dense, F_analytic_dense, label="Аналитический (sinc)")
    plt.plot(nu, np.real(F_num), '--', label="Численный (trapz)")
    plt.title(
        f"Спектр сигнала Π(t)\n"
        f"T = {T}, Δt = {dt}, V = {V}"
    )

    plt.xlabel("Частота ν")
    plt.ylabel("Амплитуда")
    plt.legend(loc="upper right")   
    plt.grid()

    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_spectrum.png"), dpi=300)
    plt.close(fig1)  # Освобождаем память

   # --------- ГРАФИК 2: восстановление ----------
    fig2 = plt.figure(figsize=(14, 6))
    plt.plot(t_fine, f_fine, label="Исходный сигнал Π(t)")
    plt.plot(t, np.real(f_rec), '--', label="Восстановленный сигнал")

    plt.title(
        f"Восстановление сигнала\n"
        f"Δt = {dt}, Δν = {dnu}, T = {T}, V = {V}"
    )

    plt.xlabel("Время t")
    plt.ylabel("Ампплитуда")
    plt.legend(loc="upper right")
    plt.grid()

    fig2.tight_layout()
    
    fig2.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_reconstruction.png"), dpi=300)
    plt.close(fig2)  # Освобождаем память

    print(f"Сохранено: 1_3_set_{i+1}")




# КОД ДЛЯ КРИВЫХ ПАРАМЕТРОВ И ВСТАВКИ В TEX

def figs_1_3(T, dt, V, dnu, i):
    t_fine = np.linspace(-1, 1, 5000)
    f_fine = pi_func(t_fine)
    T, dt, V, dnu = p["T"], p["dt"], p["V"], p["dnu"]

    nu_dense = np.linspace(-V, V, 1000)
    F_analytic_dense = pi_hat_analytic(nu_dense)

    # сетки

    t = np.arange(-T, T + dt, dt)
    nu = np.arange(-V, V + dnu, dnu)
    f = pi_func(t)

    # прямое и обратное через trapz
    F_num = fourier_trapz(t, f, nu)
    f_rec = inverse_fourier_trapz(nu, F_num, t)


        # --------- ГРАФИК 1: спектр ----------
    fig1 = plt.figure(figsize=(14, 6))
 
    plt.plot(nu_dense, F_analytic_dense, label="Аналитический (sinc)")
    plt.plot(nu, np.real(F_num), '--', label="Численный (trapz)")
    plt.title(
        f"Спектр сигнала Π(t)\n"
        f"T = {T}, Δt = {dt}, V = {V}"
    )

    plt.xlabel("Частота ν")
    plt.ylabel("Амплитуда")
    plt.legend(loc="upper right")   
    plt.grid()

    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_spectrum.png"), dpi=300)
    plt.close(fig1)  # Освобождаем память

   # --------- ГРАФИК 2: восстановление ----------
    fig2 = plt.figure(figsize=(14, 6))
    plt.plot(t_fine, f_fine, label="Исходный сигнал Π(t)")
    plt.plot(t, np.real(f_rec), '--', label="Восстановленный сигнал")

    plt.title(
        f"Восстановление сигнала\n"
        f"Δt = {dt}, Δν = {dnu}, T = {T}, V = {V}"
    )

    plt.xlabel("Время t")
    plt.ylabel("Ампплитуда")
    plt.legend(loc="upper right")
    plt.grid()

    fig2.tight_layout()
    
    fig2.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_reconstruction.png"), dpi=300)
    plt.close(fig2)  # Освобождаем память

    print(f"Сохранено: 1_3_set_{i+1}")

#figs_1_3(1,0.0001,2,0.01,0) #1
#figs_1_3(5,0.0001,2,0.01,12) #13


'''
for i, p in enumerate(param_sets):
    T, dt, V, dnu = p["T"], p["dt"], p["V"], p["dnu"]

    spectrum_file = f"{prefix}_set_{i+1}_spectrum.png"
    recon_file = f"{prefix}_set_{i+1}_reconstruction.png"

    tex_file = os.path.join(output_dir, f"set_{i+1}.tex")

    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(f"""
\\subsection*{{Параметры: $T={T}$, $\\Delta t={dt}$, $V={V}$, $\\Delta \\nu={dnu}$}}



\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{task1/figures/{spectrum_file}}}
    \\caption{{Спектр сигнала}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{task1/figures/{recon_file}}}
    \\caption{{Восстановление сигнала}}
\\end{{figure}}
""")
'''

# #  пункт 1.5 - DFT
# def fft_unitary(f, dt):
#     return np.fft.fftshift(np.fft.fft(f)) * dt

# def ifft_unitary(F, dt):
#     return np.fft.ifft(np.fft.ifftshift(F)) / dt


# def rect_f(t):
#     return np.where(np.abs(t) <= 0.5, 1.0, 0.0)

# T = 5
# dt = 0.01
# t = np.arange(-T, T + dt, dt)
# N = len(t)
# dnu = 1 / (N * dt)
# nu_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))
# f = pi_func(t)


# F_fft = fft_unitary(f, dt)
# f_rec_fft = ifft_unitary(F_fft, dnu)

# T_list = [5, 45, 15, 15, 15]
# dt_list = [0.05, 0.05, 0.05, 0.02, 0.01]
# from itertools import product

# T_values = [1, 5, 20]
# dt_values = [0.0001, 0.1, 0.3]
# V_values = [2, 10]
# dnu_values = [0.01, 1]

# param_sets = [

#     # =========================
#     # 1. Исследование T и dt
#     # (V, dnu фиксированы)
#     # =========================
#     {"group": "time", "T": 3,  "dt": 0.3,  "V": 10, "dnu": 0.01},
#     {"group": "time", "T": 10,  "dt": 0.3,  "V": 10, "dnu": 0.01},
#     {"group": "time", "T": 3,  "dt": 0.05, "V": 10, "dnu": 0.01},
#     {"group": "time", "T": 20, "dt": 0.01, "V": 10, "dnu": 0.01},

#     # =========================
#     # 2. Исследование V и dnu
#     # (T, dt фиксированы)
#     # =========================
#     {"group": "freq", "T": 3, "dt": 0.05, "V": 2,  "dnu": 1},
#     {"group": "freq", "T": 3, "dt": 0.05, "V": 10, "dnu": 1},
#     {"group": "freq", "T": 3, "dt": 0.05, "V": 10, "dnu": 0.1},
#     {"group": "freq", "T": 3, "dt": 0.05, "V": 20, "dnu": 0.01},
# ]


# for i, p in enumerate(param_sets):

#     T, dt, V, dnu = p["T"], p["dt"], p["V"], p["dnu"]
#     group = p["group"]
    
#     t = np.arange(-T, T + dt, dt)
#     f = rect_f(t)

#     prefix = f"{group}_{i+1}"

#     N = len(t)
#     nu_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))


#     nu_min, nu_max = nu_fft[0], nu_fft[-1]
#     F_fft = fft_unitary(f, dt)
#     f_rec = ifft_unitary(F_fft, dt)
    
#     # Используем тот же шаг, что и у FFT
#     dnu_fft = nu_fft[1] - nu_fft[0]
#     nu_analytic = np.arange(nu_min, nu_max + dnu_fft, dnu_fft)
#     F_analytic = T * np.sinc(T * nu_analytic)



#     nu_dense = np.linspace(nu_min, nu_max, 1000)

#     F_analytic_dense = T * np.sinc(T * nu_dense)


#     plt.figure(figsize=(14, 6))
#     F_analytic_interp = np.interp(nu_fft, nu_analytic, F_analytic)
#     plt.plot(nu_dense, F_analytic_dense, label="Аналитический (sinc)")
#     plt.plot(nu_fft, np.real(F_fft), '--', label="Численный (FFT)")
    
    


#     plt.title(
#         f"Спектр сигнала Π(t)\n"
#         f"T = {T}, Δt = {dt}, V = {V},  Δν = {dnu}"
#     )

#     plt.xlabel("Частота ν")
#     plt.ylabel("Амплитуда F(ν)")
#     plt.grid(alpha=0.4)
#     plt.legend(loc="upper right")

#     spectrum_file = f"1_5_set_{i+1}_spectrum.png"
#     plt.savefig(os.path.join("task1/figs_1_5", spectrum_file), dpi=300)
#     plt.close()


#     # =====================
#     # ВОССТАНОВЛЕНИЕ
#     # =====================

#     t_fine = np.linspace(-T, T, 5000)
#     f_fine = pi_func(t_fine)
    
#     plt.figure(figsize=(14, 6))

#     plt.plot(t_fine, f_fine, 'r', label="Исходный сигнал Π(t)")
#     plt.plot(t, np.real(f_rec), 'g--', label="Восстановленный (FFT)")

#     plt.title(
#         f"Восстановление сигнала\n"
#         f"T = {T}, Δt = {dt}, V = {V},  Δν = {dnu}"
#     )

#     plt.xlabel("Время t")
#     plt.ylabel("Амплитуда")
#     plt.grid(alpha=0.4)
#     plt.legend(loc="upper right")

#     recon_file = f"1_5_set_{i+1}_reconstruction.png"
#     plt.savefig(os.path.join("task1/figs_1_5", recon_file), dpi=300)
#     plt.close()

#     print(f"Сохранено: 1__set_{i+1}")

#     tex_file = os.path.join("task1/figs_1_5", f"set_{i+1}.tex")

#     with open(tex_file, "a", encoding="utf-8") as f:
#         f.write(f"""
#     \\subsection*{{Параметры: $T={T}$, $\\Delta t={dt}$, $V={V}$, $\\Delta \\nu={dnu}$}}

#     \\begin{{figure}}[H]
#         \\centering
#         \\includegraphics[width=0.8\\textwidth]{{task1/figs_1_5/{spectrum_file}}}
#         \\caption{{Спектр сигнала $\\Pi(t)$ (численный FFT и аналитический sinc)}}
#     \\end{{figure}}

#     \\begin{{figure}}[H]
#         \\centering
#         \\includegraphics[width=0.8\\textwidth]{{task1/figs_1_5/{recon_file}}}
#         \\caption{{Восстановление сигнала $\\Pi(t)$ методом FFT}}
#     \\end{{figure}}
#     """)
    