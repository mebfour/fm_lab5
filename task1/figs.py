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
plt.title("Π(t)")
plt.grid()
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, "Pi_1.png"), dpi=300)
plt.close(fig1)  # Освобождаем память

fig2 = plt.figure()
plt.plot(nu, pi_hat_analytic(nu))
plt.title("Analytical Π̂(ν)")
plt.grid()
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, "Pi_hat_1.png"), dpi=300)
plt.close(fig2)  # Освобождаем память
# ----------------------------------------------------------------------------------------------------

# ПУНКТ 1.3
# прямое преобразование Фурье с помощью трапеций
def fourier_trapz(t, f, nu):
    dt = t[1] - t[0]
    F = []
    for w in nu:
        integrand = f * np.exp(-2j * np.pi * w * t)
        F.append(np.trapz(integrand, t))
    return np.array(F)

# обратное преобразование Фурье 
def inverse_fourier_trapz(nu, F, t):
    dnu = nu[1] - nu[0]
    f_rec = []
    for ti in t:
        integrand = F * np.exp(2j * np.pi * nu * ti)
        f_rec.append(np.trapz(integrand, nu))
    return np.array(f_rec)


# пример расчета
T = 5
V = 10
dt = 0.08
dnu = 0.08

t = np.arange(-T, T, dt)
nu = np.arange(-V, V, dnu)

f = pi_func(t)

F_num = fourier_trapz(t, f, nu)
f_rec = inverse_fourier_trapz(nu, F_num, t)



N = len(t)
dt = t[1] - t[0]
dnu = 1 / (N * dt)



nu_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))



#-------------------------------- перебор
param_sets = [
    {"T": 5, "dt": 0.1, "V": 10, "dnu": 0.2},   # очень грубо
    {"T": 5, "dt": 0.1, "V": 10, "dnu": 0.1},

    {"T": 20, "dt": 0.1, "V": 10, "dnu": 0.1},
    {"T": 5, "dt": 0.5, "V": 10, "dnu": 0.1},

    {"T": 5, "dt": 0.01, "V": 10, "dnu": 0.1},
    {"T": 5, "dt": 0.1, "V": 20, "dnu": 0.1},

    {"T": 5, "dt": 0.1, "V": 20, "dnu": 0.1},

    {"T": 5, "dt": 0.01, "V": 50, "dnu": 0.1},  # почти идеал
]

prefix = "1_3"
for i, p in enumerate(param_sets):
    T, dt, V, dnu = p["T"], p["dt"], p["V"], p["dnu"]

    # сетки
    t = np.arange(-T, T, dt)
    nu = np.arange(-V, V, dnu)

    f = pi_func(t)

    # прямое и обратное через trapz
    F_num = fourier_trapz(t, f, nu)
    f_rec = inverse_fourier_trapz(nu, F_num, t)

    # аналитика
    F_analytic = pi_hat_analytic(nu)

    # --------- ГРАФИК 1: спектр ----------
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(nu, np.real(F_num), label="numerical trapz")
    plt.plot(nu, F_analytic, '--', label="analytic sinc")
    plt.title(f"Set {i+1}: Spectrum (T={T}, dt={dt}, V={V})")
    plt.legend()
    plt.grid()
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_spectrum.png"), dpi=300)
    plt.close(fig1)  # Освобождаем память

    # --------- ГРАФИК 2: восстановление ----------
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(t, f, label="original Π(t)")
    plt.plot(t, np.real(f_rec), '--', label="reconstructed")
    plt.title(f"Set {i+1}: Reconstruction (Δt={dt}, Δν={dnu})")
    plt.legend()
    plt.grid()
    fig2.tight_layout()
    
    fig2.savefig(os.path.join(output_dir, f"{prefix}_set_{i+1}_reconstruction.png"), dpi=300)
    plt.close(fig2)  # Освобождаем память

    print(f"Сохранено: set_{i+1}")



#  пункт 1.5 - DFT
def fft_unitary(f, dt):
    F = np.fft.fftshift(np.fft.fft(f))
    return F * dt / np.sqrt(2*np.pi)

def ifft_unitary(F, dnu):
    f = np.fft.ifft(np.fft.ifftshift(F))
    return f * len(F) * dnu / np.sqrt(2*np.pi)

def fft_clean(sig, dt):
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(sig))
    ) * dt
def ifft_clean(F, dt):
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(F))
    ) / dt

F_fft = fft_unitary(f, dt)
f_rec_fft = ifft_unitary(F_fft, dnu)

def rect_f(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)

T = 5
dt = 0.01

t = np.arange(-T, T, dt)
f = pi_func(t)

N = len(t)
dnu = 1 / (N * dt)
nu_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))


F_fft = fft_unitary(f, dt)
f_rec_fft = ifft_unitary(F_fft, dnu)

T_list = [5, 45, 15, 15, 15]
dt_list = [0.05, 0.05, 0.05, 0.02, 0.01]

for i, (T, dt) in enumerate(zip(T_list, dt_list)):

    t = np.arange(-T/2, T/2, dt)
    f = rect_f(t)

    N = len(t)
    nu = np.fft.fftshift(np.fft.fftfreq(N, dt))

    # FFT
    F = fft_clean(f, dt)

    # IFFT
    f_rec = ifft_clean(F, dt)

    # аналитика (ВАЖНО: sinc)
    F_analytic = np.sinc(nu)

    # =====================
    # СПЕКТР
    # =====================
    plt.figure(figsize=(12,6))
    plt.plot(nu, np.real(F), 'g--', label="FFT")
    plt.plot(nu, F_analytic, 'r', label="analytic sinc")
    plt.xlabel("ν")
    plt.ylabel("F(Π)")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"1_5_set_{i+1}_spectrum.png"))
    plt.show()

    # =====================
    # ВОССТАНОВЛЕНИЕ
    # =====================
    plt.figure(figsize=(12,6))
    plt.plot(t, f, 'r', label="original")
    plt.plot(t, np.real(f_rec), 'g--', label="FFT reconstructed")
    plt.xlabel("t")
    plt.ylabel("Π(t)")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"1_5_set_{i+1}_reconstruction.png"))
    plt.show()