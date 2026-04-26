import numpy as np
import matplotlib

# Чтобы код нормально работал в WSL без открытия окон
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# ПАПКА ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
# ============================================================

FIGURES_DIR = Path.cwd() / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def save_figure(filename):
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Сохранено: {path}")
    plt.close()


# ============================================================
# ПАРАМЕТРЫ ФУНКЦИЙ
# ============================================================

a1 = 2
a2 = 5
w1 = 3
w2 = 10
phi1 = 0
phi2 = np.pi / 2

b = 4


def y1_func(t):
    return a1 * np.sin(w1 * t + phi1) + a2 * np.sin(w2 * t + phi2)


def y2_func(t):
    return np.sinc(b * t)


B_y1 = max(w1, w2) / (2 * np.pi)
B_y2 = b / 2

print("=== Параметры y1(t) ===")
print(f"y1(t) = {a1} sin({w1}t + {phi1}) + {a2} sin({w2}t + pi/2)")
print(f"B_y1 = {B_y1:.4f}")
print(f"2B_y1 = {2 * B_y1:.4f}")
print(f"Условие Найквиста: fs > {2 * B_y1:.4f}")
print(f"То есть Ts < {1 / (2 * B_y1):.4f}")
print()

print("=== Параметры y2(t) ===")
print(f"y2(t) = sinc({b}t)")
print(f"B_y2 = {B_y2:.4f}")
print(f"2B_y2 = {2 * B_y2:.4f}")
print(f"Условие Найквиста: fs > {2 * B_y2:.4f}")
print(f"То есть Ts < {1 / (2 * B_y2):.4f}")
print()


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def reconstruct_sinc(t_dense, t_samples, y_samples, Ts):
    result = np.zeros_like(t_dense, dtype=float)

    for tk, yk in zip(t_samples, y_samples):
        result += yk * np.sinc((t_dense - tk) / Ts)

    return result


def smart_fft(y, dt):
    return dt * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))


def frequency_grid(N, T):
    return np.arange(-N / 2, N / 2) * (1 / T)


def normalized_abs(Y):
    Y_abs = np.abs(Y)
    max_val = np.max(Y_abs)

    if max_val == 0:
        return Y_abs

    return Y_abs / max_val


# ============================================================
# ОБЩИЕ НАСТРОЙКИ МОДЕЛИРОВАНИЯ
# ============================================================

T = 40
dt_cont = 0.001

t_cont = np.arange(-T / 2, T / 2, dt_cont)
N_cont = len(t_cont)
nu_cont = frequency_grid(N_cont, T)

plot_window_y1 = 8
plot_window_y2 = 8

Ts_values_y1 = [0.1, 0.3, 0.6]
Ts_values_y2 = [0.05, 0.2, 0.4]


# ============================================================
# ИСХОДНЫЕ ФУНКЦИИ
# ============================================================

y1_cont = y1_func(t_cont)
y2_cont = y2_func(t_cont)

plt.figure(figsize=(10, 5))
plt.plot(t_cont, y1_cont, linewidth=2)
plt.grid()
plt.xlim(-plot_window_y1, plot_window_y1)
plt.xlabel("t")
plt.ylabel("y1(t)")
save_figure("01_y1_original.png")

plt.figure(figsize=(10, 5))
plt.plot(t_cont, y2_cont, linewidth=2)
plt.grid()
plt.xlim(-plot_window_y2, plot_window_y2)
plt.xlabel("t")
plt.ylabel("y2(t)")
save_figure("02_y2_original.png")


# ============================================================
# СЭМПЛИРОВАНИЕ И ВОССТАНОВЛЕНИЕ y1(t)
# ============================================================

for Ts in Ts_values_y1:
    t_samples = np.arange(-T / 2, T / 2, Ts)
    y_samples = y1_func(t_samples)

    y_rec = reconstruct_sinc(t_cont, t_samples, y_samples, Ts)

    plt.figure(figsize=(10, 5))
    plt.plot(t_cont, y1_cont, linewidth=2, label="Исходная")
    plt.stem(
        t_samples,
        y_samples,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="C1-",
        label="Сэмплы",
    )
    plt.plot(t_cont, y_rec, "--", linewidth=1.5, label="Восстановленная")
    plt.grid()
    plt.xlim(-plot_window_y1, plot_window_y1)
    plt.xlabel("t")
    plt.ylabel("y1(t)")
    plt.legend()

    filename = f"03_y1_sampling_Ts_{str(Ts).replace('.', '_')}.png"
    save_figure(filename)


# ============================================================
# СЭМПЛИРОВАНИЕ И ВОССТАНОВЛЕНИЕ y2(t)
# ============================================================

for Ts in Ts_values_y2:
    t_samples = np.arange(-T / 2, T / 2, Ts)
    y_samples = y2_func(t_samples)

    y_rec = reconstruct_sinc(t_cont, t_samples, y_samples, Ts)

    plt.figure(figsize=(10, 5))
    plt.plot(t_cont, y2_cont, linewidth=2, label="Исходная")
    plt.stem(
        t_samples,
        y_samples,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="C1-",
        label="Сэмплы",
    )
    plt.plot(t_cont, y_rec, "--", linewidth=1.5, label="Восстановленная")
    plt.grid()
    plt.xlim(-plot_window_y2, plot_window_y2)
    plt.xlabel("t")
    plt.ylabel("y2(t)")
    plt.legend()

    filename = f"04_y2_sampling_Ts_{str(Ts).replace('.', '_')}.png"
    save_figure(filename)


# ============================================================
# СПЕКТРЫ y1(t)
# ============================================================

Y1_cont = smart_fft(y1_cont, dt_cont)

for Ts in Ts_values_y1:
    t_samples = np.arange(-T / 2, T / 2, Ts)
    y_samples = y1_func(t_samples)

    y_rec = reconstruct_sinc(t_cont, t_samples, y_samples, Ts)

    Y1_rec = smart_fft(y_rec, dt_cont)

    N_samples = len(t_samples)
    nu_samples = frequency_grid(N_samples, T)
    Y1_samples = smart_fft(y_samples, Ts)

    plt.figure(figsize=(10, 5))
    plt.plot(
        nu_cont,
        normalized_abs(Y1_cont),
        linewidth=2,
        label="Исходная",
    )
    plt.plot(
        nu_cont,
        normalized_abs(Y1_rec),
        "--",
        linewidth=1.5,
        label="Восстановленная",
    )
    plt.plot(
        nu_samples,
        normalized_abs(Y1_samples),
        ":",
        linewidth=1.5,
        label="Сэмплированная",
    )

    plt.axvline(B_y1, linestyle=":", color="black", label="B")
    plt.axvline(-B_y1, linestyle=":", color="black")

    plt.grid()
    plt.xlim(-8, 8)
    plt.xlabel("ν")
    plt.ylabel("Нормированная амплитуда")
    plt.legend()

    filename = f"05_y1_spectrum_Ts_{str(Ts).replace('.', '_')}.png"
    save_figure(filename)


# ============================================================
# СПЕКТРЫ y2(t)
# ============================================================

Y2_cont = smart_fft(y2_cont, dt_cont)

for Ts in Ts_values_y2:
    t_samples = np.arange(-T / 2, T / 2, Ts)
    y_samples = y2_func(t_samples)

    y_rec = reconstruct_sinc(t_cont, t_samples, y_samples, Ts)

    Y2_rec = smart_fft(y_rec, dt_cont)

    N_samples = len(t_samples)
    nu_samples = frequency_grid(N_samples, T)
    Y2_samples = smart_fft(y_samples, Ts)

    plt.figure(figsize=(10, 5))
    plt.plot(
        nu_cont,
        normalized_abs(Y2_cont),
        linewidth=2,
        label="Исходная",
    )
    plt.plot(
        nu_cont,
        normalized_abs(Y2_rec),
        "--",
        linewidth=1.5,
        label="Восстановленная",
    )
    plt.plot(
        nu_samples,
        normalized_abs(Y2_samples),
        ":",
        linewidth=1.5,
        label="Сэмплированная",
    )

    plt.axvline(B_y2, linestyle=":", color="black", label="B")
    plt.axvline(-B_y2, linestyle=":", color="black")

    plt.grid()
    plt.xlim(-8, 8)
    plt.xlabel("ν")
    plt.ylabel("Нормированная амплитуда")
    plt.legend()

    filename = f"06_y2_spectrum_Ts_{str(Ts).replace('.', '_')}.png"
    save_figure(filename)


print()
print("Готово. Все графики сохранены в папку figures.")