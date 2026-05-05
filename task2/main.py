import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# ПАПКА ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
# ============================================================

FIGURES_DIR = Path.cwd() / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# ЕДИНЫЙ СТИЛЬ ГРАФИКОВ
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.35,
})


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

dt_limit_y1 = 1 / (2 * B_y1)
dt_limit_y2 = 1 / (2 * B_y2)


print("=== Параметры функции y1(t) ===")
print(f"y1(t) = {a1} sin({w1}t) + {a2} sin({w2}t + pi/2)")
print(f"B1 = {B_y1:.4f}")
print(f"Предельный шаг по теореме Котельникова: Δt < {dt_limit_y1:.4f}")
print()

print("=== Параметры функции y2(t) ===")
print(f"y2(t) = sinc({b}t)")
print(f"B2 = {B_y2:.4f}")
print(f"Предельный шаг по теореме Котельникова: Δt < {dt_limit_y2:.4f}")
print()


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def make_time_grid(T, dt):
    k_max = int(np.floor((T / 2) / dt))
    k = np.arange(-k_max, k_max + 1)
    return k * dt


def reconstruct_sinc(t_dense, t_samples, y_samples, Ts):
    result = np.zeros_like(t_dense, dtype=float)

    for tk, yk in zip(t_samples, y_samples):
        result += yk * np.sinc((t_dense - tk) / Ts)

    return result


def smart_fft(y, dt):
    """
    Приближение непрерывного Фурье-образа через FFT:
        F(ν) ≈ Δt * FFT(y)
    """
    return dt * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))


def frequency_grid(N, dt):
    return np.fft.fftshift(np.fft.fftfreq(N, d=dt))


def spectrum_part(y, dt, part="real"):
    Y = smart_fft(y, dt)
    nu = frequency_grid(len(y), dt)

    if part == "real":
        return nu, Y.real

    if part == "imag":
        return nu, Y.imag

    raise ValueError("part должен быть 'real' или 'imag'")


def add_case_label(ax, text):
    ax.text(
        0.02,
        0.92,
        text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )


# ============================================================
# ИСХОДНЫЕ ФУНКЦИИ
# ============================================================

def plot_original_function(func, T, dt_dense, xlim, ylabel, filename):
    t_dense = make_time_grid(T, dt_dense)
    y_dense = func(t_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(t_dense, y_dense, label="Исходная функция")
    plt.xlim(-xlim, xlim)
    plt.xlabel("Время $t$")
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    save_figure(filename)


# ============================================================
# ВОССТАНОВЛЕНИЕ ПО Δt
# ============================================================

def plot_reconstruction_by_dt(
    func,
    T,
    dt_dense,
    dt_values,
    xlim,
    ylabel,
    filename,
):
    t_dense = make_time_grid(T, dt_dense)
    y_dense = func(t_dense)

    fig, axes = plt.subplots(
        len(dt_values),
        1,
        figsize=(11, 11),
        sharex=True,
    )

    for i, (ax, Ts) in enumerate(zip(axes, dt_values)):
        t_samples = make_time_grid(T, Ts)
        y_samples = func(t_samples)
        y_rec = reconstruct_sinc(t_dense, t_samples, y_samples, Ts)

        ax.plot(t_dense, y_dense, label="Исходная функция")
        ax.plot(t_dense, y_rec, "--", label="Восстановленная функция")
        ax.scatter(
            t_samples,
            y_samples,
            color="C2",
            s=30,
            label="Сэмплы",
        )

        ax.set_xlim(-xlim, xlim)
        ax.set_ylabel(ylabel)
        add_case_label(ax, rf"$\Delta t = {Ts}$")

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Время $t$")
    save_figure(filename)


# ============================================================
# ВОССТАНОВЛЕНИЕ ПО T
# ============================================================

def plot_reconstruction_by_T(
    func,
    T_values,
    dt_dense,
    Ts,
    xlim,
    ylabel,
    filename,
):
    fig, axes = plt.subplots(
        len(T_values),
        1,
        figsize=(11, 11),
        sharex=True,
    )

    for i, (ax, T) in enumerate(zip(axes, T_values)):
        t_dense = make_time_grid(T, dt_dense)
        y_dense = func(t_dense)

        t_samples = make_time_grid(T, Ts)
        y_samples = func(t_samples)
        y_rec = reconstruct_sinc(t_dense, t_samples, y_samples, Ts)

        ax.plot(t_dense, y_dense, label="Исходная функция")
        ax.plot(t_dense, y_rec, "--", label="Восстановленная функция")
        ax.scatter(
            t_samples,
            y_samples,
            color="C2",
            s=30,
            label="Сэмплы",
        )

        ax.set_xlim(-xlim, xlim)
        ax.set_ylabel(ylabel)
        add_case_label(ax, rf"$T = {T}$, $\Delta \nu = {1 / T:.3f}$")

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Время $t$")
    save_figure(filename)


# ============================================================
# ФУРЬЕ-ОБРАЗЫ ПО Δt
# ============================================================

def plot_spectrum_by_dt(
    func,
    T,
    dt_dense,
    dt_values,
    B,
    xlim_nu,
    filename,
    part="real",
):
    t_dense = make_time_grid(T, dt_dense)
    y_dense = func(t_dense)

    nu_dense, Y_dense = spectrum_part(y_dense, dt_dense, part=part)

    fig, axes = plt.subplots(
        len(dt_values),
        1,
        figsize=(11, 11),
        sharex=True,
    )

    for i, (ax, Ts) in enumerate(zip(axes, dt_values)):
        t_samples = make_time_grid(T, Ts)
        y_samples = func(t_samples)
        y_rec = reconstruct_sinc(t_dense, t_samples, y_samples, Ts)

        nu_rec, Y_rec = spectrum_part(y_rec, dt_dense, part=part)
        nu_samples, Y_samples = spectrum_part(y_samples, Ts, part=part)

        ax.plot(nu_dense, Y_dense, label="Исходная функция")
        ax.plot(nu_rec, Y_rec, "--", label="Восстановленная функция")
        ax.scatter(
            nu_samples,
            Y_samples,
            color="g",
            s=10,
            label="Сэмплированная функция",
        )
        ax.axvline(B, linestyle=":", color="black")
        ax.axvline(-B, linestyle=":", color="black")

        ax.set_xlim(-xlim_nu, xlim_nu)

        if part == "real":
            ax.set_ylabel(r"$\mathrm{Re}\,\hat{y}(\nu)$")
        else:
            ax.set_ylabel(r"$\mathrm{Im}\,\hat{y}(\nu)$")

        add_case_label(ax, rf"$\Delta t = {Ts}$")

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Частота ν")
    save_figure(filename)


# ============================================================
# ФУРЬЕ-ОБРАЗЫ ПО T
# ============================================================

def plot_spectrum_by_T(
    func,
    T_values,
    dt_dense,
    Ts,
    B,
    xlim_nu,
    filename,
    part="real",
):
    fig, axes = plt.subplots(
        len(T_values),
        1,
        figsize=(11, 11),
        sharex=True,
    )

    for i, (ax, T) in enumerate(zip(axes, T_values)):
        t_dense = make_time_grid(T, dt_dense)
        y_dense = func(t_dense)

        t_samples = make_time_grid(T, Ts)
        y_samples = func(t_samples)
        y_rec = reconstruct_sinc(t_dense, t_samples, y_samples, Ts)

        nu_dense, Y_dense = spectrum_part(y_dense, dt_dense, part=part)
        nu_rec, Y_rec = spectrum_part(y_rec, dt_dense, part=part)
        nu_samples, Y_samples = spectrum_part(y_samples, Ts, part=part)

        ax.plot(nu_dense, Y_dense, label="Исходная функция")
        ax.plot(nu_rec, Y_rec, "--", label="Восстановленная функция")
        ax.scatter(
            nu_samples,
            Y_samples,
            color="g",
            s=10,
            label="Сэмплированная функция",
        )
        ax.axvline(B, linestyle=":", color="black")
        ax.axvline(-B, linestyle=":", color="black")

        ax.set_xlim(-xlim_nu, xlim_nu)

        if part == "real":
            ax.set_ylabel(r"$\mathrm{Re}\,\hat{y}(\nu)$")
        else:
            ax.set_ylabel(r"$\mathrm{Im}\,\hat{y}(\nu)$")

        add_case_label(ax, rf"$T = {T}$, $\Delta \nu = {1 / T:.3f}$")

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Частота ν")
    save_figure(filename)


# ============================================================
# ОБЩИЕ ПАРАМЕТРЫ ЭКСПЕРИМЕНТОВ
# ============================================================

dt_dense = 0.001

T_for_dt_study = 40
T_values_for_T_study = [3, 7, 23]

dt_values_y1 = [0.1, 0.3, 0.6]
dt_values_y2 = [0.05, 0.2, 0.4]

dt_good_y1 = 0.3
dt_bad_y1 = 0.6

dt_good_y2 = 0.2
dt_bad_y2 = 0.4

plot_window_y1 = 8
plot_window_y2 = 8

spectrum_window_y1 = 3
spectrum_window_y2 = 6


# ============================================================
# ИСХОДНЫЕ ФУНКЦИИ
# ============================================================

plot_original_function(
    func=y1_func,
    T=40,
    dt_dense=dt_dense,
    xlim=plot_window_y1,
    ylabel="$y_1(t)$",
    filename="01_y1_original.png",
)

plot_original_function(
    func=y2_func,
    T=40,
    dt_dense=dt_dense,
    xlim=plot_window_y2,
    ylabel="$y_2(t)$",
    filename="02_y2_original.png",
)


# ============================================================
# y1(t): ВЛИЯНИЕ Δt ПРИ T = 40
# ============================================================

plot_reconstruction_by_dt(
    func=y1_func,
    T=T_for_dt_study,
    dt_dense=dt_dense,
    dt_values=dt_values_y1,
    xlim=plot_window_y1,
    ylabel="$y_1(t)$",
    filename="03_y1_reconstruction_by_dt.png",
)

plot_spectrum_by_dt(
    func=y1_func,
    T=T_for_dt_study,
    dt_dense=dt_dense,
    dt_values=dt_values_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="04_y1_spectrum_by_dt_real.png",
    part="real",
)

plot_spectrum_by_dt(
    func=y1_func,
    T=T_for_dt_study,
    dt_dense=dt_dense,
    dt_values=dt_values_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="04b_y1_spectrum_by_dt_imag.png",
    part="imag",
)


# ============================================================
# y1(t): ВЛИЯНИЕ T ПРИ ХОРОШЕМ И ПЛОХОМ Δt
# ============================================================

plot_reconstruction_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_good_y1,
    xlim=plot_window_y1,
    ylabel="$y_1(t)$",
    filename="05_y1_reconstruction_by_T_good_dt.png",
)

plot_spectrum_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_good_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="06_y1_spectrum_by_T_good_dt_real.png",
    part="real",
)

plot_spectrum_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_good_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="06b_y1_spectrum_by_T_good_dt_imag.png",
    part="imag",
)

plot_reconstruction_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_bad_y1,
    xlim=plot_window_y1,
    ylabel="$y_1(t)$",
    filename="07_y1_reconstruction_by_T_bad_dt.png",
)

plot_spectrum_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_bad_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="08_y1_spectrum_by_T_bad_dt_real.png",
    part="real",
)

plot_spectrum_by_T(
    func=y1_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_bad_y1,
    B=B_y1,
    xlim_nu=spectrum_window_y1,
    filename="08b_y1_spectrum_by_T_bad_dt_imag.png",
    part="imag",
)


# ============================================================
# y2(t): ВЛИЯНИЕ Δt ПРИ T = 40
# ============================================================

plot_reconstruction_by_dt(
    func=y2_func,
    T=T_for_dt_study,
    dt_dense=dt_dense,
    dt_values=dt_values_y2,
    xlim=plot_window_y2,
    ylabel="$y_2(t)$",
    filename="09_y2_reconstruction_by_dt.png",
)

plot_spectrum_by_dt(
    func=y2_func,
    T=T_for_dt_study,
    dt_dense=dt_dense,
    dt_values=dt_values_y2,
    B=B_y2,
    xlim_nu=spectrum_window_y2,
    filename="10_y2_spectrum_by_dt_real.png",
    part="real",
)


# ============================================================
# y2(t): ВЛИЯНИЕ T ПРИ ХОРОШЕМ И ПЛОХОМ Δt
# ============================================================

plot_reconstruction_by_T(
    func=y2_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_good_y2,
    xlim=plot_window_y2,
    ylabel="$y_2(t)$",
    filename="11_y2_reconstruction_by_T_good_dt.png",
)

plot_spectrum_by_T(
    func=y2_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_good_y2,
    B=B_y2,
    xlim_nu=spectrum_window_y2,
    filename="12_y2_spectrum_by_T_good_dt_real.png",
    part="real",
)

plot_reconstruction_by_T(
    func=y2_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_bad_y2,
    xlim=plot_window_y2,
    ylabel="$y_2(t)$",
    filename="13_y2_reconstruction_by_T_bad_dt.png",
)

plot_spectrum_by_T(
    func=y2_func,
    T_values=T_values_for_T_study,
    dt_dense=dt_dense,
    Ts=dt_bad_y2,
    B=B_y2,
    xlim_nu=spectrum_window_y2,
    filename="14_y2_spectrum_by_T_bad_dt_real.png",
    part="real",
)


print()
print("Готово. Все графики сохранены в папку figures.")