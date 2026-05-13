import os
import time
import numpy as np
import matplotlib.pyplot as plt


FIG_DIR = "task1/figs_1_5"

def trapz(y, x):
    """Совместимость со старыми и новыми версиями NumPy."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def savefig(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def pi_func(t):
    """Прямоугольная функция Π(t)."""
    return (np.abs(t) <= 0.5).astype(float)


def analytic_pi_ft(v):
    """Аналитический Фурье-образ Π(t): sinc(v)=sin(pi v)/(pi v)."""
    return np.sinc(v)


def make_time_grid(T, dt):
    """Симметричная сетка [-T/2, T/2)."""
    return np.arange(-T / 2, T / 2, dt)


def make_centered_freq_grid(N, dt):
    """Центрированная частотная сетка для FFT."""
    dv = 1.0 / (N * dt)
    v = (np.arange(N) - N // 2) * dv
    return v, dv


def continuous_ft_trapz(f_t, t, v):
    """Численное прямое непрерывное ПФ через trapz."""
    F = np.zeros_like(v, dtype=complex)
    for k, vk in enumerate(v):
        F[k] = trapz(f_t * np.exp(-2j * np.pi * vk * t), t)
    return F


def continuous_ift_trapz(F_v, v, t):
    """Численное обратное непрерывное ПФ через trapz."""
    f = np.zeros_like(t, dtype=complex)
    for n, tn in enumerate(t):
        f[n] = trapz(F_v * np.exp(2j * np.pi * v * tn), v)
    return f


def dft_unitary_raw(x):
    """Унитарный DFT."""
    return np.fft.fft(x) / np.sqrt(len(x))


def idft_unitary_raw(X):
    """Обратный унитарный DFT."""
    return np.fft.ifft(X) * np.sqrt(len(X))


def smart_ctft_raw(f_t, t):
    """
    Приближение непрерывного ПФ через сумму Римана и FFT:
    F(ν_m) ≈ c_m * fft(f)_m,
    где c_m = dt * exp(-2πi * m * dv * t0).
    """
    N = len(f_t)
    dt = t[1] - t[0]
    T = N * dt
    t0 = t[0]
    dv = 1.0 / T
    m = np.arange(N)

    c = dt * np.exp(-2j * np.pi * m * dv * t0)
    F_raw = c * np.fft.fft(f_t)
    return F_raw


def smart_ctft_centered(f_t, t):
    """Центрированный спектр для отображения."""
    N = len(f_t)
    dt = t[1] - t[0]
    v, _ = make_centered_freq_grid(N, dt)
    F = np.fft.fftshift(smart_ctft_raw(f_t, t))
    return v, F


def smart_ictft_from_centered(F_centered, t):
    """Обратное восстановление для smart-метода."""
    N = len(F_centered)
    dt = t[1] - t[0]
    T = N * dt
    t0 = t[0]
    dv = 1.0 / T
    m = np.arange(N)

    F_raw = np.fft.ifftshift(F_centered)
    c = dt * np.exp(-2j * np.pi * m * dv * t0)
    return np.fft.ifft(F_raw / c)


def sample_signal(t_dense, y_dense, dt_sample):
    """Сэмплирование через интерполяцию на нужных узлах."""
    ts = np.arange(t_dense[0], t_dense[-1] + 0.5 * dt_sample, dt_sample)
    ys = np.interp(ts, t_dense, y_dense)
    return ts, ys


def sinc_interpolate(ts, ys, t_dense, dt_sample):
    """Интерполяция по формуле Котельникова."""
    M = (t_dense[:, None] - ts[None, :]) / dt_sample
    return np.sum(ys[None, :] * np.sinc(M), axis=1)


# ============================================================
# ФУНКЦИИ ОТРИСОВКИ
# ============================================================

# def plot_time_signal(t, y, title, filename, xlabel="Время t", ylabel="Амплитуда", xlim=None):
#     plt.figure()
#     plt.plot(t, np.real(y), linewidth=2)
#     plt.grid(True)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     if xlim is not None:
#         plt.xlim(xlim)
    
#     savefig(filename)


def plot_compare_time(t_true, y_true, t_num, y_num, title, filename,
                      label_true="Исходная функция",
                      label_num="Восстановленная функция",
                      xlim=None):
    plt.figure()
    plt.plot(t_true, np.real(y_true), linewidth=2, label=label_true)
    plt.plot(t_num, np.real(y_num), "--", linewidth=2, label=label_num)
    plt.grid(True)
    plt.xlabel("Время t")
    plt.ylabel("Амплитуда")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    #plt.show()
    savefig(filename)


def plot_compare_freq(v_ref, F_ref, v_num, F_num, title, filename,
                      label_num="Численный образ", xlim=None):
    plt.figure()
    plt.plot(v_ref, np.real(F_ref), linewidth=2.2, label="Аналитический образ")
    plt.plot(v_num, np.real(F_num), "--", linewidth=1.8, label=label_num)
    plt.grid(True)
    plt.xlabel("Частота ν")
    plt.ylabel("Значение образа")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.show()
    savefig(filename)


# def plot_combined_time_freq(v_ref, F_ref, v_num, F_num,
#                             t_ref, x_ref, t_num, x_num,
#                             title_freq, title_time, filename,
#                             label_freq="Численный образ",
#                             label_time="Восстановленная функция",
#                             xlim_freq=(-10, 10), xlim_time=(-2, 2)):
#     """Один рисунок из двух графиков: спектр + восстановление."""
#     plt.figure(figsize=(10, 8))

#     plt.subplot(2, 1, 1)
#     plt.plot(v_ref, np.real(F_ref), linewidth=2, label="Аналитический образ")
#     plt.plot(v_num, np.real(F_num), "--", linewidth=1.5, label=label_freq)
#     plt.grid(True)
#     plt.xlabel("Частота ν")
#     plt.ylabel("Значение")
#     plt.title(title_freq)
#     plt.xlim(xlim_freq)
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(t_ref, np.real(x_ref), linewidth=2, label="Исходная функция")
#     plt.plot(t_num, np.real(x_num), "--", linewidth=1.5, label=label_time)
#     plt.grid(True)
#     plt.xlabel("Время t")
#     plt.ylabel("Амплитуда")
#     plt.title(title_time)
#     plt.xlim(xlim_time)
#     plt.legend()

#     savefig(filename)


def plot_dft_spectrum(v_ref, F_ref, v_dft, X_dft, title, filename, xlim=None):
    """
    Обычный DFT рисуется одной линией, без отдельной мнимой/действительной части.
    Для визуального сравнения с вещественным sinc используется вещественная часть.
    """
    plt.figure()
    plt.plot(v_ref, np.real(F_ref), linewidth=2.2, label="Аналитический образ")
    plt.plot(v_dft, np.real(X_dft), "o--", markersize=2.2, linewidth=1.2, label="Обычный DFT")
    plt.grid(True)
    plt.xlabel("Частота ν")
    plt.ylabel("Значение образа")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    #plt.show
    savefig(filename)


# def plot_methods_compare(v_ref, F_ref,
#                          v_trap, F_trap,
#                          v_dft, F_dft,
#                          v_smart, F_smart,
#                          filename):
#     plt.figure()
#     plt.plot(v_ref, np.real(F_ref), linewidth=2.4, label="Аналитический образ")
#     plt.plot(v_trap, np.real(F_trap), "--", linewidth=2.0, label="Численное интегрирование")
#     plt.plot(v_dft, np.real(F_dft), "-.", linewidth=1.8, label="Обычный DFT")
#     plt.plot(v_smart, np.real(F_smart), ":", linewidth=2.6, label="Разработанный метод")
#     plt.grid(True)
#     plt.xlabel("Частота ν")
#     plt.ylabel("Значение образа")
#     plt.title("Сравнение методов вычисления Фурье-образа")
#     plt.xlim(-12, 12)
#     plt.legend()
#     savefig(filename)


# def plot_sampling_and_interp(t_dense, y_dense, ts, ys, y_interp, title, filename, xlim=None):
#     plt.figure()
#     plt.plot(t_dense, y_dense, linewidth=2, label="Исходная функция")
#     plt.stem(ts, ys, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Отсчёты")
#     plt.plot(t_dense, y_interp, "--", linewidth=2, label="Интерполяция")
#     plt.grid(True)
#     plt.xlabel("Время t")
#     plt.ylabel("Амплитуда")
#     plt.title(title)
#     if xlim is not None:
#         plt.xlim(xlim)
#     plt.legend()
#     savefig(filename)


# def plot_spectrum_three_lines(v, Y_dense, Y_sampled, Y_interp, title, filename, xlim=None):
#     """
#     Упрощённый спектральный график: только три линии,
#     как в большинстве отчётов:
#     1) исходная функция,
#     2) сэмплированный сигнал,
#     3) интерполяция.
#     """
#     plt.figure()
#     plt.plot(v, np.abs(Y_dense), linewidth=2.2, label="Исходная функция")
#     plt.plot(v, np.abs(Y_sampled), "--", linewidth=1.9, label="Сэмплированный сигнал")
#     plt.plot(v, np.abs(Y_interp), "-.", linewidth=1.9, label="Интерполяция")
#     plt.grid(True)
#     plt.xlabel("Частота ν")
#     plt.ylabel("Амплитуда")
#     plt.title(title)
#     if xlim is not None:
#         plt.xlim(xlim)
#     plt.legend()
#     savefig(filename)




def section_3_dft():
    v_ref = np.linspace(-12, 12, 4000)
    F_ref = analytic_pi_ft(v_ref)

    # Неудачный случай по T
    T = 2.0
    dt = 0.001
    t = make_time_grid(T, dt)
    x = pi_func(t)
    v_dft, _ = make_centered_freq_grid(len(t), dt)
    X = np.fft.fftshift(dft_unitary_raw(x))

    plot_dft_spectrum(
        v_ref, F_ref, v_dft, X,
        "DFT: маленький параметр T",
        "dft_bad_T_spectrum.png",
        xlim=(-12, 12)
    )

    # Удачный случай по T
    T = 8.0
    dt = 0.001
    t = make_time_grid(T, dt)
    x = pi_func(t)
    v_dft, _ = make_centered_freq_grid(len(t), dt)
    X = np.fft.fftshift(dft_unitary_raw(x))

    plot_dft_spectrum(
        v_ref, F_ref, v_dft, X,
        "DFT:большой параметр T",
        "dft_good_T_spectrum.png",
        xlim=(-12, 12)
    )

    # Неудачный случай по dt: восстановление
    T = 8.0
    dt = 0.1
    t = make_time_grid(T, dt)
    x = pi_func(t)
    X_raw = dft_unitary_raw(x)
    x_rec = idft_unitary_raw(X_raw)

    plot_compare_time(
        t, x, t, x_rec,
        "IDFT: большой параметр dt",
        "dft_bad_dt_reconstruction.png",
        label_true="Исходная функция",
        label_num="Восстановленная функция",
        xlim=(-4, 4)
    )

    # Удачный случай по dt: восстановление
    T = 8.0
    dt = 0.001
    t = make_time_grid(T, dt)
    x = pi_func(t)
    X_raw = dft_unitary_raw(x)
    x_rec = idft_unitary_raw(X_raw)

    plot_compare_time(
        t, x, t, x_rec,
        "IDFT: малый параметр dt",
        "dft_good_dt_reconstruction.png",
        label_true="Исходная функция",
        label_num="Восстановленная функция",
        xlim=(-4, 4)
    )
section_3_dft()