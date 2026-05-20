import numpy as np
import matplotlib.pyplot as plt
import os

outdir = "task1/test_1_5"
os.makedirs(outdir, exist_ok=True)

COL_ANALYTIC = [0.494, 0.184, 0.556]
COL_UNITARY  = [0.850, 0.325, 0.098]
COL_ORIG     = [0.0,   0.447, 0.741]
COL_REC      = [0.929, 0.694, 0.125]

def rect_func(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)

def sinc_safe(nu):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.sin(np.pi * nu) / (np.pi * nu)
        res[np.abs(nu) < 1e-12] = 1.0
    return res

def unitary_dft(x, dt):
    N = len(x)
    F = np.fft.fftshift(np.fft.fft(x)) / np.sqrt(N)
    nu = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    return nu, F

def inverse_unitary_dft(F, dt):
    N = len(F)
    x_rec = np.fft.ifft(np.fft.ifftshift(F)) * np.sqrt(N)
    return np.real(x_rec)

def get_grid(T, dt):
    N = int(round(T / dt))
    if N % 2 != 0:
        N += 1
    dt = T / N
    t = -T/2 + np.arange(N) * dt
    x = rect_func(t)
    return t, x, dt, N

def plot_spectrum(T, dt, outdir, suffix):
    t, x, dt, N = get_grid(T, dt)
    fs = 1.0 / dt               # частота дискретизации
    fs2 = fs / 2.0              # частота Найквиста
    
    # Получаем DFT и его частоты
    nu, F_unitary = unitary_dft(x, dt)
    
    # Диапазон для отображения: от -max_freq до max_freq
    # Аналитический sinc рисуем до 35 Гц (или больше, если Найквист больше 35)
    display_max = max(35.0, fs2 + 5.0)
    nu_lim = [-display_max, display_max]
    
    # Плотная сетка для аналитической функции
    nu_dense = np.linspace(nu_lim[0], nu_lim[1], 5000)
    y_sinc = sinc_safe(nu_dense)
    
    # Маска для унитарного DFT, чтобы отобразить только частоты в пределах nu_lim
    mask = (nu >= nu_lim[0]) & (nu <= nu_lim[1])
    nu_show = nu[mask]
    F_show = np.real(F_unitary[mask])
    
    # Автоматический вертикальный масштаб
    all_values = np.concatenate([y_sinc, F_show])
    max_val = np.max(all_values)
    min_val = np.min(all_values)
    y_margin = 0.15 * (max_val - min_val) if (max_val - min_val) > 0 else 0.2
    y_min = min_val - y_margin
    y_max = max_val + y_margin
    
    plt.figure(figsize=(10, 5), facecolor='white')
    plt.plot(nu_dense, y_sinc, color=COL_ANALYTIC, linewidth=2.0,
             label=r'Аналитический $\hat{\Pi}(\nu)$')
    plt.plot(nu_show, F_show, '--', color=COL_UNITARY, linewidth=1.5,
             label='Унитарный DFT')
    plt.grid(True, alpha=0.3)
    plt.xlim(nu_lim)
    plt.ylim(y_min, y_max)
    plt.xlabel(r'Частота $\nu$', fontsize=12)
    plt.ylabel(r'Спектр $\Pi(\nu)$', fontsize=12)
    plt.title(f'Сравнение спектров (T={T}, dt={dt:.3f}), fs/2 = {fs2:.2f} Гц', fontsize=12)
    text = (f"Унитарный DFT существует только на частотах до |ν|={fs2:.2f} Гц (частота Найквиста).\n"
            "Аналитический sinc имеет бесконечный спектр. При малом fs/2 (большом dt)\n"
            "видна только центральная часть sinc.")
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
             fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_signal_reconstruction(T, dt, outdir, suffix, col_orig, col_rec):
    t_dft, x_dft, dt, N = get_grid(T, dt)
    nu, F = unitary_dft(x_dft, dt)
    x_rec = inverse_unitary_dft(F, dt)
    error = np.linalg.norm(x_dft - x_rec) / np.linalg.norm(x_dft)

    t_dense = np.linspace(t_dft.min(), t_dft.max(), 2000)
    x_dense = rect_func(t_dense)

    plt.figure(figsize=(10, 4.2), facecolor='white')
    plt.plot(t_dense, x_dense, color=col_orig, linewidth=2.0, label='Исходный Π(t) (плотная сетка)')
    plt.plot(t_dft, x_rec, 'o', color=col_rec, markersize=4, label=f'Восстановленный из DFT (отсчёты)\nошибка = {error:.2e}')
    plt.plot(t_dft, x_rec, '--', color=col_rec, linewidth=0.8, alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlim(t_dft.min(), t_dft.max())
    plt.ylim(-0.2, 1.2)
    plt.xlabel('Время t', fontsize=12)
    plt.ylabel('Сигнал Π(t)', fontsize=12)
    plt.title(f'Исходный и восстановленный сигнал (T={T}, dt={dt:.3f})', fontsize=12)
    text = ("Отсчёты восстановленного сигнала совпадают с исходными.\n"
            "Между отсчётами сигнал не определён – показана идеальная форма Π(t).")
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
             fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()
    print(f"  Ошибка восстановления (унитарный DFT): {error:.2e}")

if __name__ == "__main__":
    cases = [
        # (10, 0.5,  'very_bad_dt'),
        (10, 0.05, 'bad_dt'),
        # (10, 0.01, 'good_dt'),
        # (1,  0.01, 'bad_T'),
        # (10, 0.01, 'good_T'),
        # (5,  0.02, 'extra'),
        # (3,  0.005, 'dense')
    ]

    for T, dt, suffix in cases:
        print(f"\nОбработка: T={T}, dt={dt}")
        plot_spectrum(T, dt, outdir, suffix)
        plot_signal_reconstruction(T, dt, outdir, suffix, COL_ORIG, COL_REC)

    print(f"\nГрафики сохранены в {outdir}")