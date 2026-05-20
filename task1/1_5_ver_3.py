import numpy as np
import matplotlib.pyplot as plt
import os

outdir = "task1/test_1_5"
os.makedirs(outdir, exist_ok=True)

# Цвета
COL_ANALYTIC = [0.494, 0.184, 0.556]
COL_UNITARY  = [0.850, 0.325, 0.098]
COL_ORIG     = [0.0,   0.447, 0.741]
COL_REC      = [0.929, 0.694, 0.125]

def rect_func(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)

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

def plot_spectrum(T, dt, outdir, suffix, col_analytic, col_unitary):
    # Расширенные пределы (гарантированно без обрезания)
    nu_lim = [-35, 35]
    t, x, dt, N = get_grid(T, dt)
    nu, F_unitary = unitary_dft(x, dt)

    nu_dense = np.linspace(nu_lim[0], nu_lim[1], 3000)
    y_sinc = np.sinc(nu_dense)

    mask = (nu >= nu_lim[0]) & (nu <= nu_lim[1])

    plt.figure(figsize=(10, 5), facecolor='white', constrained_layout=True)
    plt.plot(nu_dense, y_sinc, color=col_analytic, linewidth=2.0,
             label=r'Аналитический $\hat{\Pi}(\nu)$')
    plt.plot(nu[mask], np.real(F_unitary[mask]), '--', color=col_unitary, linewidth=0.5,
             label='Унитарный DFT')
    plt.grid(True, color=[0.7, 0.7, 0.7])
    plt.xlim(nu_lim)
    plt.ylim(-0.6, 1.4)   # запас сверху и снизу
    plt.xlabel(r'Частота $\nu$', fontsize=12)
    plt.ylabel(r'Спектр $\Pi(\nu)$', fontsize=12)
    plt.title(f'Сравнение спектров для T={T}, dt={dt:.3f}', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    filename = f'spectrum_{suffix}.png'
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()
    print(f"Сохранён спектр: {filename}")

def plot_reconstruction(T, dt, outdir, suffix, col_orig, col_rec):
    t_dft, x_dft, dt, N = get_grid(T, dt)
    
    # Выполняем прямое и обратное унитарное DFT
    nu, F = unitary_dft(x_dft, dt)
    x_rec = inverse_unitary_dft(F, dt)   # восстановленные отсчёты (совпадают с x_dft с точностью до ошибок округления)
    error = np.linalg.norm(x_dft - x_rec) / np.linalg.norm(x_dft)
    
    # Создаём плотную временную сетку для отображения идеального прямоугольного сигнала
    t_dense = np.linspace(t_dft.min(), t_dft.max(), 2000)
    x_dense = rect_func(t_dense)   # идеальный сигнал
    
    # Построение
    plt.figure(figsize=(10, 4.2), facecolor='white')
    plt.plot(t_dense, x_dense, color=col_orig, linewidth=2.0, label='Исходный Π(t) (плотная сетка)')
    plt.plot(t_dft, x_rec, 'o', color=col_rec, markersize=4, label=f'Восстановленный из DFT (отсчёты)\nошибка = {error:.2e}')
    # Для наглядности можно соединить отсчёты пунктиром (по желанию)
    plt.plot(t_dft, x_rec, '--', color=col_rec, linewidth=0.8, alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlim(t_dft.min(), t_dft.max())
    plt.ylim(-0.2, 1.2)
    plt.xlabel('Время t', fontsize=12)
    plt.ylabel('Сигнал Π(t)', fontsize=12)
    plt.title(f'Исходный и восстановленный сигнал (T={T}, dt={dt:.3f})', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'signal_reconstruction_{suffix}.png'), dpi=150)
    plt.close()
    print(f"  Ошибка восстановления (унитарный DFT): {error:.2e}")

cases = [
    (10, 0.05, 'bad_dt'),
    (10, 0.01, 'good_dt'),
    (10,  0.005, 'dense'),
    (20,  0.01, 'bad_T'),
    (1, 0.01, 'good_T')
]

for T, dt, suffix in cases:
    print(f"\nОбработка случая: T={T}, dt={dt} -> {suffix}")
    plot_spectrum(T, dt, outdir, suffix, COL_ANALYTIC, COL_UNITARY)
#    plot_reconstruction(T, dt, outdir, suffix, COL_ORIG, COL_REC)

print(f"\nВсе графики сохранены в папку: {outdir}")