import string

import numpy as np
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog


def open_heatmap(
    gui: ttk.Window,
    row_window: dict,
    cached_metrics: dict,
    gui_theme_hex: str,
    min_freq: float,
    smooth: int,
    filename: str = "",
) -> None:
    """
    Open a Toplevel window showing two plate heatmaps.

    Displays one heatmap for beat frequency [Hz] and one for mean
    relative contraction Force[%], arranged as an 8-row × 6-column
    plate layout.  Color limits are set robustly via median ± k·MAD to
    suppress the influence of outlier wells (e.g. dead or artefact wells).
    A colorbar (blue → red) is shown left of each heatmap.

    Parameters
    ----------
    gui : ttk.Window
        Parent window.
    row_window : dict[int, RowWindow]
        Mapping of well index → RowWindow; used as fallback if a well
        is absent from ``cached_metrics``.
    cached_metrics : dict[int, dict]
        Pre-computed metrics from ``GUI._cached_metrics``.  Wells
        missing from the cache are computed on the fly.
    gui_theme_hex : str
        Background colour of the active ttkbootstrap theme.
    min_freq : float
        Current max-frequency parameter [Hz]; used for on-the-fly
        computation only.
    smooth : int
        Current smoothing kernel width; used for on-the-fly computation
        only.
    filename : str, optional
        Base name (no extension) used as the default save filename.

    Notes
    -----
    Color limit derivation (k = 2.0):

    .. code-block:: text

        center = median(values)
        MAD    = median(|values − center|)
        vmin   = center − k · MAD
        vmax   = center + k · MAD
    """
    win = ttk.Toplevel("Heatmap", master=gui)
    win.geometry("1600x900")

    win.columnconfigure(0, weight=0, minsize=100)
    win.columnconfigure(1, weight=1)
    win.rowconfigure(0, weight=0, minsize=60)
    win.rowconfigure(1, weight=1)

    fig, axes = plt.subplots(1, 2, figsize=(1, 1))
    fig.patch.set_facecolor(gui_theme_hex)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.4)

    ttk.Button(
        win, text="Save Heatmap",
        command=lambda: _save_heatmap(fig, filename),
    ).grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    plate_rows, plate_cols = 8, 6
    freq_matrix        = np.zeros((plate_rows, plate_cols))
    contraction_matrix = np.zeros((plate_rows, plate_cols))

    for idx, rw in row_window.items():
        r, c = idx // plate_cols, idx % plate_cols
        m = cached_metrics.get(idx) or rw.compute_metrics(min_freq, smooth)
        freq_matrix[r, c]        = m["freq"]
        contraction_matrix[r, c] = m["mean_contraction"]

    k = 2.0
    datasets = [
        (freq_matrix,        "Frequency [Hz]"),
        (contraction_matrix, "Force [%]"),
    ]

    for ax, (data, title) in zip(axes, datasets):
        flat   = data.flatten()
        center = float(np.median(flat))
        mad    = float(np.median(np.abs(flat - center)))
        vmin   = center - k * mad
        vmax   = center + k * mad

        im   = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, location="left", pad=0.18, fraction=0.046)
        cbar.set_label(title, fontsize=9, color="white")
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

        ax.set_xticks(range(plate_cols))
        ax.set_xticklabels([str(i + 1) for i in range(plate_cols)], color="white")
        ax.set_yticks(range(plate_rows))
        ax.set_yticklabels(list(string.ascii_uppercase[:plate_rows]), color="white")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=11, pad=10)

        for r in range(plate_rows):
            for c in range(plate_cols):
                ax.text(c, r, f"{data[r, c]:.1f}",
                        ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().grid(column=1, row=1, sticky="nsew")
    canvas.get_tk_widget().config(width=1, height=1)
    canvas.draw()

    win.protocol("WM_DELETE_WINDOW", win.destroy)


def _save_heatmap(fig: plt.Figure, filename: str = "") -> None:
    """
    Open a save-as dialog and write ``fig`` to a PDF file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The heatmap figure to save.
    filename : str, optional
        Default filename stem (no extension).
    """
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Save Heatmap",
        initialfile=f"{filename}_heatmap" if filename else "heatmap",
    )
    if file_path:
        fig.savefig(file_path, format="pdf", bbox_inches="tight")