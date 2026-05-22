import numpy as np
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

import plot_math


class RowWindow:
    """
    Per-well data container, signal processor, metric engine, and detail view.

    Encapsulates the raw time-series of one well together with all
    signal-processing logic (smoothing, baseline estimation, peak
    detection) and the derived contractility metrics.  Also manages a
    lazily created Toplevel window with a full-resolution interactive
    plot and its own parameter controls.

    Parameters
    ----------
    row_idx : int
        Zero-based well index on the plate.
    row_name : str
        Human-readable well label, e.g. ``'A1'``, ``'B3'``.
    time : np.ndarray, shape (n,)
        Time vector in seconds.
    values : np.ndarray, shape (n,)
        Raw inter-pole distance signal in pixels.

        Sign convention (Y-axis displayed inverted):

        .. code-block:: text

            large px  →  poles far apart  →  tissue relaxed   (diastole)
            small px  →  poles close      →  tissue contracted (systole)

    gui : ttk.Window
        Parent window used when constructing the Toplevel detail window.
    min_frq_default : float
        Initial value for the detail window's max-frequency control [Hz].
    smooth_default : int
        Initial value for the detail window's smoothing control [samples].

    Attributes
    ----------
    detail_fig : matplotlib.figure.Figure or None
        Figure embedded in the detail Toplevel; created lazily on first
        call to :meth:`open`.
    detail_canvas : FigureCanvasTkAgg or None
        Canvas embedding :attr:`detail_fig` into the Toplevel widget.
    detail_ax : matplotlib.axes.Axes or None
        Single axes inside :attr:`detail_fig`.
    metrics : dict
        Most recently computed metrics dict; empty until first call to
        :meth:`compute_metrics`.
    """

    def __init__(
        self,
        row_idx: int,
        row_name: str,
        time: np.ndarray,
        values: np.ndarray,
        gui: ttk.Window,
        min_frq_default: float = 1.2,
        smooth_default: int = 3,
    ):
        self.row_idx  = row_idx
        self.row_name = row_name
        self.time     = time
        self.values   = values

        self.window = ttk.Toplevel(self.row_name, master=gui)
        self.window.geometry("1920x800")
        self.window.withdraw()

        self.detail_fig    = None
        self.detail_canvas = None
        self.detail_ax     = None
        self.metrics: dict = {}

        self.min_freq_var = ttk.DoubleVar(value=min_frq_default)
        self.smooth_var   = ttk.IntVar(value=smooth_default)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        min_frq: float,
        smooth_buffer: int,
        run_max_window_sec: float = 2.0,
    ) -> dict:
        """
        Compute and cache all contractility metrics for this well.

        Delegates to :func:`plot_math.compute_metrics` and stores the
        result in :attr:`metrics`.

        Parameters
        ----------
        min_frq : float
            Maximum beat frequency in Hz.
        smooth_buffer : int
            Moving-average kernel width.
        run_max_window_sec : float, optional
            Diastolic baseline window in seconds. Default ``2.0``.

        Returns
        -------
        metrics : dict
            See :func:`plot_math.compute_metrics` for full key listing.
        """
        self.metrics = plot_math.compute_metrics(
            self.time, self.values, min_frq, smooth_buffer, run_max_window_sec
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: plt.Axes,
        min_frq: float = 1.2,
        smooth_buffer: int = 3,
        show_y_labels: bool = False,
        metrics: dict | None = None,
    ) -> dict:
        """
        Render the well signal onto ``ax`` and return the metrics dict.

        Draws four data layers in order:

        1. **Raw signal** — semi-transparent (alpha 0.3), steel-blue.
        2. **Smoothed signal** — solid, steel-blue.
        3. **Diastolic baseline (runMax)** — gold, linewidth 1.2.
        4. **Detected peaks (neigMin)** — red dots, markersize 4.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        min_frq : float, optional
            Maximum beat frequency in Hz. Default ``1.2``.
        smooth_buffer : int, optional
            Moving-average kernel width. Default ``3``.
        show_y_labels : bool, optional
            If ``True``, add x/y axis labels. Default ``False``.
        metrics : dict or None, optional
            Pre-computed metrics dict. If ``None``, computed internally.

        Returns
        -------
        metrics : dict
            The metrics dict used for this plot call.
        """
        if metrics is None:
            metrics = self.compute_metrics(min_frq, smooth_buffer)

        ax.plot(self.time, self.values, alpha=0.3, color="steelblue")
        ax.plot(self.time, metrics["smoothed"], color="steelblue")
        ax.plot(self.time, metrics["run_max"], color="gold", linewidth=1.2)
        ax.plot(
            self.time[metrics["peak_idx"]],
            metrics["peak_val"],
            "ro", markersize=4,
        )
        ax.grid(True)
        ax.invert_yaxis()

        if not show_y_labels:
            ax.tick_params(left=False, labelleft=False)
        else:
            ax.set_xlabel("Time in s")
            ax.set_ylabel("Distance in pixel")

        return metrics

    # ------------------------------------------------------------------
    # Detail window helpers
    # ------------------------------------------------------------------

    def _make_zoom_handler(self, canvas: FigureCanvasTkAgg):
        """
        Return a scroll-event callback that zooms the x-axis of ``canvas``.

        Parameters
        ----------
        canvas : FigureCanvasTkAgg
            Canvas whose active axes should be zoomed.

        Returns
        -------
        on_zoom : callable
            Matplotlib ``scroll_event`` callback.
        """
        def on_zoom(event):
            if event.inaxes is None:
                return
            base_scale = 1.2
            if event.button == "up":
                scale = 1 / base_scale
            elif event.button == "down":
                scale = base_scale
            else:
                return
            ax    = event.inaxes
            xlim  = ax.get_xlim()
            xdata = event.xdata
            if xdata is None:
                return
            x_range = (xlim[1] - xlim[0]) * scale
            relx    = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            ax.set_xlim([xdata - x_range * relx, xdata + x_range * (1 - relx)])
            canvas.draw_idle()

        return on_zoom

    def _build_controls(self, parent: ttk.Frame) -> None:
        """
        Build the ``< entry >`` parameter controls inside ``parent``.

        Creates controls for Max Frq and Smooth that are independent of
        the main GUI and affect only this detail window.

        Parameters
        ----------
        parent : ttk.Frame
            Container frame for the controls.

        Notes
        -----
        Layout (4 rows x 3 cols):

        .. code-block:: text

            row 0: Max Frq label
            row 1: < entry >
            row 2: Smooth label
            row 3: < entry >
        """
        for i in range(4):
            parent.rowconfigure(i, weight=1)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.columnconfigure(2, weight=1)

        # Max Frq
        ttk.Label(parent, text="Max Frq in Hz", font=("Helvetica", 13), anchor="c").grid(
            row=0, column=0, columnspan=3, sticky="new", padx=8
        )
        frq_entry = ttk.Entry(parent, textvariable=self.min_freq_var, justify="center")
        frq_entry.grid(row=0, column=1, sticky="ew")
        frq_entry.bind("<Return>", self._on_param_changed)
        ttk.Button(parent, text="<",
                   command=lambda: [
                       self.min_freq_var.set(round(max(0.1, self.min_freq_var.get() - 0.1), 1)),
                       self._on_param_changed(None),
                   ]).grid(row=0, column=0, sticky="ew", padx=(8, 0))
        ttk.Button(parent, text=">",
                   command=lambda: [
                       self.min_freq_var.set(round(self.min_freq_var.get() + 0.1, 1)),
                       self._on_param_changed(None),
                   ]).grid(row=0, column=2, sticky="ew", padx=(0, 8))

        # Smooth
        ttk.Label(parent, text="Smooth", font=("Helvetica", 13), anchor="c").grid(
            row=1, column=0, columnspan=3, sticky="new", padx=8
        )
        smooth_entry = ttk.Entry(parent, textvariable=self.smooth_var, justify="center")
        smooth_entry.grid(row=1, column=1, sticky="ew")
        smooth_entry.bind("<Return>", self._on_param_changed)
        ttk.Button(parent, text="<",
                   command=lambda: [
                       self.smooth_var.set(max(1, self.smooth_var.get() - 1)),
                       self._on_param_changed(None),
                   ]).grid(row=1, column=0, sticky="ew", padx=(8, 0))
        ttk.Button(parent, text=">",
                   command=lambda: [
                       self.smooth_var.set(min(99, self.smooth_var.get() + 1)),
                       self._on_param_changed(None),
                   ]).grid(row=1, column=2, sticky="ew", padx=(0, 8))

    def _on_param_changed(self, event) -> None:
        """
        Validate controls and redraw the detail plot with updated parameters.

        Clamps ``min_freq_var`` to ``[0.1, ∞)`` and ``smooth_var`` to
        ``[1, 99]``, then redraws the detail axes.
        """
        if self.min_freq_var.get() < 0.1:
            self.min_freq_var.set(0.1)
        if self.smooth_var.get() < 1:
            self.smooth_var.set(1)
        if self.smooth_var.get() > 99:
            self.smooth_var.set(99)

        if self.detail_ax is not None:
            self.detail_ax.clear()
            self.plot(
                self.detail_ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get(),
                show_y_labels=True,
            )
            self.detail_canvas.draw_idle()

    def _save_pdf(self) -> None:
        """
        Open a save-as dialog and export the detail figure as a PDF.

        The default filename is the well name (e.g. ``'A1.pdf'``).
        Does nothing if the dialog is cancelled or the figure has not
        been created yet.
        """
        if self.detail_fig is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Plot",
            initialfile=f"{self.row_name}",
        )
        if file_path:
            self.detail_fig.savefig(file_path, format="pdf", bbox_inches="tight")

    # Public open
    def open(self, min_frq: float = 1.2, smooth_buffer: int = 3) -> None:
        """
        Open or raise the full-resolution detail Toplevel for this well.

        The figure, canvas, and control panel are created lazily on the
        first call and reused on subsequent ones; only the axes are
        cleared and redrawn.  If the window is already visible it is
        raised to the front without redrawing.

        Parameters
        ----------
        min_frq : float, optional
            Maximum beat frequency in Hz; syncs the control to the
            current main-GUI value on every open. Default ``1.2``.
        smooth_buffer : int, optional
            Smoothing kernel width; syncs on every open. Default ``3``.
        """
        if self.window.winfo_ismapped():
            self.window.lift()
            return

        self.min_freq_var.set(min_frq)
        self.smooth_var.set(smooth_buffer)

        if self.detail_fig is None:
            self.window.rowconfigure(0, weight=0)
            self.window.rowconfigure(1, weight=1)
            self.window.columnconfigure(0, weight=0, minsize=140)
            self.window.columnconfigure(1, weight=1)

            ttk.Button(
                self.window, text="Save as PDF",
                command=self._save_pdf,
                bootstyle="outline",
            ).grid(row=0, column=0, sticky="nw", padx=8, pady=(8, 0))

            ctrl_frame = ttk.Frame(self.window)
            ctrl_frame.grid(row=1, column=0, sticky="ns", padx=8, pady=30)
            self._build_controls(ctrl_frame)

            self.detail_fig, self.detail_ax = plt.subplots(figsize=(16, 4))
            self.detail_fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)

            self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=self.window)
            self.detail_canvas.get_tk_widget().grid(
                row=0, column=1, rowspan=2, sticky="nsew"
            )
            self.detail_canvas.mpl_connect(
                "scroll_event", self._make_zoom_handler(self.detail_canvas)
            )

        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)
        self.detail_ax.clear()
        self.plot(
            self.detail_ax,
            min_frq=self.min_freq_var.get(),
            smooth_buffer=self.smooth_var.get(),
            show_y_labels=True,
        )
        self.detail_canvas.draw()
        self.window.deiconify()
        self.window.lift()