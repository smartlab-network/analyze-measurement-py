import os

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from tkinter import filedialog
from scipy.ndimage import minimum_filter1d


class GUI:

    """
    - **Header** (row 0): CSV load button.
    - **Left panel** (col 0): parameter controls — visible rows, maximum
      beat frequency, and smoothing kernel width.
    - **Plot area** (col 1): scrollable stack of per-well subplots with
      inline metric labels (frequency, Force[%]) and well-name buttons
      rendered directly as Matplotlib axes.
    - **Scrollbar** (col 3): vertical scroll control linked to the
      subplot stack.

    Parameters
    ----------
    plate_rows : int, optional
        Number of rows on the well plate. Default is 8.
    plate_columns : int, optional
        Number of columns on the well plate. Default is 6.

    Attributes
    ----------
    wells : int
        Total number of wells: ``plate_rows * plate_columns``.
    rows_var : ttk.IntVar
        Number of subplots rendered simultaneously. Range ``[1, 48]``.
    min_freq_var : ttk.DoubleVar
        Maximum expected beat frequency [Hz]. Sets the minimum allowed
        RR distance for peak detection: ``min_spacing = 1 / min_freq``.
    smooth_var : ttk.IntVar
        Moving-average kernel width for signal pre-smoothing. Range
        ``[1, 99]``; forced odd internally.
    start_idx : int
        Zero-based index of the first well currently visible in the
        subplot stack.
    row_window : dict[int, RowWindow]
        Maps well index → :class:`RowWindow` instance.
    gui_theme_hex : str
        Background colour of the active ttkbootstrap theme, used to
        match the Matplotlib figure background.
    """
    def __init__(self, plate_rows: int = 8, plate_columns: int = 6):
        self.gui = ttk.Window(title="Analyze measurement", themename="superhero")
        self.gui.geometry("1920x1080")

        self.gui_theme_hex = self.gui.style.colors.bg

        self.wells = plate_rows * plate_columns

        self.rows_var = ttk.IntVar(value=10)
        self.height_spacing_var = ttk.DoubleVar(value=1.4)
        self.min_freq_var = ttk.DoubleVar(value=1.2)
        self.smooth_var = ttk.IntVar(value=3)

        self.loaded_file = False
        self.filename: str = ""
        self.start_idx = 0
        self.mouse_inside_plot = False
        self.zoom_active = False

        self.gui.rowconfigure(0, weight=0, minsize=60)
        self.gui.rowconfigure(1, weight=1)
        self.gui.columnconfigure(0, weight=0, minsize=120)
        self.gui.columnconfigure(1, weight=1)
        self.gui.columnconfigure(3, weight=0, minsize=20)

        self.header_frame = ttk.Frame(self.gui)
        self.header_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=20)
        self.header_frame.columnconfigure(0, weight=0, minsize=120)
        self.header_frame.columnconfigure(1, weight=0, minsize=120)
        self.header_frame.columnconfigure(2, weight=1)
        self.header_frame.rowconfigure(0, weight=2)

        self.left_panel = ttk.Frame(self.gui)
        self.left_panel.grid(row=1, column=0, sticky="ns")

        for i in range(9):
            self.left_panel.rowconfigure(i, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=2)
        self.left_panel.columnconfigure(2, weight=1)

        ttk.Button(self.header_frame, text="CSV", command=self.load_file).grid(
            row=0, column=0, pady=5, padx=5, sticky = "nsew"
        )

        ttk.Button(self.header_frame, text = "Heatmap", command=self.heat_map_callback).grid(
            row = 0, column = 1, pady= 5, padx = 5, sticky = "nsew"
        )

        ttk.Label(self.left_panel, text="Rows:", font=("Helvetica", 16), anchor="c").grid(
            row=1, column=0, sticky="new", columnspan=3
        )
        self.rows_entry = ttk.Entry(self.left_panel, textvariable=self.rows_var, justify="center")
        self.rows_entry.grid(row=1, column=1, sticky="ew")
        self.rows_entry.bind("<Return>", self.on_rows_changed)
        ttk.Button(self.left_panel, text="<",
                   command=lambda: [self.rows_var.set(max(1, self.rows_var.get() - 1)),
                                    self.on_rows_changed(None)]
                   ).grid(row=1, column=0, sticky="ew", padx=(8, 0))
        ttk.Button(self.left_panel, text=">",
                   command=lambda: [self.rows_var.set(min(48, self.rows_var.get() + 1)),
                                    self.on_rows_changed(None)]
                   ).grid(row=1, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(self.left_panel, text="Max Frq in hz", font=("Helvetica", 16), anchor="c").grid(
            row=2, column=0, sticky="new", padx=8, columnspan=3
        )
        self.distance_entry = ttk.Entry(self.left_panel, textvariable=self.min_freq_var, justify="center")
        self.distance_entry.grid(row=2, column=1, sticky="ew")
        self.distance_entry.bind("<Return>", self.on_distance_entry)
        ttk.Button(self.left_panel, text="<",
                   command=lambda: [self.min_freq_var.set(round(max(0.1, self.min_freq_var.get() - 0.1), 1)),
                                    self.on_distance_entry(None)]
                   ).grid(row=2, column=0, sticky="ew", padx=(8, 0))
        ttk.Button(self.left_panel, text=">",
                   command=lambda: [self.min_freq_var.set(round(self.min_freq_var.get() + 0.1, 1)),
                                    self.on_distance_entry(None)]
                   ).grid(row=2, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(self.left_panel, text="Smooth", font=("Helvetica", 16), anchor="c").grid(
            row=3, column=0, sticky="new", padx=8, columnspan=3
        )
        self.smooth_entry = ttk.Entry(self.left_panel, textvariable=self.smooth_var, justify="center")
        self.smooth_entry.grid(row=3, column=1, sticky="ew")
        self.smooth_entry.bind("<Return>", self.on_smooth_changed)
        ttk.Button(self.left_panel, text="<",
                   command=lambda: [self.smooth_var.set(max(1, self.smooth_var.get() - 1)),
                                    self.on_smooth_changed(None)]
                   ).grid(row=3, column=0, sticky="ew", padx=(8, 0))
        ttk.Button(self.left_panel, text=">",
                   command=lambda: [self.smooth_var.set(min(99, self.smooth_var.get() + 1)),
                                    self.on_smooth_changed(None)]
                   ).grid(row=3, column=2, sticky="ew", padx=(0, 8))

        self.plot_frame = ttk.Frame(self.gui)
        self.plot_frame.grid(row=1, column=1, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.scrollbar = ttk.Scrollbar(self.gui, orient="vertical", command=self.on_scrollbar)
        self.scrollbar.grid(row=0, column=3, sticky="ns", rowspan = 3)

        self.gui.bind_all("<MouseWheel>", self._on_mousewheel)

        self.row_window: dict[int, RowWindow] = {}

    def load_file(self) -> None:
        """
        Open a file-chooser dialog, load a CSV and trigger plot creation.

        The CSV must have shape ``(n_samples, 1 + n_wells)`` where
        column 0 is the time vector [s] and columns ``1…n_wells`` are
        per-well distance signals [px]. Calls :meth:`create_plots` on
        success; returns silently if the dialog is cancelled.
        """
        path = filedialog.askopenfilename(
            title="CSV auswählen", filetypes=[("CSV Dateien", "*.csv")]
        )
        if not path:
            return

        file = os.path.basename(path)
        self.filename = os.path.splitext(file)[0]

        self.loaded_file = True
        self.data = np.loadtxt(path, delimiter=",")
        self.create_plots()

    def heat_map_callback(self):
        """
            Open a Toplevel window showing two plate heatmaps.

            Displays one heatmap for beat frequency [Hz] and one for mean
            relative contraction Force[%], both arranged as an 8-row x 6-column
            plate layout.  Color limits are set robustly via median ± k·MAD to
            suppress the influence of outlier wells (e.g. dead or artefact wells).
            A shared colorbar (blue → red) is shown on the left of each heatmap.

            Notes
            -----
            Color limit derivation (k = 2.0):

            .. code-block:: text

                center = median(values)
                MAD    = median(|values − center|)
                vmin   = center − k · MAD
                vmax   = center + k · MAD
            """
        if not self.loaded_file:
            return
        win = ttk.Toplevel("Heatmap", master=self.gui)
        win.geometry("1600x900")

        win.columnconfigure(0, weight=0,minsize=100)
        win.columnconfigure(1, weight=1)
        win.rowconfigure(0, weight=0, minsize=60)
        win.rowconfigure(1, weight=1)

        fig, axes = plt.subplots(1, 2, figsize=(1, 1))
        fig.patch.set_facecolor(self.gui_theme_hex)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.4)

        ttk.Button(win, text = "Safe Heatmap", command=lambda: self.callback_safe_heatmap(fig)).grid(
            row = 0, column = 0, sticky ="nw", padx = 5, pady = 5)

        plate_rows, plate_cols = 8, 6
        freq_matrix = np.zeros((plate_rows, plate_cols))
        contraction_matrix = np.zeros((plate_rows, plate_cols))

        for idx, rw in self.row_window.items():
            r, c = idx // plate_cols, idx % plate_cols
            m = rw.compute_metrics(self.min_freq_var.get(), self.smooth_var.get())
            freq_matrix[r, c] = m["freq"]
            contraction_matrix[r, c] = m["mean_contraction"]

        k = 2.0
        datasets = [
            (freq_matrix, "Frequency [Hz]"),
            (contraction_matrix, "Force [%]"),
        ]

        for ax, (data, title) in zip(axes, datasets):
            flat = data.flatten()

            center = float(np.median(flat))
            mad = float(np.median(np.abs(flat - center)))
            vmin = center - k * mad
            vmax = center + k * mad

            im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto",
                           vmin=vmin, vmax=vmax)

            cbar = fig.colorbar(im, ax=ax, location="left", pad=0.18, fraction=0.046)
            cbar.set_label(title, fontsize=9, color="white")
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

            # Tick labels: columns 1–6, rows A–H
            ax.set_xticks(range(plate_cols))
            ax.set_xticklabels([str(i + 1) for i in range(plate_cols)], color="white")
            ax.set_yticks(range(plate_rows))
            ax.set_yticklabels(list(string.ascii_uppercase[:plate_rows]), color="white")
            ax.tick_params(colors="white")
            ax.set_title(title, color="white", fontsize=11, pad=10)

            # Value annotation per cell
            for r in range(plate_rows):
                for c in range(plate_cols):
                    ax.text(c, r, f"{data[r, c]:.1f}",
                            ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().grid(column = 1, row=1, sticky="nsew")
        canvas.draw()

        win.protocol("WM_DELETE_WINDOW", win.destroy)

    def callback_safe_heatmap(self, fig: plt.Figure):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Safe Heatmap",
            initialfile=f"{self.filename}_heatmap"
        )

        if file_path:
            fig.savefig(file_path, format="pdf", bbox_inches="tight")

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the loaded data matrix into time and signal arrays.

        Returns
        -------
        time : np.ndarray, shape (n,)
            Time vector in seconds.
        values : np.ndarray, shape (n, wells)
            Per-well distance signals in pixels.
        """
        return self.data[:, 0], self.data[:, 1:]

    def create_plots(self) -> None:
        """
        Build or rebuild the Matplotlib figure, canvas, and RowWindow objects.

        Destroys the existing canvas widget and closes the old figure
        (preventing memory leaks) before constructing fresh ones.  All
        Toplevel detail windows from the previous load are also destroyed.

        The figure is initialised with ``figsize=(1, 1)`` so that the
        Tkinter grid — not the figure's intrinsic size — controls the
        rendered dimensions.  ``widget.config(width=1, height=1)`` enforces
        the same constraint on the canvas widget.

        Notes
        -----
        Figure margins (in figure-coordinate units, range 0–1):

        .. code-block:: text

            left  = 0.10  →  space for metric text labels
            right = 0.88  →  space for well-name buttons
            hspace = 0    →  subplots flush against each other
        """
        time, values = self.get_data()

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)

        for rw in self.row_window.values():
            rw.window.destroy()
        self.row_window = {}

        rows = self.rows_var.get()

        self.fig, self.axes = plt.subplots(rows, 1, figsize=(1, 1))
        self.fig.subplots_adjust(top=0.98, bottom=0.04, left=0.08, right=0.90, hspace=0)
        self.fig.patch.set_facecolor(self.gui_theme_hex)

        if not isinstance(self.axes, np.ndarray):
            self.axes = [self.axes]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        widget.config(width=1, height=1)

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_leave)
        self.canvas.mpl_connect("scroll_event", self.on_zoom)

        for row_idx in range(self.wells):
            row_letter = string.ascii_uppercase[row_idx // 6]
            col_number = (row_idx % 6) + 1
            name = f"{row_letter}{col_number}"
            self.row_window[row_idx] = RowWindow(
                row_idx, name, time, values[:, row_idx], gui=self.gui
            )

        self.update_plots()

    def update_plots(self) -> None:
        """
        Redraw all visible subplots and refresh metric labels and buttons.

        For each visible well the method:

        1. Clears the subplot and calls :meth:`RowWindow.plot`, which
           returns a ``metrics`` dict (see :meth:`RowWindow.compute_metrics`).
        2. Renders ``freq`` [Hz] and ``mean_contraction`` [%] as two
           left-aligned columns in the figure left margin, with a shared
           column header and separator line above the first subplot.
        3. Refreshes well-name buttons via :meth:`update_buttons` and
           synchronises the scrollbar via :meth:`update_scrollbar`.

        Column x-positions are derived from the figure left margin so that
        the label columns scale correctly when the window is resized:

        .. code-block:: text

            fig_left  = subplots_adjust(left)   e.g. 0.10
            X_HZ      = fig_left * 0.05         leftmost column
            X_FORCE   = fig_left * 0.50         right column, centred in margin
        """
        if hasattr(self, '_metric_texts'):
            for txt in self._metric_texts:
                txt.remove()
        self._metric_texts = []

        # Derive column positions from the actual figure left margin
        fig_left = self.fig.subplotpars.left  # e.g. 0.10 — set in create_plots
        X_HZ = fig_left * 0.05
        X_FORCE = fig_left * 0.52

        first_pos = self.axes[0].get_position()
        header_y = first_pos.y0 + first_pos.height + 0.004

        # Column headers
        for x, label in [(X_HZ, "Hz"), (X_FORCE, "F[%]")]:
            txt = self.fig.text(
                x, header_y, label,
                va="bottom", ha="left",
                fontsize=10, color="red", fontstyle="italic",
            )
            self._metric_texts.append(txt)

        # Separator line under headers
        line = self.fig.add_artist(
            plt.Line2D(
                [X_HZ, fig_left * 0.97], [header_y, header_y],
                transform=self.fig.transFigure,
                color="white", linewidth=0.6, alpha=0.5,
            )
        )
        self._metric_texts.append(line)

        for i, ax in enumerate(self.axes):
            ax.clear()
            data_idx = self.start_idx + i
            if data_idx >= self.wells:
                continue

            metrics = self.row_window[data_idx].plot(
                ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get(),
            )

            pos = ax.get_position()
            y_center = pos.y0 + pos.height / 2

            for x, val in [(X_HZ, metrics["freq"]),
                           (X_FORCE, metrics["mean_contraction"])]:
                txt = self.fig.text(
                    x, y_center,
                    f"{val:.1f}",
                    va="center", ha="left",
                    fontsize=8, color="white",
                )
                self._metric_texts.append(txt)

        self.fig.supxlabel("Time in s", fontsize=10)
        self.canvas.draw_idle()
        self.update_buttons()
        self.update_scrollbar()

    def update_buttons(self) -> None:
        """
        Rebuild well-name buttons as Matplotlib axes aligned to subplots.

        Each button occupies a new ``Axes`` whose position in figure
        coordinates is taken directly from the corresponding subplot via
        ``ax.get_position()``, guaranteeing pixel-perfect vertical
        alignment regardless of the number of visible rows or window size.

        Old button axes are explicitly removed before new ones are added.
        Button objects are kept in ``_btn_objects`` to prevent the garbage
        collector from destroying the Matplotlib widget callbacks.

        Notes
        -----
        Button axes position (figure coordinates):

        .. code-block:: text

            x      = 0.89   (starts immediately after right plot margin)
            y      = pos.y0 (flush with subplot bottom)
            width  = 0.10
            height = pos.height  (identical to subplot height)
        """
        if hasattr(self, '_btn_axes'):
            for bax in self._btn_axes:
                bax.remove()
        self._btn_axes = []
        self._btn_objects = []

        rows = self.rows_var.get()
        for i in range(rows):
            data_idx = self.start_idx + i
            if data_idx >= self.wells:
                break

            ax = self.axes[i]
            pos = ax.get_position()

            btn_ax = self.fig.add_axes([0.89, pos.y0, 0.10, pos.height])
            self._btn_axes.append(btn_ax)

            btn = Button(btn_ax, self.row_window[data_idx].row_name)
            btn.on_clicked(lambda _, idx=data_idx: self.open_row(idx))
            self._btn_objects.append(btn)

        self.canvas.draw_idle()

    def on_zoom(self, event) -> None:
        """
        Zoom the x-axis of the hovered subplot on scroll-wheel events.

        Zooming is centred on the cursor's x-position so that the data
        under the cursor stays fixed.  Scroll **up** narrows the view
        (zoom in, scale factor ``1/1.2``); scroll **down** widens it
        (zoom out, scale factor ``1.2``).  Sets ``zoom_active = True``
        to suppress hover-dimming while zooming.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Scroll event forwarded from the Matplotlib canvas.

        Notes
        -----
        New x-limits after zoom:

        .. code-block:: text

            x_range  = (xlim[1] - xlim[0]) * scale
            relx     = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            new_xlim = [xdata - x_range * relx,
                        xdata + x_range * (1 - relx)]
        """
        if event.inaxes is None:
            return
        self.zoom_active = True
        base_scale = 1.2
        scale = 1 / base_scale if event.button == "up" else base_scale if event.button == "down" else None
        if scale is None:
            return
        xlim = event.inaxes.get_xlim()
        xdata = event.xdata
        if xdata is None:
            return
        x_range = (xlim[1] - xlim[0]) * scale
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        event.inaxes.set_xlim([xdata - x_range * relx, xdata + x_range * (1 - relx)])
        self.canvas.draw_idle()

    def on_hover(self, event) -> None:
        """
        Dim all subplots to alpha 0.3 except the one under the cursor.

        Provides a visual focus cue when the user moves the mouse over
        the plot area.  No action is taken while ``zoom_active`` is set.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Motion event forwarded from the Matplotlib canvas.
        """
        self.mouse_inside_plot = True
        if self.zoom_active:
            return
        for ax in self.axes:
            ax.set_alpha(0.3)
        if event.inaxes:
            event.inaxes.set_alpha(1.0)
        self.canvas.draw_idle()

    def on_leave(self, event) -> None:
        """
        Restore full opacity for all subplots when the cursor leaves the figure.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Figure-leave event forwarded from the Matplotlib canvas.
        """
        self.mouse_inside_plot = False
        for ax in self.axes:
            ax.set_alpha(1.0)
        self.canvas.draw_idle()

    def update_scrollbar(self) -> None:
        """
        Synchronise the scrollbar thumb with the current subplot stack position.

        The thumb spans the fractional interval
        ``[start_idx / max_start, (start_idx + rows) / wells]``
        as required by ``ttk.Scrollbar.set``.
        """
        total = self.wells - self.rows_var.get()
        if total <= 0:
            self.scrollbar.set(0, 1)
            return
        start = self.start_idx / total
        end = (self.start_idx + self.rows_var.get()) / self.wells
        self.scrollbar.set(start, end)

    def _on_mousewheel(self, event) -> None:
        """
        Scroll the subplot stack on mouse-wheel events outside the plot area.

        Ignored when the cursor is inside the plot area (scroll is
        consumed by :meth:`on_zoom`) or when a Toplevel detail window
        holds keyboard focus.

        Parameters
        ----------
        event : tkinter.Event
            MouseWheel event from the Tkinter event loop.
            ``event.delta`` is ``±120`` per notch on Windows/macOS.
        """
        if not hasattr(self, 'axes'):
            return
        focused = self.gui.focus_get()
        if focused is not None and str(focused.winfo_toplevel()) != str(self.gui):
            return
        if self.mouse_inside_plot:
            return
        direction = int(-event.delta / 120) if event.delta else 0
        self.start_idx += direction
        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def on_scrollbar(self, *args) -> None:
        """
        Handle scrollbar drag/click events and redraw the subplot stack.

        Parameters
        ----------
        *args : tuple
            Passed by Tkinter.  When ``args[0] == 'moveto'``, ``args[1]``
            is the fractional target position in ``[0, 1]``.
        """
        if not hasattr(self, 'axes'):
            return
        if args[0] == "moveto":
            fraction = float(args[1])
            max_start = self.wells - self.rows_var.get()
            self.start_idx = int(fraction * max_start)
        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def open_row(self, idx: int) -> None:
        """
        Open or raise the detail Toplevel window for well ``idx``.

        Parameters
        ----------
        idx : int
            Well index into :attr:`row_window`.
        """
        self.row_window[idx].open(
            min_frq=self.min_freq_var.get(),
            smooth_buffer=self.smooth_var.get(),
        )

    def on_rows_changed(self, event) -> None:
        """
        Validate the rows control and rebuild the figure.

        Clamps :attr:`rows_var` to ``[1, 48]``, then calls
        :meth:`create_plots`.  The ``AttributeError`` guard handles the
        case where no CSV has been loaded yet.
        """
        if self.rows_var.get() < 1:
            self.rows_var.set(1)
        if self.rows_var.get() > 48:
            self.rows_var.set(48)
        self.start_idx = 0
        try:
            self.create_plots()
        except AttributeError:
            pass

    def on_distance_entry(self, event) -> None:
        """
        Validate the max-frequency control and redraw plots.

        Clamps :attr:`min_freq_var` to a minimum of ``0.1`` Hz.
        """
        if self.min_freq_var.get() < 0.1:
            self.min_freq_var.set(0.1)
        try:
            self.update_plots()
        except AttributeError:
            pass

    def on_smooth_changed(self, event=None) -> None:
        """
        Validate the smooth control and redraw plots.

        Clamps :attr:`smooth_var` to a minimum of ``1``.
        """
        if self.smooth_var.get() < 1:
            self.smooth_var.set(1)
        try:
            self.update_plots()
        except AttributeError:
            pass


class RowWindow:
    """
    Per-well data container, signal processor, metric engine, and detail view.

    Encapsulates the raw time-series of one well together with all
    signal-processing logic (smoothing, baseline estimation, peak
    detection) and the derived contractility metrics.  Also manages a
    lazily created Toplevel window with a full-resolution interactive
    plot of the well.

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

    Attributes
    ----------
    detail_fig : matplotlib.figure.Figure or None
        Figure embedded in the detail Toplevel; created lazily on first
        call to :meth:`open`.
    detail_canvas : FigureCanvasTkAgg or None
        Canvas embedding :attr:`detail_fig` into the Toplevel widget.
    detail_ax : matplotlib.axes.Axes or None
        Single axes inside :attr:`detail_fig`.
    """

    def __init__(
        self,
        row_idx: int,
        row_name: str,
        time: np.ndarray,
        values: np.ndarray,
        gui: ttk.Window,
    ):
        self.row_idx = row_idx
        self.row_name = row_name
        self.time = time
        self.values = values

        self.window = ttk.Toplevel(self.row_name, master=gui)
        self.window.geometry("1920x720")
        self.window.withdraw()

        self.detail_fig = None
        self.detail_canvas = None
        self.detail_ax = None

        self.metrics: dict = {}

    def smooth(self, values: np.ndarray, buffer: int) -> np.ndarray:
        """
        Moving-average smoothing with edge preservation.

        Convolves ``values`` with a uniform kernel of width ``buffer``
        using ``np.convolve(..., mode='valid')`` to avoid zero-padding
        artefacts at the signal boundaries.  The ``half = buffer // 2``
        edge samples at each end are filled with the original unsmoothed
        values so the returned array has the same length as the input.

        Parameters
        ----------
        values : np.ndarray, shape (n,)
            Input signal (raw or previously processed).
        buffer : int
            Kernel width in samples.  Incremented by 1 if even (forced
            odd for symmetric centring).  Values ``< 2`` return a copy
            of the input unchanged.

        Returns
        -------
        result : np.ndarray, shape (n,)
            Smoothed signal with original-value edges.

        Notes
        -----
        For odd kernel width ``w`` and ``half = w // 2``:

        .. code-block:: text

            result[i] = mean(values[i - half : i + half + 1])
                        for i in [half, n - half)
            result[i] = values[i]   otherwise  (edge fill)
        """
        if buffer < 2:
            return values.copy()
        if buffer % 2 == 0:
            buffer += 1
        half = buffer // 2
        kernel = np.ones(buffer) / buffer
        convolved = np.convolve(values, kernel, mode='valid')
        result = values.copy()
        result[half: len(values) - half] = convolved
        return result

    def running_max(self, values: np.ndarray, window_sec: float, dt: float) -> np.ndarray:
        """
        Diastolic baseline estimated as a slow running maximum (runMax).

        Applies ``scipy.ndimage.maximum_filter1d`` with a window that
        spans several beat periods.  Because the window is much wider
        than a single contraction, only the slow relaxation envelope
        of the signal is tracked, i.e. the diastolic (maximum pixel
        distance) state.

        Parameters
        ----------
        values : np.ndarray, shape (n,)
            Smoothed inter-pole distance signal in pixels.
        window_sec : float
            Filter window duration in seconds.  Must satisfy
            ``window_sec >> 1 / beat_frequency`` to avoid capturing
            individual contractions.  Default caller value: ``2.0 s``.
        dt : float
            Sampling interval in seconds (``1 / sampling_rate``).

        Returns
        -------
        run_max : np.ndarray, shape (n,)
            Diastolic baseline signal in pixels.

        Notes
        -----
        Window size derivation (forced odd for symmetric centring):

        .. code-block:: text

            w_samples = ceil(window_sec / dt)  →  forced odd
            runMax[i] = max(values[i - w//2 : i + w//2 + 1])
        """
        from scipy.ndimage import maximum_filter1d
        window_samples = max(1, int(window_sec / dt))
        if window_samples % 2 == 0:
            window_samples += 1
        return maximum_filter1d(values, size=window_samples)

    def find_peaks(
        self, values: np.ndarray, min_frq: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect systolic peaks (local minima) with guaranteed minimum spacing.

        Uses a two-stage algorithm:

        **Stage 1 — candidate detection.**
        ``minimum_filter1d`` with kernel size ``2 * min_spacing + 1``
        marks every sample that equals the local minimum within its
        neighbourhood as a candidate.

        **Stage 2 — greedy deduplication.**
        Iterates candidates in chronological order and enforces a hard
        minimum RR distance of ``min_spacing`` samples between accepted
        peaks.  When a new candidate falls within the exclusion zone of
        the previously accepted peak, the one with the smaller pixel
        value (deeper contraction) replaces the other.

        Parameters
        ----------
        values : np.ndarray, shape (n,)
            Smoothed signal in pixels.
        min_frq : float
            Maximum expected beat frequency in Hz.  Determines the
            minimum allowed RR distance:

            .. code-block:: text

                min_spacing [samples] = floor(1 / min_frq / dt)

        dt : float
            Sampling interval in seconds.

        Returns
        -------
        peak_idx : np.ndarray, shape (k,)
            Sample indices of accepted peaks, satisfying
            ``peak_idx[i+1] - peak_idx[i] >= min_spacing`` for all ``i``.
        peak_val : np.ndarray, shape (k,)
            Signal values at accepted peaks in pixels.

        Notes
        -----
        .. code-block:: text

            min_spacing [samples] = floor(1 / min_frq / dt)
            kernel_size           = 2 * min_spacing + 1
        """
        min_spacing = int((1.0 / min_frq) / dt)
        kernel_size = 2 * min_spacing + 1

        window_min = minimum_filter1d(values, size=kernel_size)
        candidates = np.where(values == window_min)[0]

        if len(candidates) == 0:
            return np.array([], dtype=int), np.array([])

        peaks = [candidates[0]]
        for idx in candidates[1:]:
            if idx - peaks[-1] >= min_spacing:
                peaks.append(idx)
            elif values[idx] < values[peaks[-1]]:
                peaks[-1] = idx

        peak_idx = np.array(peaks)
        return peak_idx, values[peak_idx]

    def compute_metrics(
        self,
        min_frq: float,
        smooth_buffer: int,
        run_max_window_sec: float = 2.0,
    ) -> dict:
        """
        Compute all contractility metrics for this well.

        Executes the full analysis pipeline:
        smoothing → baseline estimation → peak detection →
        amplitude computation → relative contraction → frequency.

        Parameters
        ----------
        min_frq : float
            Maximum beat frequency in Hz (sets peak-detection spacing).
        smooth_buffer : int
            Moving-average kernel width passed to :meth:`smooth`.
        run_max_window_sec : float, optional
            Duration of the diastolic baseline window in seconds passed
            to :meth:`running_max`.  Default is ``2.0``.

        Returns
        -------
        metrics : dict
            Dictionary with the following entries:

            ``smoothed`` : np.ndarray, shape (n,)
                Moving-average smoothed signal [px].
            ``run_max`` : np.ndarray, shape (n,)
                Diastolic baseline (runMax) [px].
            ``peak_idx`` : np.ndarray, shape (k,)
                Sample indices of detected systolic peaks.
            ``peak_val`` : np.ndarray, shape (k,)
                Signal values at detected peaks [px] (neigMin).
            ``peak_height`` : np.ndarray, shape (k,)
                Absolute contraction amplitude per peak [px]:

                .. code-block:: text

                    peakHeight[i] = runMax[peak_idx[i]] - neigMin[i]

            ``contraction`` : np.ndarray, shape (k,)
                Relative contraction per peak [%]:

                .. code-block:: text

                    contraction[i] = peakHeight[i] / runMax[peak_idx[i]] * 100

            ``mean_contraction`` : float
                Arithmetic mean of ``contraction`` over all peaks [%].
                Returns ``0.0`` if no peaks were found.
            ``freq`` : float
                Beat frequency [Hz] derived from mean RR interval:

                .. code-block:: text

                    RR[i]  = time[peak_idx[i+1]] - time[peak_idx[i]]  [s]
                    meanRR = mean(RR)                                  [s]
                    freq   = 1 / meanRR                                [Hz]

                Returns ``0.0`` if fewer than 2 peaks were found.

        Notes
        -----
        Sign convention (Y-axis displayed inverted):

        .. code-block:: text

            large px value  →  poles far apart  →  diastole  →  runMax
            small px value  →  poles close      →  systole   →  neigMin
            peakHeight = runMax - neigMin  ≥ 0  always

        Frequency is derived from the mean RR interval rather than a
        simple peak count to be robust against spurious detections at
        the recording boundaries.
        """
        dt = float(self.time[1] - self.time[0])

        smoothed = self.smooth(self.values, smooth_buffer)
        run_max = self.running_max(smoothed, run_max_window_sec, dt)
        peak_idx, peak_val = self.find_peaks(smoothed, min_frq, dt)

        run_max_at_peaks = run_max[peak_idx]
        peak_height = run_max_at_peaks - peak_val

        with np.errstate(invalid='ignore', divide='ignore'):
            contraction = np.where(
                run_max_at_peaks > 0,
                peak_height / run_max_at_peaks * 100,
                0.0,
            )

        mean_contraction = float(np.mean(contraction)) if len(contraction) > 0 else 0.0

        if len(peak_idx) >= 2:
            t_peaks = self.time[peak_idx]
            rr_intervals = np.diff(t_peaks)
            freq = float(1.0 / np.mean(rr_intervals))
        else:
            freq = 0.0

        self.metrics = dict(
            smoothed=smoothed,
            run_max=run_max,
            peak_idx=peak_idx,
            peak_val=peak_val,
            peak_height=peak_height,
            contraction=contraction,
            mean_contraction=mean_contraction,
            freq=freq,
        )
        return self.metrics


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
            Maximum beat frequency in Hz. Default is ``1.2``.
        smooth_buffer : int, optional
            Moving-average kernel width. Default is ``3``.
        show_y_labels : bool, optional
            If ``True``, add x/y axis labels.  Used in the detail
            Toplevel where the axes are large enough to be readable.
            Default is ``False``.
        metrics : dict or None, optional
            Pre-computed metrics dict from :meth:`compute_metrics`.
            If ``None``, metrics are computed internally.  Passing a
            pre-computed dict avoids redundant computation when the
            caller already holds the result.

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

    def _make_zoom_handler(self, canvas: FigureCanvasTkAgg):
        """
        Return a scroll-event callback that zooms the x-axis of ``canvas``.

        Uses a closure to capture ``canvas``, allowing the same factory
        to be used for both the main figure and each detail Toplevel.

        Parameters
        ----------
        canvas : FigureCanvasTkAgg
            Canvas whose active axes should be zoomed.

        Returns
        -------
        on_zoom : callable
            Matplotlib ``scroll_event`` callback with signature
            ``on_zoom(event) -> None``.
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
            ax = event.inaxes
            xlim = ax.get_xlim()
            xdata = event.xdata
            if xdata is None:
                return
            x_range = (xlim[1] - xlim[0]) * scale
            relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            ax.set_xlim([xdata - x_range * relx, xdata + x_range * (1 - relx)])
            canvas.draw_idle()

        return on_zoom

    def open(self, min_frq: float = 1.2, smooth_buffer: int = 3) -> None:
        """
        Open or raise the full-resolution detail Toplevel for this well.

        The Matplotlib figure and canvas are created lazily on the first
        call and reused on all subsequent calls; only the axes are
        cleared and redrawn with the current parameter values.  If the
        window is already mapped to the screen it is raised to the front
        without redrawing.

        The window's close button is rebound to ``withdraw`` rather than
        ``destroy`` so that the figure and canvas can be reused.

        Parameters
        ----------
        min_frq : float, optional
            Maximum beat frequency in Hz passed to :meth:`plot`.
            Default is ``1.2``.
        smooth_buffer : int, optional
            Moving-average kernel width passed to :meth:`plot`.
            Default is ``3``.
        """
        if self.window.winfo_ismapped():
            self.window.lift()
            return

        if self.detail_fig is None:
            self.detail_fig, self.detail_ax = plt.subplots(figsize=(16, 4))
            self.detail_fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)
            self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=self.window)
            self.detail_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.detail_canvas.mpl_connect(
                "scroll_event", self._make_zoom_handler(self.detail_canvas)
            )

        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)
        self.detail_ax.clear()
        self.plot(self.detail_ax, min_frq=min_frq, smooth_buffer=smooth_buffer, show_y_labels=True)
        self.detail_canvas.draw()
        self.window.deiconify()
        self.window.lift()


gui = GUI()
root = gui.gui
root.mainloop()