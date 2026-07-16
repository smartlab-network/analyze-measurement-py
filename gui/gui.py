import os

import ttkbootstrap as ttk
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from tkinter import filedialog

from row_window import RowWindow
from heatmap import open_heatmap



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

    def heat_map_callback(self) -> None:
        if not self.loaded_file:
            return
        open_heatmap(
            gui=self.gui,
            row_window=self.row_window,
            cached_metrics=getattr(self, '_cached_metrics', {}),
            gui_theme_hex=self.gui_theme_hex,
            min_freq=self.min_freq_var.get(),
            smooth=self.smooth_var.get(),
            filename=self.filename,
        )

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
                row_idx, name, time, values[:, row_idx],
                gui=self.gui,
                min_frq_default=self.min_freq_var.get(),
                smooth_default=self.smooth_var.get(),
            )
            self.row_window[row_idx]._main_gui_instance = self

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
                show_manual_peaks_green=False
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

gui = GUI()
root = gui.gui
root.mainloop()