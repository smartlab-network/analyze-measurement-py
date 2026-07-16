import numpy as np
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

import plot_math


class RowWindow:
    """
    Per-well data container, signal processor, metric engine, and detail view.
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

        # Storage for user-placed peak times
        self.manual_peak_times: list[float] = []
        # Storage for auto peaks that were removed by user
        self.removed_auto_peak_times: list[float] = []

        # Undo history: list of actions that can be undone
        self._undo_stack: list[dict] = []

        # Store current zoom limits to restore after refresh
        self._stored_xlim: tuple[float, float] | None = None

        # Store reference to main GUI instance
        self._main_gui_instance = None

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
        """
        extra_peaks = np.array(self.manual_peak_times) if self.manual_peak_times else None

        self.metrics = plot_math.compute_metrics(
            self.time, self.values, min_frq, smooth_buffer, run_max_window_sec,
            extra_peak_times=extra_peaks,
            excluded_peak_times=np.array(self.removed_auto_peak_times) if self.removed_auto_peak_times else None,
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Undo logic
    # ------------------------------------------------------------------

    def _undo_last_action(self):
        """
        Undo the last peak modification action.
        """
        if not self._undo_stack:
            return

        action = self._undo_stack.pop()

        if action["type"] == "add_manual":
            peak_time = action["time"]
            for i, t in enumerate(self.manual_peak_times):
                if abs(t - peak_time) < 0.05:
                    self.manual_peak_times.pop(i)
                    break

        elif action["type"] == "remove_manual":
            peak_time = action["time"]
            if not any(abs(t - peak_time) < 0.05 for t in self.manual_peak_times):
                self.manual_peak_times.append(peak_time)
                self.manual_peak_times.sort()

        elif action["type"] == "remove_auto":
            peak_time = action["time"]
            for i, t in enumerate(self.removed_auto_peak_times):
                if abs(t - peak_time) < 0.05:
                    self.removed_auto_peak_times.pop(i)
                    break

        self._refresh_detail_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: plt.Axes,
        min_frq: float = 1.2,
        smooth_buffer: int = 3,
        show_y_labels: bool = False,
        show_manual_peaks_green: bool = True,
        metrics: dict | None = None,
    ) -> dict:
        """
        Render the well signal onto ``ax`` and return the metrics dict.
        """
        if metrics is None:
            metrics = self.compute_metrics(min_frq, smooth_buffer)

        ax.plot(self.time, self.values, alpha=0.3, color="steelblue")
        ax.plot(self.time, metrics["smoothed"], color="steelblue")
        ax.plot(self.time, metrics["run_max"], color="gold", linewidth=1.2)

        if show_manual_peaks_green and len(self.manual_peak_times) > 0:
            all_peak_times = self.time[metrics["peak_idx"]]
            auto_mask = np.ones(len(all_peak_times), dtype=bool)
            for mt in self.manual_peak_times:
                auto_mask &= ~(np.abs(all_peak_times - mt) < 0.05)

            if np.any(auto_mask):
                ax.plot(
                    all_peak_times[auto_mask],
                    metrics["peak_val"][auto_mask],
                    "ro", markersize=4,
                )

            manual_indices = np.searchsorted(self.time, self.manual_peak_times)
            manual_indices = np.clip(manual_indices, 0, len(self.time) - 1)
            manual_vals = metrics["smoothed"][manual_indices]
            ax.plot(
                self.manual_peak_times,
                manual_vals,
                "go", markersize=6, markeredgecolor="black",
                label="User peaks"
            )
            if ax.get_legend() is None:
                ax.legend(loc="upper right", fontsize=8)
        else:
            if len(metrics["peak_idx"]) > 0:
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
    # Click handlers for manual peak manipulation
    # ------------------------------------------------------------------

    def _on_plot_click(self, event):
        """
        Handle click events in the detail plot.
        Left-click (button 1): add peak at x-position
        Right-click (button 3): remove nearest peak (auto or manual)
        Shift + Right-click: undo last action
        """
        if event.inaxes != self.detail_ax:
            return

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        # Store current zoom before redrawing
        self._stored_xlim = self.detail_ax.get_xlim()

        if event.button == 1:  # Left click: add peak
            idx = np.abs(self.time - xdata).argmin()
            time_val = self.time[idx]

            if not any(abs(t - time_val) < 0.05 for t in self.manual_peak_times):
                self.manual_peak_times.append(time_val)
                self.manual_peak_times.sort()
                self._undo_stack.append({"type": "add_manual", "time": time_val})
                self._refresh_detail_plot()

        elif event.button == 3:  # Right click
            # Shift + Right click: undo last action
            if event.key == 'shift':
                self._undo_last_action()
                return

            # Single right click: remove nearest peak
            if "peak_idx" not in self.metrics or len(self.metrics["peak_idx"]) == 0:
                return

            all_peak_times = self.time[self.metrics["peak_idx"]]

            if len(all_peak_times) == 0:
                return

            nearest_idx = np.abs(all_peak_times - xdata).argmin()
            peak_to_remove = all_peak_times[nearest_idx]

            # Check if it's a manual peak -> remove from manual list
            removed = False
            for i, t in enumerate(self.manual_peak_times):
                if abs(t - peak_to_remove) < 0.05:
                    self.manual_peak_times.pop(i)
                    self._undo_stack.append({"type": "remove_manual", "time": t})
                    removed = True
                    break

            # Otherwise it's an auto peak -> add to removed list
            if not removed:
                already_removed = any(abs(t - peak_to_remove) < 0.05 for t in self.removed_auto_peak_times)
                if not already_removed:
                    self.removed_auto_peak_times.append(peak_to_remove)
                    self._undo_stack.append({"type": "remove_auto", "time": peak_to_remove})
                    removed = True

            if removed:
                self._refresh_detail_plot()

    def _refresh_detail_plot(self):
        """Refresh the detail plot after peak changes, preserving zoom."""
        if self.detail_ax is not None:
            self.detail_ax.clear()
            self.plot(
                self.detail_ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get(),
                show_y_labels=True,
                show_manual_peaks_green=True,
            )
            if self._stored_xlim is not None:
                self.detail_ax.set_xlim(self._stored_xlim)
            self.detail_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Detail window helpers
    # ------------------------------------------------------------------

    def _make_zoom_handler(self, canvas: FigureCanvasTkAgg):
        """Return a scroll-event callback that zooms the x-axis of ``canvas``."""
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

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build the parameter controls inside ``parent``."""
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
        """Validate controls and redraw the detail plot with updated parameters."""
        if self.min_freq_var.get() < 0.1:
            self.min_freq_var.set(0.1)
        if self.smooth_var.get() < 1:
            self.smooth_var.set(1)
        if self.smooth_var.get() > 99:
            self.smooth_var.set(99)

        if self.detail_ax is not None:
            self._stored_xlim = self.detail_ax.get_xlim()
            self.detail_ax.clear()
            self.plot(
                self.detail_ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get(),
                show_y_labels=True,
                show_manual_peaks_green=True,
            )
            if self._stored_xlim is not None:
                self.detail_ax.set_xlim(self._stored_xlim)
            self.detail_canvas.draw_idle()

    def _save_pdf(self) -> None:
        """Open a save-as dialog and export the detail figure as a PDF."""
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

    def _on_close(self):
        """Handle window close: refresh main GUI."""
        self.window.withdraw()
        if self._main_gui_instance is not None:
            try:
                self._main_gui_instance.update_plots()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public open method
    # ------------------------------------------------------------------

    def open(self, min_frq: float = 1.2, smooth_buffer: int = 3) -> None:
        """
        Open or raise the full-resolution detail Toplevel for this well.
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
            # Connect click events for manual peak manipulation
            self.detail_canvas.mpl_connect(
                "button_press_event", self._on_plot_click
            )

        # Use custom close handler
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.detail_ax.clear()
        self.plot(
            self.detail_ax,
            min_frq=self.min_freq_var.get(),
            smooth_buffer=self.smooth_var.get(),
            show_y_labels=True,
            show_manual_peaks_green=True,
        )
        self.detail_canvas.draw()
        self.window.deiconify()
        self.window.lift()