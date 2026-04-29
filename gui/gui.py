import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from scipy.ndimage import minimum_filter1d


class GUI:
    def __init__(self, plate_rows=8, plate_columns=6):
        self.gui = ttk.Window(title="Analyze measurement", themename="superhero")
        self.gui.geometry("1920x1080")

        self.wells = plate_rows * plate_columns

        self.rows_var = ttk.IntVar(value=48)
        self.height_spacing_var = ttk.DoubleVar(value=1.4)
        self.min_freq_var = ttk.DoubleVar(value=1.2)
        self.smooth_var = ttk.IntVar(value=3)

        self.start_idx = 0
        self.mouse_inside_plot = False
        self.zoom_active = False

        self.gui.rowconfigure(0, weight=0, minsize=60)
        self.gui.rowconfigure(1, weight=1)
        self.gui.columnconfigure(0, weight=1)
        self.gui.columnconfigure(1, weight=10)
        self.gui.columnconfigure(2, weight=1)
        self.gui.columnconfigure(3, weight=0)

        #HEADER FRAME
        self.header_frame = ttk.Frame(self.gui)
        self.header_frame.grid(row = 0, column=0, columnspan=4, sticky="nsew", pady=20)

        self.header_frame.columnconfigure(0, weight=0)
        self.header_frame.columnconfigure(1, weight=1)
        self.header_frame.rowconfigure(0, weight=2)
        # LEFT PANEL
        self.left_panel = ttk.Frame(self.gui)
        self.left_panel.grid(row=1, column=0, sticky="ns")

        for i in range(9):
            self.left_panel.rowconfigure(i, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=1)

        ttk.Button(self.header_frame, text="CSV", command=self.load_file).grid(
            row=0, column=0, pady=5, padx=5
        )

        ttk.Label(self.left_panel, text="Rows:", font=("Helvetica", 16), anchor="c").grid(
            row=1, column=0, sticky="new"
        )
        self.rows_entry = ttk.Entry(self.left_panel, textvariable=self.rows_var, justify="center")
        self.rows_entry.grid(row=1, column=0, sticky="ew", padx=8)
        self.rows_entry.bind("<Return>", self.on_rows_changed)

        ttk.Label(self.left_panel, text="Min Frq in hz", font=("Helvetica", 16), anchor="c").grid(
            row=3, column=0, sticky="new", padx=8
        )
        self.distance_entry = ttk.Entry(self.left_panel, textvariable=self.min_freq_var, justify="center")
        self.distance_entry.grid(row=3, column=0, sticky="ew", padx=8)
        self.distance_entry.bind("<Return>", self.on_distance_entry)

        ttk.Label(self.left_panel, text="Smooth", font=("Helvetica", 16), anchor="c").grid(
            row=4, column=0, sticky="new", padx=8
        )
        self.smooth_entry = ttk.Entry(self.left_panel, textvariable=self.smooth_var, justify="center")
        self.smooth_entry.grid(row=4, column=0, sticky="ew", padx=8)
        self.smooth_entry.bind("<Return>", self.on_smooth_changed)

        # PLOT FRAME
        self.plot_frame = ttk.Frame(self.gui)
        self.plot_frame.grid(row=1, column=1, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        # RIGHT BUTTON PANEL
        self.button_container = ttk.Frame(self.gui)
        self.button_container.grid(row=1, column=2, sticky="nsew", pady=5, ipady=0)

        # SCROLLBAR
        self.scrollbar = ttk.Scrollbar(self.gui, orient="vertical", command=self.on_scrollbar)
        self.scrollbar.grid(row=0, column=3, sticky="ns")

        self.gui.bind_all("<MouseWheel>", self._on_mousewheel)

        self.row_window: dict[int, RowWindow] = {}

    def load_file(self):
        path = filedialog.askopenfilename(title="CSV auswählen", filetypes=[("CSV Dateien", "*.csv")])
        if not path:
            return
        self.data = np.loadtxt(path, delimiter=",")
        self.create_plots()

    def get_data(self):
        return self.data[:, 0], self.data[:, 1:]

    def create_plots(self):
        time, values = self.get_data()

        self.fig, self.axes = plt.subplots(self.rows_var.get(), 1, figsize=(6, 10))
        self.fig.subplots_adjust(top=0.98, bottom=0.02, left=0.08, right=0.98, hspace=0)

        if not isinstance(self.axes, np.ndarray):
            self.axes = [self.axes]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_leave)
        self.canvas.mpl_connect("scroll_event", self.on_zoom)

        for row_idx in range(self.wells):
            row_letter = string.ascii_uppercase[row_idx // 6]
            col_number = (row_idx % 6) + 1
            name = f"{row_letter}{col_number}"
            self.row_window[row_idx] = RowWindow(row_idx, name, time, values[:, row_idx], gui=self.gui)

        self.update_plots()

    def on_zoom(self, event):
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

    def on_hover(self, event):
        self.mouse_inside_plot = True
        if self.zoom_active:
            return
        for ax in self.axes:
            ax.set_alpha(0.3)
        if event.inaxes:
            event.inaxes.set_alpha(1.0)
        self.canvas.draw_idle()

    def on_leave(self, event):
        self.mouse_inside_plot = False
        for ax in self.axes:
            ax.set_alpha(1.0)
        self.canvas.draw_idle()

    def update_plots(self):
        for i, ax in enumerate(self.axes):
            ax.clear()
            data_idx = self.start_idx + i
            if data_idx >= self.wells:
                continue
            self.row_window[data_idx].plot(
                ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get(),
            )

        self.fig.supxlabel("Time in s", fontsize=10)

        self.canvas.draw_idle()
        self.update_buttons()
        self.update_scrollbar()

    def update_buttons(self):
        for w in self.button_container.winfo_children():
            w.destroy()
        rows = self.rows_var.get()
        for i in range(rows):
            self.button_container.rowconfigure(i, weight=1)
        self.button_container.columnconfigure(0, weight=1)
        for i in range(rows):
            data_idx = self.start_idx + i
            if data_idx >= self.wells:
                continue
            btn = ttk.Button(
                self.button_container,
                text=self.row_window[data_idx].row_name,
                command=lambda idx=data_idx: self.open_row(idx),
                bootstyle="outline",
            )
            btn.grid(row=i, column=0, sticky="nsew", padx=5, pady=0)

    def update_scrollbar(self):
        total = self.wells - self.rows_var.get()
        if total <= 0:
            self.scrollbar.set(0, 1)
            return
        start = self.start_idx / total
        end = (self.start_idx + self.rows_var.get()) / self.wells
        self.scrollbar.set(start, end)

    def _on_mousewheel(self, event):
        if not hasattr(self, 'axes'):
            return

        # Fokus liegt auf einem Toplevel-Fenster → Haupt-Scroll ignorieren
        focused = self.gui.focus_get()
        if focused is not None and str(focused.winfo_toplevel()) != str(self.gui):
            return

        if self.mouse_inside_plot:
            return

        direction = int(-event.delta / 120) if event.delta else 0
        self.start_idx += direction
        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def on_scrollbar(self, *args):
        if not hasattr(self, 'axes'):
            return
        if args[0] == "moveto":
            fraction = float(args[1])
            max_start = self.wells - self.rows_var.get()
            self.start_idx = int(fraction * max_start)
        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def open_row(self, idx):
        self.row_window[idx].open(
            min_frq=self.min_freq_var.get(),
            smooth_buffer=self.smooth_var.get(),
        )

    def on_rows_changed(self, event):
        try:
            self.create_plots()
        except AttributeError:
            pass

    def on_distance_entry(self, event):
        try:
            self.update_plots()
        except AttributeError:
            pass

    def on_smooth_changed(self, event=None):
        try:
            self.update_plots()
        except AttributeError:
            pass


class RowWindow:
    def __init__(self, row_idx: int, row_name: str, time: np.ndarray, values: np.ndarray, gui: ttk.Window):
        self.row_idx = row_idx
        self.row_name = row_name
        self.time = time
        self.values = values

        self.window = ttk.Toplevel(self.row_name, master=gui)
        self.window.geometry("1920x720")
        self.window.withdraw()

        # Canvas/fig für das Toplevel – einmalig erstellt, nicht bei jedem open()
        self.detail_fig = None
        self.detail_canvas = None
        self.detail_ax = None

    def smooth(self, values: np.ndarray, buffer: int):
        if buffer < 2:
            return values.copy()
        # Ungerade erzwingen
        if buffer % 2 == 0:
            buffer += 1
        half_buffer = buffer // 2
        kernel = np.ones(buffer) / buffer
        # 'valid' vermeidet Zero-Padding an den Rändern
        smoothed = np.convolve(values, kernel, mode='valid')
        # Ränder mit Originalwerten auffüllen
        result = values.copy()
        result[half_buffer: len(values) - half_buffer] = smoothed
        return result

    def find_peaks(self, min_frq, values=None):
        if values is None:
            values = self.values
        buffer = int(60 / min_frq)
        half_buffer = buffer // 2
        kernel_size = 2 * half_buffer + 1
        window_min = minimum_filter1d(values, size=kernel_size)
        mask = values == window_min
        peaks_idx = np.where(mask)[0]
        return peaks_idx, values[peaks_idx]

    def plot(self, ax, min_frq=1.2, smooth_buffer=3, show_y_labels=False):
        smoothed = self.smooth(self.values, smooth_buffer)
        ax.plot(self.time, self.values, alpha=0.3)
        ax.plot(self.time, smoothed, label="smoothed")
        pt, pv = self.find_peaks(min_frq=min_frq, values=smoothed)
        ax.plot(self.time[pt], pv, "ro", label="peaks",markersize=4)
        ax.grid(True)
        ax.invert_yaxis()
        if not show_y_labels:
            ax.tick_params(left=False, labelleft=False)
        else:
            ax.set_xlabel("Time in s")
            ax.set_ylabel("Distance in pixel")

    def _make_zoom_handler(self, canvas):
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

    def _make_tk_zoom_blocker(self, canvas):

        def block(event):
            # matplotlib bekommt den Event über mpl_connect — tkinter-Propagation stoppen
            return "break"

        return block

    def open(self, min_frq=1.2, smooth_buffer=3):
        # Guard: Fenster sichtbar → nur nach vorne
        if self.window.winfo_ismapped():
            self.window.lift()
            return

        if self.detail_fig is None:
            self.detail_fig, self.detail_ax = plt.subplots(figsize=(16, 4))
            self.detail_fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)

            self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=self.window)
            widget = self.detail_canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)

            self.detail_canvas.mpl_connect(
                "scroll_event", self._make_zoom_handler(self.detail_canvas)
            )

        #on close
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        self.detail_ax.clear()
        self.plot(self.detail_ax, min_frq=min_frq, smooth_buffer=smooth_buffer, show_y_labels=True)
        self.detail_canvas.draw()

        self.window.deiconify()
        self.window.lift()


gui = GUI()
root = gui.gui
root.mainloop()