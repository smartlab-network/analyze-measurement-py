import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, TclError
from scipy.ndimage import minimum_filter1d


class GUI:
    def __init__(self, plate_rows=8, plate_columns=6):
        self.gui = ttk.Window(title="Analyze measurement", themename="darkly")
        self.gui.geometry("1920x1080")

        self.wells = plate_rows * plate_columns

        self.rows_var = ttk.IntVar(value=6)
        self.height_spacing_var = ttk.DoubleVar(value=1.4)
        self.min_freq_var = ttk.DoubleVar(value=1.2)
        self.smooth_var = ttk.IntVar(value=3)

        self.start_idx = 0

        self.mouse_inside_plot = False
        self.zoom_active = False

        #GRID
        self.gui.rowconfigure(0, weight=1)
        self.gui.columnconfigure(0, weight=0)
        self.gui.columnconfigure(1, weight=4)
        self.gui.columnconfigure(2, weight=1)
        self.gui.columnconfigure(3, weight=0)

        #LEFT PANEL
        self.left_panel = ttk.Frame(self.gui)
        self.left_panel.grid(row=0, column=0, sticky="ns")

        for i in range(9):
            self.left_panel.rowconfigure(i, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=1)

        ttk.Button(
            self.left_panel,
            text="Load CSV",
            command=self.load_file
        ).grid(row=0, column=0, columnspan=2, sticky="nsew", pady=6, padx=8)

        ttk.Label(self.left_panel, text="Rows:", font=('Helvetica', 16), anchor="c").grid(row=1, column=0, sticky="new")

        self.rows_entry = ttk.Entry(self.left_panel, textvariable=self.rows_var, justify="center")
        self.rows_entry.grid(row=1, column=0, sticky="ew", padx=8)
        self.rows_entry.bind("<Return>", self.on_rows_changed)

        ttk.Label(self.left_panel, text="Height spacing", font=('Helvetica', 16), anchor="c").grid(row=2, column=0, sticky="new", padx=8)

        self.height_spacing_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.height_spacing_var,
            justify="center"
        )
        self.height_spacing_entry.grid(row=2, column=0, sticky="ew", padx=8)
        self.height_spacing_entry.bind("<Return>", self.on_height_spacing)

        ttk.Label(self.left_panel, text="Min Freq in hz", font=('Helvetica', 16), anchor="c").grid(row=3, column=0, sticky="new", padx=8)

        self.distance_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.min_freq_var,
            justify="center"
        )
        self.distance_entry.grid(row=3, column=0, sticky="ew", padx=8)
        self.distance_entry.bind("<Return>", self.on_distance_entry)

        ttk.Label(self.left_panel, text="Smooth", font=('Helvetica', 16), anchor="c").grid(row=4, column=0,  sticky="new", padx=8)
        self.smooth_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.smooth_var,
            justify="center"
        )
        self.smooth_entry.grid(row=4, column=0, sticky="ew", padx=8)
        self.smooth_entry.bind("<Return>", self.on_smooth_changed)

        # PLOT frame(middle)
        self.plot_frame = ttk.Frame(self.gui)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        #BUTTON PANEL(right pannel)
        self.button_container = ttk.Frame(self.gui)
        self.button_container.grid(row=0, column=2, sticky="nsew")

        #SCROLLBAR
        self.scrollbar = ttk.Scrollbar(
            self.gui,
            orient="vertical",
            command=self.on_scrollbar
        )
        self.scrollbar.grid(row=0, column=3, sticky="ns")

        self.gui.bind_all("<MouseWheel>", self._on_mousewheel)

        self.row_window:dict[int, RowWindow] = {}

    #LOAD DATA
    def load_file(self):
        path = filedialog.askopenfilename(
            title="CSV auswählen",
            filetypes=[("CSV Dateien", "*.csv")]
        )
        if not path:
            return

        self.data = np.loadtxt(path, delimiter=",")
        self.create_plots()

    def get_data(self):
        return self.data[:, 0], self.data[:, 1:]

    #CREATE PLOTS
    def create_plots(self):
        time, values = self.get_data()

        self.fig, self.axes = plt.subplots(
            self.rows_var.get(),
            1,
            figsize=(6, 10)
        )

        self.fig.subplots_adjust(
            top=0.96,
            bottom=0.04,
            left=0.08,
            right=0.98,
            hspace=self.height_spacing_var.get()
        )

        if not isinstance(self.axes, np.ndarray):
            self.axes = [self.axes]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Matplotlib Events
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_leave)
        self.canvas.mpl_connect("scroll_event", self.on_zoom)

        # Row objects
        for row_idx in range(self.wells):
            row_letter = string.ascii_uppercase[row_idx // 6]
            col_number = (row_idx % 6) + 1
            name = f"{row_letter}{col_number}"

            self.row_window[row_idx] = RowWindow(
                row_idx,
                name,
                time,
                values[:, row_idx]
            )
        self.update_plots()

    #ZOOM
    def on_zoom(self, event):
        if event.inaxes is None:
            return

        ax = event.inaxes
        self.zoom_active = True

        base_scale = 1.2

        if event.button == "up":
            scale = 1 / base_scale
        elif event.button == "down":
            scale = base_scale
        else:
            return

        xlim = ax.get_xlim()
        xdata = event.xdata

        if xdata is None:
            return

        # current x lenght
        x_range = (xlim[1] - xlim[0]) * scale

        # relative cursor pos
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])

        #new Limits
        new_xlim = [
            xdata - x_range * relx,
            xdata + x_range * (1 - relx)
        ]

        ax.set_xlim(new_xlim)
        self.canvas.draw_idle()

    #HOVER
    def on_hover(self, event):
        self.mouse_inside_plot = True

        if self.zoom_active:
            return

        for ax in self.axes:
            ax.set_alpha(0.3)

        if event.inaxes:
            ax = event.inaxes
            ax.set_alpha(1.0)

        self.canvas.draw_idle()

    def on_leave(self, event):
        self.mouse_inside_plot = False

        for ax in self.axes:
            ax.set_alpha(1.0)

        self.canvas.draw_idle()

    #UPDATE
    def update_plots(self):
        for i, ax in enumerate(self.axes):
            ax.clear()

            data_idx = self.start_idx + i
            if data_idx >= self.wells:
                continue

            self.row_window[data_idx].plot(
                ax,
                min_frq=self.min_freq_var.get(),
                smooth_buffer=self.smooth_var.get()
            )

        self.canvas.draw_idle()
        self.update_buttons()
        self.update_scrollbar()

    #BUTTONS
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
                command=lambda idx=data_idx: self.open_row(idx)
            )

            btn.grid(row=i, column=0, sticky="nsew", padx=5, pady=3)

    #SCROLL
    def update_scrollbar(self):
        total = self.wells - self.rows_var.get()
        if total <= 0:
            self.scrollbar.set(0, 1)
            return

        start = self.start_idx / total
        end = (self.start_idx + self.rows_var.get()) / self.wells
        self.scrollbar.set(start, end)

    def on_scrollbar(self, *args):
        if args[0] == "moveto":
            fraction = float(args[1])
            max_start = self.wells - self.rows_var.get()
            self.start_idx = int(fraction * max_start)

        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def _on_mousewheel(self, event):
        if self.mouse_inside_plot:
            return

        direction = int(-event.delta / 120) if event.delta else 0
        self.start_idx += direction

        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def open_row(self, idx):
        self.row_window[idx].open(self.gui)

    def on_height_spacing(self, event):
        try:
            self.fig.subplots_adjust(hspace=self.height_spacing_var.get())
            self.canvas.draw_idle()
        except Exception:
            pass

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

#ROW WINDOW
class RowWindow:
    def __init__(self, row_idx: int, row_name: str, time: np.ndarray, values: np.ndarray):
        self.row_idx = row_idx
        self.row_name = row_name
        self.time = time
        self.values = values

    """
    running avg with buffer range. numpy methods are used to make it O(n), insted of O(n * buffer)
    """
    def smooth(self, values: np.ndarray, buffer: int):
        len_values = len(values)
        new_values = np.zeros(len_values)

        half_buffer = buffer // 2

        new_values[:half_buffer] = values[:half_buffer]
        new_values[len_values - half_buffer:] = values[len_values - half_buffer:]

        kernel_size = 2 * half_buffer + 1
        kernel = np.ones(kernel_size) / kernel_size

        # Convolution applies a sliding window over `values`, multiplies it element-wise
        # with the kernel, and sums the result. 'valid' avoids boundary issues. for every valid window.
        smoothed = np.convolve(values, kernel, mode='valid')

        new_values[half_buffer:len_values - half_buffer] = smoothed

        return new_values

    def open(self, master):
        fig, ax = plt.subplots()
        self.plot(ax)

        win = ttk.Toplevel(master)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

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

    def plot(self, ax, min_frq=1.2, smooth_buffer=3):
        smoothed = self.smooth(self.values, smooth_buffer)

        ax.plot(self.time, self.values, alpha=0.3)  # optional: raw
        ax.plot(self.time, smoothed, label="smoothed")

        pt, pv = self.find_peaks(min_frq=min_frq, values = smoothed)

        ax.plot(self.time[pt], pv, "ro", label="peaks")

        ax.set_title(self.row_name)
        ax.grid(True)
        ax.invert_yaxis()


GUI().gui.mainloop()

