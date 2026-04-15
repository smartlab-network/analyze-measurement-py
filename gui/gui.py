import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, TclError
from scipy.signal import find_peaks


class GUI:
    def __init__(self, plate_rows=8, plate_columns=6):
        self.gui = ttk.Window(title="Analyze measurement", themename="darkly")
        self.gui.geometry("1920x1080")

        self.wells = plate_rows * plate_columns

        self.rows_var = ttk.IntVar(value=6)
        self.height_spacing_var = ttk.DoubleVar(value=1.4)
        self.min_distance_var = ttk.IntVar(value=10)

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
        ).grid(row=0, column=0, columnspan=2, sticky="nsew", pady=8)

        ttk.Label(self.left_panel, text="Rows:", font=('Helvetica', 16)).grid(row=1, column=0)
        self.rows_entry = ttk.Entry(self.left_panel, textvariable=self.rows_var, justify="center")
        self.rows_entry.grid(row=1, column=1, sticky="ew")
        self.rows_entry.bind("<Return>", self.on_rows_changed)

        ttk.Label(self.left_panel, text="Height spacing", font=('Helvetica', 16)).grid(row=2, column=0)
        self.height_spacing_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.height_spacing_var,
            justify="center"
        )
        self.height_spacing_entry.grid(row=2, column=1, sticky="ew")
        self.height_spacing_entry.bind("<Return>", self.on_height_spacing)

        ttk.Label(self.left_panel, text="Min Peak Distance", font=('Helvetica', 16)).grid(row=3, column=0)
        self.distance_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.min_distance_var,
            justify="center"
        )
        self.distance_entry.grid(row=3, column=1, sticky="ew")
        self.distance_entry.bind("<Return>", self.on_distance_entry)

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

        self.row_window = {}

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
                min_distance=self.min_distance_var.get()
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
        except Exception:
            pass

    def on_distance_entry(self, event):
        self.update_plots()

#ROW WINDOW
class RowWindow:
    def __init__(self, row_idx, row_name, time, values):
        self.row_idx = row_idx
        self.row_name = row_name
        self.time = time
        self.values = values

    def smooth(self, v, w=7):
        return np.convolve(v, np.ones(w)/w, mode="same")

    def find_peaks(self, min_dist):
        v = self.smooth(self.values)
        inv = -v

        dt = np.mean(np.diff(self.time)) if len(self.time) > 1 else 1

        min_dist_sec = min_dist / 1000.0

        dist = max(1, int(min_dist_sec / dt))

        peaks, _ = find_peaks(
            inv,
            distance=dist,
            prominence=np.std(v) * 0.3
        )

        return self.time[peaks], self.values[peaks]

    def plot(self, ax, min_distance=10):
        ax.plot(self.time, self.values)

        pt, pv = self.find_peaks(min_distance)
        ax.plot(pt, pv, "ro")

        ax.set_title(self.row_name)
        ax.grid(True)
        ax.invert_yaxis()

    def open(self, master):
        fig, ax = plt.subplots()
        self.plot(ax)

        win = ttk.Toplevel(master)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()


GUI().gui.mainloop()

