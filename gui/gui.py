import ttkbootstrap as ttk
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog


class GUI:
    def __init__(self, plate_rows=8, plate_columns=6):
        self.gui = ttk.Window(title="Analyze measurement", themename="darkly")

        self.wells = plate_rows * plate_columns
        self.visible_plots = 6
        self.start_idx = 0
        self.gui.rowconfigure(0, weight=1)

        self.gui.columnconfigure(0, weight=0)  # future buttons
        self.gui.columnconfigure(1, weight=4)  # plots
        self.gui.columnconfigure(2, weight=1)  # well buttons
        self.gui.columnconfigure(3, weight=0)  # scrollbar

        self.left_panel = ttk.Frame(self.gui)
        self.left_panel.grid(row=0, column=0, sticky="ns")

        ttk.Button(
            self.left_panel,
            text="Load CSV",
            command=self.load_file
        ).pack(fill="x", pady=5, padx=5)

        self.plot_frame = ttk.Frame(self.gui)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")

        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.button_container = ttk.Frame(self.gui)
        self.button_container.grid(row=0, column=2, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self.gui,
            orient="vertical",
            command=self.on_scrollbar
        )

        self.scrollbar.grid(row=0, column=3, sticky="ns")

        # mouse wheel
        self.gui.bind_all("<MouseWheel>", self._on_mousewheel)
        self.gui.bind_all("<Button-4>", self._on_mousewheel)
        self.gui.bind_all("<Button-5>", self._on_mousewheel)

        self.row_window = {}

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
        time = self.data[:, 0]
        values = self.data[:, 1:]

        return time, values

    def create_plots(self):
        time, values = self.get_data()

        self.fig, self.axes = plt.subplots(
            self.visible_plots,
            1,
            figsize=(6, 10)
        )

        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.04, hspace=0.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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

    def update_plots(self):
        for i, ax in enumerate(self.axes):
            ax.clear()

            data_idx = self.start_idx + i

            if data_idx >= self.wells:
                continue

            self.row_window[data_idx].plot(ax)

        self.canvas.draw_idle()

        self.update_buttons()
        self.update_scrollbar()

    def update_buttons(self):
        for w in self.button_container.winfo_children():
            w.destroy()

        for i in range(self.visible_plots):
            data_idx = self.start_idx + i

            if data_idx >= self.wells:
                continue

            btn = ttk.Button(
                self.button_container,
                text=self.row_window[data_idx].row_name,
                command=lambda idx=data_idx: self.open_row(idx)
            )

            btn.grid(row=i, column=0, sticky="nsew", padx=5, pady=3)

            self.button_container.rowconfigure(i, weight=1)

        self.button_container.columnconfigure(0, weight=1)

    def update_scrollbar(self):
        total = self.wells - self.visible_plots
        if total <= 0:
            self.scrollbar.set(0, 1)
            return

        start = self.start_idx / total
        end = (self.start_idx + self.visible_plots) / self.wells

        self.scrollbar.set(start, end)

    def on_scrollbar(self, *args):
        if args[0] == "moveto":
            fraction = float(args[1])
            max_start = self.wells - self.visible_plots

            self.start_idx = int(fraction * max_start)
        elif args[0] == "scroll":
            self.start_idx += int(args[1])

        self.start_idx = max(0, min(self.start_idx, self.wells - self.visible_plots))
        self.update_plots()

    def _on_mousewheel(self, event):
        if event.delta:
            direction = int(-event.delta / 120)
        else:
            direction = 1 if event.num == 5 else -1
        self.start_idx += direction
        self.start_idx = max(0, min(self.start_idx, self.wells - self.visible_plots))

        self.update_plots()

    def open_row(self, idx):

        self.row_window[idx].open(self.gui)


class RowWindow:
    def __init__(self, row_idx, row_name, time, values):
        self.row_idx = row_idx
        self.row_name = row_name
        self.time = time
        self.values = values

    def plot(self, ax):
        ax.plot(self.time, self.values)

        ax.set_title(self.row_name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pole distance in Pixel")

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