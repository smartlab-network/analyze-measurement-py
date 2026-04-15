import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, TclError

class GUI:
    def __init__(self, plate_rows=8, plate_columns=6):
        self.gui = ttk.Window(title="Analyze measurement", themename="darkly")
        self.gui.geometry("1920x1080")

        self.wells = plate_rows * plate_columns
        self.rows_var = ttk.IntVar(value=6)
        self.height_spacing_var = ttk.DoubleVar(value = 1.4)
        self.start_idx = 0
        self.gui.rowconfigure(0, weight=1)

        self.gui.columnconfigure(0, weight=0)  # future buttons
        self.gui.columnconfigure(1, weight=4)  # plots
        self.gui.columnconfigure(2, weight=1)  # well buttons
        self.gui.columnconfigure(3, weight=0)  # scrollbar

        self.left_panel = ttk.Frame(self.gui)
        self.left_panel.grid(row=0, column=0, sticky="ns")

        self.left_panel.rowconfigure(0, weight=1)
        self.left_panel.rowconfigure(1, weight=1)
        self.left_panel.rowconfigure(2, weight=1)
        self.left_panel.rowconfigure(3, weight=1)
        self.left_panel.rowconfigure(4, weight=1)
        self.left_panel.rowconfigure(5, weight=1)
        self.left_panel.rowconfigure(6, weight=1)
        self.left_panel.rowconfigure(7, weight=1)
        self.left_panel.rowconfigure(8, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=1)

        ttk.Button(
            self.left_panel,
            text="Load CSV",
            command=self.load_file
        ).grid(row = 0, column=0, sticky="nsew", columnspan=2, pady=8)


        ttk.Label(self.left_panel,
                  text="Rows:",
                  font=('Helvetica', 16),
                  anchor="c"
        ).grid(row=1, column=0, sticky="new")

        self.rows_entry =   ttk.Entry(
                            self.left_panel,
                            textvariable=self.rows_var,
                            justify="center"
        )
        self.rows_entry.grid(row=1, column=0, sticky="ew")
        self.rows_entry.bind("<Return>", self.on_rows_changed)

        ttk.Label(self.left_panel,
                  text = "Height spacing",
                  font=('Helvetica', 16),
                  anchor="c"
        ).grid(row=2, column=0, sticky="new")

        self.height_spacing_entry = ttk.Entry(
            self.left_panel,
            textvariable=self.height_spacing_var,
            justify="center"
        )
        self.height_spacing_entry.grid(row=2, column=0, sticky="ew")
        self.height_spacing_entry.bind("<Return>", self.on_height_spacing)

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

    def on_rows_changed(self, event):
        try:
            value = self.rows_var.get()
            self.create_plots()
        except AttributeError:
            pass
        except TclError:
            Messagebox.show_error(
                title="Ungültige Eingabe",
                message="Bitte eine ganze Zahl eingeben.",
                parent=self.gui
            )

    def on_height_spacing(self, event):
        try:
            self.fig.subplots_adjust(hspace=self.height_spacing_var.get())
            self.canvas.draw_idle()
        except AttributeError:
            value = self.height_spacing_var.get()

        except TclError:
            Messagebox.show_error(
                title="Ungültige Eingabe",
                message="Bitte komma Zahl eingeben.",
                parent=self.gui
            )

    def create_plots(self):
        time, values = self.get_data()

        self.fig, self.axes = plt.subplots(
            self.rows_var.get(),
            1,
            figsize=(6, 10)
        )

        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.04, hspace=self.height_spacing_var.get())

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

        for i in range(self.wells):
            self.button_container.rowconfigure(i, weight=0, minsize=0)

        for i in range(self.rows_var.get()):
            data_idx = self.start_idx + i

            if data_idx >= self.wells:
                continue

            btn = ttk.Button(
                self.button_container,
                text=self.row_window[data_idx].row_name,
                command=lambda idx=data_idx: self.open_row(idx)
            )

            btn.grid(row=i, column=0, sticky="nsew", padx=5, pady=3)

            self.button_container.rowconfigure(i, weight=1)  # nur aktive Zeilen

        self.button_container.columnconfigure(0, weight=1)

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
        elif args[0] == "scroll":
            self.start_idx += int(args[1])

        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))
        self.update_plots()

    def _on_mousewheel(self, event):
        if event.delta:
            direction = int(-event.delta / 120)
        else:
            direction = 1 if event.num == 5 else -1
        self.start_idx += direction
        self.start_idx = max(0, min(self.start_idx, self.wells - self.rows_var.get()))

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