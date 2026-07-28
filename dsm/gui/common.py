"""GUI の共通部品。"""

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

BAR_WIDTH = 18


def bar_text(ratio, width=BAR_WIDTH):
    """割合を横棒で表す（表の中で構成比を一目で見るため）。"""
    filled = int(round(ratio / 100.0 * width))
    return '█' * filled + '·' * (width - filled)


class PathRow(ttk.Frame):
    """ラベル＋入力欄＋参照ボタンの1行。"""

    def __init__(self, master, label, mode='dir', width=52, on_change=None):
        super().__init__(master)
        self.mode = mode
        ttk.Label(self, text=label).pack(side='left')
        self.var = tk.StringVar()
        if on_change:
            self.var.trace_add('write', lambda *a: on_change())
        ttk.Entry(self, textvariable=self.var, width=width).pack(
            side='left', fill='x', expand=True, padx=(6, 4))
        ttk.Button(self, text='参照...', command=self._browse).pack(side='left')

    def _browse(self):
        if self.mode == 'dir':
            p = filedialog.askdirectory(title='フォルダを選択')
        elif self.mode == 'savezip':
            p = filedialog.asksaveasfilename(title='保存先', defaultextension='.zip',
                                             filetypes=[('ZIP', '*.zip')])
        else:
            p = filedialog.askopenfilename(title='ファイルを選択')
        if p:
            self.var.set(p)

    def get(self):
        return self.var.get().strip()

    def set(self, value):
        self.var.set(str(value))


class SummaryTable(ttk.Frame):
    """(値, 枚数, 割合) を棒付きで表示する表。extra_col で列を1つ足せる。"""

    def __init__(self, master, first_col='区分', height=10, extra_col=None):
        super().__init__(master)
        self.has_extra = bool(extra_col)
        cols = ('count', 'ratio', 'bar') + (('extra',) if extra_col else ())
        self.tree = ttk.Treeview(self, columns=cols, height=height)
        self.tree.heading('#0', text=first_col)
        self.tree.heading('count', text='枚数')
        self.tree.heading('ratio', text='割合')
        self.tree.heading('bar', text='')
        self.tree.column('#0', width=260, stretch=True)
        self.tree.column('count', width=80, anchor='e', stretch=False)
        self.tree.column('ratio', width=70, anchor='e', stretch=False)
        self.tree.column('bar', width=170, stretch=False)
        if extra_col:
            self.tree.heading('extra', text=extra_col)
            self.tree.column('extra', width=90, anchor='e', stretch=False)
        vsb = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='left', fill='y')
        self.tree.tag_configure('total', font=('TkDefaultFont', 10, 'bold'))

    def show(self, rows, total=None, total_label='合計', total_extra=''):
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            value, n, ratio = row[0], row[1], row[2]
            values = [f'{n:,}', f'{ratio:.1f}%', bar_text(ratio)]
            if self.has_extra:
                extra = row[3] if len(row) > 3 else ''
                values.append(f'{extra:,}' if isinstance(extra, int) else extra)
            self.tree.insert('', 'end', text=str(value), values=values)
        if total is not None:
            values = [f'{total:,}', '', '']
            if self.has_extra:
                values.append(f'{total_extra:,}' if isinstance(total_extra, int)
                              else total_extra)
            self.tree.insert('', 'end', text=total_label, tags=('total',), values=values)

    def clear(self):
        self.tree.delete(*self.tree.get_children())


class LogPane(ttk.LabelFrame):
    def __init__(self, master, height=7):
        super().__init__(master, text='ログ')
        self.text = ScrolledText(self, height=height, state='disabled')
        self.text.pack(fill='both', expand=True, padx=6, pady=6)

    def write(self, msg):
        self.text.configure(state='normal')
        self.text.insert('end', str(msg) + '\n')
        self.text.see('end')
        self.text.configure(state='disabled')
        self.update_idletasks()

    def clear(self):
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.configure(state='disabled')


class StatusBar(ttk.Frame):
    """下部の進捗バーとメッセージ。長い処理の途中経過はここに出す。"""

    def __init__(self, master):
        super().__init__(master)
        self.var = tk.StringVar(value='')
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True)
        ttk.Label(self, textvariable=self.var, width=34, anchor='w').pack(
            side='left', padx=(8, 0))

    def step(self, done, total, prefix=''):
        self.progress.configure(maximum=max(total, 1), value=done)
        self.var.set(f'{prefix}{done:,}/{total:,}')
        self.update()

    def message(self, text):
        self.var.set(text)
        self.update_idletasks()

    def reset(self):
        self.progress.configure(value=0)
        self.var.set('')


def labeled_spin(master, text, from_, to, default, width=8, increment=1, on_change=None):
    """ラベル付きのスピンボックス。(frame, var) を返す。"""
    frame = ttk.Frame(master)
    ttk.Label(frame, text=text).pack(side='left')
    var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
    if on_change:
        var.trace_add('write', lambda *a: on_change())
    ttk.Spinbox(frame, from_=from_, to=to, textvariable=var, width=width,
                increment=increment).pack(side='left', padx=(4, 0))
    return frame, var
