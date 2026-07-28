"""概要タブ — 読み込んだデータセットの内訳を枚数と割合で表示する。"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .. import dataset as ds_mod
from .common import SummaryTable

# 表示する区分（キー, ラベル）
VIEWS = [
    ('class',   'クラス'),
    ('cam',     'カメラ'),
    ('date',    '取得日'),
    ('session', 'セッション'),
    ('split',   'split'),
    ('source',  '取得元データセット'),
    ('mod',     'モダリティ'),
]


class OverviewTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state

        left = ttk.LabelFrame(self, text='区分')
        left.pack(side='left', fill='y', padx=(8, 4), pady=8)
        self.key_var = tk.StringVar(value='class')
        for key, label in VIEWS:
            ttk.Radiobutton(left, text=label, value=key, variable=self.key_var,
                            command=self.refresh).pack(anchor='w', padx=8, pady=2)
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=6)
        ttk.Button(left, text='CSVに保存', command=self.save_csv).pack(
            fill='x', padx=8, pady=(0, 8))

        right = ttk.Frame(self)
        right.pack(side='left', fill='both', expand=True, padx=(4, 8), pady=8)

        self.head_var = tk.StringVar(value='')
        ttk.Label(right, textvariable=self.head_var).pack(anchor='w', pady=(0, 4))

        self.table = SummaryTable(right, first_col='区分', height=16, extra_col='BBox数')
        self.table.pack(fill='both', expand=True)

        self.note_var = tk.StringVar(value='')
        ttk.Label(right, textvariable=self.note_var, foreground='#a60',
                  wraplength=700, justify='left').pack(anchor='w', pady=(6, 0))

    # ------------------------------------------------------------------

    def refresh(self):
        items = self.state.items
        self.table.clear()
        if not items:
            self.head_var.set('')
            self.note_var.set('')
            return

        key = self.key_var.get()
        classes = self.state.classes
        total = len(items)
        rows = ds_mod.summary_rows(items, key, classes, total)

        # クラス別のときは BBox 数（インスタンス数）も併記する
        if key == 'class':
            boxes = ds_mod.box_counts_by_class(items, classes)
            rows = [(v, n, r, boxes.get(v, 0)) for v, n, r in rows]
            total_boxes = sum(boxes.values())
        else:
            rows = [(v, n, r, '') for v, n, r in rows]
            total_boxes = ''

        self.table.show(rows, total=total, total_label='合計（画像）',
                        total_extra=total_boxes)

        label = dict(VIEWS)[key]
        self.head_var.set(f'{label}別の内訳 — {len(rows)} 区分 / 全 {total:,} 件')

        notes = []
        if key == 'class':
            notes.append('1枚に複数クラスが写っている場合、各クラスに1枚ずつ数えるため'
                         '割合の合計は100%を超えることがあります。')
        unknown = sum(1 for i in items if not i.info.matched)
        if unknown and key in ('cam', 'date', 'session', 'mod'):
            notes.append(f'{unknown:,} 件は命名規則の対象外のため「不明」に入っています'
                         '（旧命名や外部データセット）。')
        conflict = self.state.class_conflict()
        if conflict:
            notes.append('クラス定義の不一致 — ' + conflict)
        self.note_var.set('\n'.join(notes))

    def save_csv(self):
        items = self.state.items
        if not items:
            messagebox.showinfo('データがありません', 'データセットを追加してください。')
            return
        path = filedialog.asksaveasfilename(
            title='内訳を保存', defaultextension='.csv',
            initialfile='dataset_summary.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return
        ds_mod.write_summary_csv(path, items, self.state.classes,
                                 keys=[k for k, _ in VIEWS])
        self.state.log.write(f'内訳を保存しました: {path}')
