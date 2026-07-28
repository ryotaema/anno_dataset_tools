"""抽出・間引きタブ — 条件で絞り込み、枚数や割合の上限でサブセットを作る。"""

import tkinter as tk
from tkinter import ttk, messagebox

from .. import dataset as ds_mod
from .. import ops
from .common import PathRow, SummaryTable

# 絞り込みに使う区分（キー, 見出し, 表示幅）
FILTERS = [
    ('cam',     'カメラ',      14),
    ('date',    '取得日',      14),
    ('session', 'セッション',  24),
    ('class',   'クラス',      20),
    ('split',   'split',       12),
]


class SubsetTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.result = None

        # --- 絞り込み ---
        filt = ttk.LabelFrame(self, text='絞り込み（何も選ばなければすべて対象）')
        filt.pack(fill='x', padx=8, pady=(8, 4))

        self.lists = {}
        row = ttk.Frame(filt)
        row.pack(fill='x', padx=8, pady=8)
        for key, label, width in FILTERS:
            box = ttk.Frame(row)
            box.pack(side='left', fill='both', expand=True, padx=(0, 6))
            ttk.Label(box, text=label).pack(anchor='w')
            lb = tk.Listbox(box, selectmode='extended', height=5, width=width,
                            exportselection=False)
            lb.pack(fill='both', expand=True)
            lb.bind('<<ListboxSelect>>', lambda e: self._invalidate())
            self.lists[key] = lb
        ttk.Button(filt, text='選択をすべて解除', command=self.clear_filters).pack(
            anchor='w', padx=8, pady=(0, 8))

        # --- 上限 ---
        lim = ttk.LabelFrame(self, text='枚数・割合の上限')
        lim.pack(fill='x', padx=8, pady=4)

        r1 = ttk.Frame(lim)
        r1.pack(fill='x', padx=8, pady=(8, 2))
        self.limit_mode = tk.StringVar(value='none')
        for key, label in [('none', '制限しない'), ('count', '枚数で上限'),
                           ('ratio', '割合で上限')]:
            ttk.Radiobutton(r1, text=label, value=key, variable=self.limit_mode,
                            command=self._invalidate).pack(side='left', padx=(0, 12))
        self.limit_value = tk.IntVar(value=500)
        self.limit_value.trace_add('write', lambda *a: self._invalidate())
        ttk.Spinbox(r1, from_=1, to=1000000, textvariable=self.limit_value, width=8,
                    increment=50).pack(side='left')
        self.unit_var = tk.StringVar(value='枚')
        ttk.Label(r1, textvariable=self.unit_var).pack(side='left', padx=(4, 0))

        r2 = ttk.Frame(lim)
        r2.pack(fill='x', padx=8, pady=(2, 8))
        self.per_class = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text='クラスごとに上限を適用（クラス不均衡の是正）',
                        variable=self.per_class, command=self._invalidate).pack(side='left')
        ttk.Label(r2, text='  選び方:').pack(side='left', padx=(12, 0))
        self.strategy = tk.StringVar(value='spread')
        self.strategy_box = ttk.Combobox(
            r2, textvariable=self.strategy, state='readonly', width=52,
            values=['spread — セッション均等・セッション内は等間隔（時刻が散る）',
                    'spread_random — セッション均等・セッション内はランダム',
                    'random — 全体からランダムに選ぶ',
                    'head — 先頭から順に選ぶ'])
        self.strategy_box.current(0)
        self.strategy_box.pack(side='left', padx=(6, 0))
        self.strategy_box.bind('<<ComboboxSelected>>', lambda e: self._invalidate())

        # --- 出力 ---
        out = ttk.LabelFrame(self, text='出力')
        out.pack(fill='x', padx=8, pady=4)
        self.out_row = PathRow(out, '出力フォルダ:', on_change=self._invalidate)
        self.out_row.pack(fill='x', padx=8, pady=(8, 4))
        lay = ttk.Frame(out)
        lay.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(lay, text='形式:').pack(side='left')
        self.layout_var = tk.StringVar(value='flat')
        for key, label in [('flat', 'images/ labels/'),
                           ('yolo', 'train/images/（YOLO 標準）')]:
            ttk.Radiobutton(lay, text=label, value=key, variable=self.layout_var,
                            command=self._invalidate).pack(side='left', padx=(8, 0))
        self.move_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lay, text='移動', variable=self.move_var).pack(side='left', padx=(16, 0))

        # --- 実行ボタン ---
        # 先に下端を確保しておく（expand する「確認」枠より後に pack すると、
        # ウィンドウが小さいときに領域が足りず表示されなくなる）
        self.run_btn = ttk.Button(self, text='② 抽出して書き出す', command=self.run)
        self.run_btn.pack(side='bottom', anchor='w', padx=8, pady=(4, 8))
        self.run_btn.state(['disabled'])

        # --- 確認 ---
        prev = ttk.LabelFrame(self, text='確認')
        prev.pack(fill='both', expand=True, padx=8, pady=4)
        bar = ttk.Frame(prev)
        bar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Button(bar, text='① プレビューを作成', command=self.preview).pack(side='left')
        self.summary_var = tk.StringVar(value='データセットを追加してください')
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=(12, 0))

        self.table = SummaryTable(prev, first_col='抽出後のクラス内訳', height=5)
        self.table.pack(fill='both', expand=True, padx=8, pady=(0, 8))

    # ------------------------------------------------------------------

    def _invalidate(self, *_):
        self.result = None
        if hasattr(self, 'run_btn'):
            self.run_btn.state(['disabled'])
            self.table.clear()
        self.unit_var.set('%' if self.limit_mode.get() == 'ratio' else '枚')

    def clear_filters(self):
        for lb in self.lists.values():
            lb.selection_clear(0, 'end')
        self._invalidate()

    def refresh(self):
        items = self.state.items
        classes = self.state.classes
        for key, _, _ in FILTERS:
            lb = self.lists[key]
            selected = {lb.get(i) for i in lb.curselection()}
            lb.delete(0, 'end')
            for value in ds_mod.distinct_values(items, key, classes):
                lb.insert('end', value)
                if value in selected:
                    lb.selection_set('end')
        self._invalidate()
        self.summary_var.set('「プレビューを作成」で内容を確認してください' if items
                             else 'データセットを追加してください')

    def _filters(self):
        out = {}
        for key, _, _ in FILTERS:
            lb = self.lists[key]
            picked = {lb.get(i) for i in lb.curselection()}
            if picked:
                out[key] = picked
        return out

    # ------------------------------------------------------------------

    def preview(self):
        items = [i for i in self.state.items if i.has_image]
        if not items:
            messagebox.showinfo('データがありません', 'データセットを追加してください。')
            return
        if not self.out_row.get():
            messagebox.showinfo('出力先が未指定', '出力フォルダを指定してください。')
            return

        classes = self.state.classes
        filters = self._filters()
        strategy = self.strategy.get().split(' — ')[0]
        filtered = ops.filter_items(items, filters, classes)
        kept = ops.limit_items(
            filtered,
            mode=self.limit_mode.get(),
            value=self.limit_value.get(),
            per_class=self.per_class.get(),
            classes=classes,
            strategy=strategy,
        )
        self.result = kept

        # セッションが取れないデータでは spread はランダムに切り替わる
        fallback = (strategy.startswith('spread')
                    and self.limit_mode.get() != 'none'
                    and not ops.spreadable(filtered))
        if fallback:
            self.state.log.write(
                'セッション情報がないため、spread ではなくランダムに選びました'
                '（命名規則の対象外データです）')

        rows = ds_mod.summary_rows(kept, 'class', classes)
        self.table.show(rows, total=len(kept), total_label='合計（画像）')

        cond = '、'.join(f'{dict((k, l) for k, l, _ in FILTERS)[k]}={len(v)}件選択'
                        for k, v in filters.items()) or '条件なし'
        self.summary_var.set(
            f'全 {len(items):,} 件 → 絞り込み {len(filtered):,} 件 → 抽出 {len(kept):,} 件'
            f'（{cond}）'
            + ('　※セッション情報がないためランダム選択' if fallback else '')
            + ('　▶ 下の「② 抽出して書き出す」で出力します' if kept else ''))
        self.run_btn.state(['!disabled'] if kept else ['disabled'])

    def run(self):
        if not self.result:
            return
        out = self.out_row.get()
        move = self.move_var.get()
        if not messagebox.askyesno(
                '確認',
                f'{len(self.result):,} 件を {out} に'
                f'{"移動" if move else "コピー"}します。実行しますか？'):
            return

        split_map = ({'train': self.result} if self.layout_var.get() == 'yolo'
                     else {'': self.result})
        plan = ops.plan_output(split_map, out, layout=self.layout_var.get())

        self.run_btn.state(['disabled'])
        log, status = self.state.log, self.state.status
        log.write(f'抽出開始 → {out}（{len(self.result):,} 件）')
        stats = ops.write_plan(plan, classes=self.state.classes, move=move,
                               on_progress=lambda d, t: status.step(d, t, '書き出し中 '))
        log.write(f'  画像 {stats["images"]:,} / ラベル {stats["labels"]:,} を出力')
        for src, err in stats['errors'][:10]:
            log.write(f'  [エラー] {src}: {err}')
        status.reset()
        messagebox.showinfo('完了', f'{stats["images"]:,} 件を書き出しました。')
        self._invalidate()
