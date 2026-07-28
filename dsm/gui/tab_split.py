"""分割タブ — train / val / test に分ける。割合でも枚数でも指定できる。"""

import tkinter as tk
from collections import Counter
from tkinter import ttk, messagebox

from .. import ops
from ..dataset import class_name
from .common import PathRow


class SplitTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.split_map = None
        self.plan = None

        # --- 指定方法 ---
        opt = ttk.LabelFrame(self, text='分け方')
        opt.pack(fill='x', padx=8, pady=(8, 4))

        mode_row = ttk.Frame(opt)
        mode_row.pack(fill='x', padx=8, pady=(8, 2))
        self.mode_var = tk.StringVar(value='ratio')
        ttk.Radiobutton(mode_row, text='割合で指定', value='ratio', variable=self.mode_var,
                        command=self._switch_mode).pack(side='left')
        ttk.Radiobutton(mode_row, text='枚数で指定', value='count', variable=self.mode_var,
                        command=self._switch_mode).pack(side='left', padx=(12, 0))

        # 割合／枚数の入力欄はこの中で差し替える
        holder = ttk.Frame(opt)
        holder.pack(fill='x', padx=8, pady=2)

        # 割合
        self.ratio_frame = ttk.Frame(holder)
        self.ratio_vars = {}
        for name, default in [('train', 80), ('val', 20), ('test', 0)]:
            ttk.Label(self.ratio_frame, text=f'{name}:').pack(side='left', padx=(0, 2))
            var = tk.IntVar(value=default)
            var.trace_add('write', lambda *a: self._update_hint())
            ttk.Spinbox(self.ratio_frame, from_=0, to=100, textvariable=var, width=5,
                        increment=5).pack(side='left', padx=(0, 2))
            ttk.Label(self.ratio_frame, text='%').pack(side='left', padx=(0, 12))
            self.ratio_vars[name] = var

        # 枚数
        self.count_frame = ttk.Frame(holder)
        self.count_vars = {}
        for name, default in [('train', 800), ('val', 200), ('test', 0)]:
            ttk.Label(self.count_frame, text=f'{name}:').pack(side='left', padx=(0, 2))
            var = tk.IntVar(value=default)
            var.trace_add('write', lambda *a: self._update_hint())
            ttk.Spinbox(self.count_frame, from_=0, to=1000000, textvariable=var,
                        width=8, increment=50).pack(side='left', padx=(0, 2))
            ttk.Label(self.count_frame, text='枚').pack(side='left', padx=(0, 12))
            self.count_vars[name] = var

        self.hint_var = tk.StringVar(value='')
        ttk.Label(opt, textvariable=self.hint_var, foreground='#a60').pack(
            anchor='w', padx=8, pady=(2, 4))

        # --- まとまりの単位・その他 ---
        grp = ttk.Frame(opt)
        grp.pack(fill='x', padx=8, pady=2)
        ttk.Label(grp, text='まとまりの単位:').pack(side='left')
        self.group_var = tk.StringVar(value='none')
        self.group_box = ttk.Combobox(grp, textvariable=self.group_var, state='readonly',
                                      width=52, values=[f'{k} — {v}' for k, v
                                                        in ops.GROUP_MODES.items()])
        self.group_box.current(0)
        self.group_box.pack(side='left', padx=(6, 0))
        self.group_box.bind('<<ComboboxSelected>>', lambda e: self._invalidate())

        etc = ttk.Frame(opt)
        etc.pack(fill='x', padx=8, pady=(2, 8))
        self.strat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(etc, text='クラス比を保つ（stratified）', variable=self.strat_var,
                        command=self._invalidate).pack(side='left')
        ttk.Label(etc, text='  seed:').pack(side='left', padx=(12, 0))
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(etc, from_=0, to=99999, textvariable=self.seed_var, width=7
                    ).pack(side='left', padx=(4, 0))

        # --- 出力 ---
        out = ttk.LabelFrame(self, text='出力')
        out.pack(fill='x', padx=8, pady=4)
        self.out_row = PathRow(out, '出力フォルダ:', on_change=self._invalidate)
        self.out_row.pack(fill='x', padx=8, pady=(8, 4))
        lay = ttk.Frame(out)
        lay.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(lay, text='形式:').pack(side='left')
        self.layout_var = tk.StringVar(value='yolo')
        for key, label in [('yolo', 'train/images/（YOLO 標準・data.yaml 付き）'),
                           ('split', 'images/train/（split_dataset.py と同じ）')]:
            ttk.Radiobutton(lay, text=label, value=key, variable=self.layout_var,
                            command=self._invalidate).pack(side='left', padx=(8, 0))
        self.move_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lay, text='移動', variable=self.move_var).pack(side='left', padx=(16, 0))

        # --- 実行ボタン ---
        # 先に下端を確保しておく（expand する「確認」枠より後に pack すると、
        # ウィンドウが小さいときに領域が足りず表示されなくなる）
        self.run_btn = ttk.Button(self, text='② 分割を実行', command=self.run)
        self.run_btn.pack(side='bottom', anchor='w', padx=8, pady=(4, 8))
        self.run_btn.state(['disabled'])

        # --- プレビュー ---
        prev = ttk.LabelFrame(self, text='確認')
        prev.pack(fill='both', expand=True, padx=8, pady=4)
        bar = ttk.Frame(prev)
        bar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Button(bar, text='① プレビューを作成', command=self.preview).pack(side='left')
        self.summary_var = tk.StringVar(value='データセットを追加してください')
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=(12, 0))

        self.tree = ttk.Treeview(prev, columns=('count', 'ratio', 'bar'), height=6)
        self.tree.heading('#0', text='split / クラス')
        self.tree.heading('count', text='枚数')
        self.tree.heading('ratio', text='割合')
        self.tree.heading('bar', text='')
        self.tree.column('#0', width=280, stretch=True)
        self.tree.column('count', width=90, anchor='e', stretch=False)
        self.tree.column('ratio', width=70, anchor='e', stretch=False)
        self.tree.column('bar', width=180, stretch=False)
        self.tree.tag_configure('split', font=('TkDefaultFont', 10, 'bold'))
        self.tree.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self._switch_mode()

    # ------------------------------------------------------------------

    def _switch_mode(self):
        self.ratio_frame.pack_forget()
        self.count_frame.pack_forget()
        target = self.ratio_frame if self.mode_var.get() == 'ratio' else self.count_frame
        target.pack(fill='x')
        self._update_hint()
        self._invalidate()

    def _values(self):
        if self.mode_var.get() == 'ratio':
            return [self.ratio_vars[k].get() for k in ('train', 'val', 'test')]
        return [self.count_vars[k].get() for k in ('train', 'val', 'test')]

    def _update_hint(self):
        try:
            vals = self._values()
        except tk.TclError:
            return
        total_items = len(self.state.items)
        if self.mode_var.get() == 'ratio':
            s = sum(vals)
            self.hint_var.set('' if s == 100 else f'合計が {s}% です（100% にしてください）')
        else:
            s = sum(vals)
            if s > total_items:
                self.hint_var.set(f'合計 {s:,} 枚 > 全 {total_items:,} 枚。'
                                  'そのままだと足りない分は減らされます')
            elif s < total_items:
                self.hint_var.set(f'合計 {s:,} 枚（全 {total_items:,} 枚のうち '
                                  f'{total_items - s:,} 枚は使いません）')
            else:
                self.hint_var.set('')
        self._invalidate()

    def _invalidate(self, *_):
        self.split_map = None
        self.plan = None
        if hasattr(self, 'run_btn'):
            self.run_btn.state(['disabled'])
            self.tree.delete(*self.tree.get_children())

    def refresh(self):
        self._invalidate()
        n = len(self.state.items)
        self.summary_var.set('「プレビューを作成」で内容を確認してください' if n
                             else 'データセットを追加してください')
        if n and not self.out_row.get() and self.state.datasets:
            root = self.state.datasets[0].root
            self.out_row.set(root.parent / f'{root.name}_split')
        self._update_hint()

    # ------------------------------------------------------------------

    def preview(self):
        items = [i for i in self.state.items if i.has_image]
        if not items:
            messagebox.showinfo('データがありません', 'データセットを追加してください。')
            return
        if not self.out_row.get():
            messagebox.showinfo('出力先が未指定', '出力フォルダを指定してください。')
            return

        vals = self._values()
        mode = self.mode_var.get()
        if mode == 'ratio' and sum(vals) != 100:
            messagebox.showwarning('割合が不正', f'合計が {sum(vals)}% です。100% にしてください。')
            return
        if mode == 'count' and sum(vals) == 0:
            messagebox.showwarning('枚数が未指定', '枚数を指定してください。')
            return

        group = self.group_var.get().split(' — ')[0]
        self.split_map = ops.split_items(
            items,
            mode=mode,
            ratios=tuple(v / 100 for v in vals),
            counts=tuple(vals),
            stratified=self.strat_var.get(),
            group=group,
            seed=self.seed_var.get(),
        )
        self.split_map = {k: v for k, v in self.split_map.items() if v}
        self.plan = ops.plan_output(self.split_map, self.out_row.get(),
                                    layout=self.layout_var.get())

        classes = self.state.classes
        self.tree.delete(*self.tree.get_children())
        total = sum(len(v) for v in self.split_map.values())
        for name in ('train', 'val', 'test'):
            group_items = self.split_map.get(name)
            if not group_items:
                continue
            n = len(group_items)
            ratio = n / total * 100 if total else 0
            node = self.tree.insert('', 'end', text=name, tags=('split',),
                                    values=(f'{n:,}', f'{ratio:.1f}%',
                                            '█' * int(ratio / 100 * 18)))
            c = Counter()
            for item in group_items:
                for cid in (item.class_ids or (None,)):
                    c[cid] += 1
            for cid, cn in c.most_common():
                label = '(ラベルなし)' if cid is None else class_name(classes, cid)
                r = cn / n * 100 if n else 0
                self.tree.insert(node, 'end', text='    ' + label,
                                 values=(f'{cn:,}', f'{r:.1f}%', ''))
            self.tree.item(node, open=True)

        unused = len(items) - total
        msg = f'{total:,} 件を分割します'
        if unused > 0:
            msg += f'（{unused:,} 件は使いません）'
        if self.plan.renamed:
            msg += f'（名前重複 {self.plan.renamed} 件をリネーム）'
        # グループ単位だと途中で切れないので、指定値ちょうどにはならない
        if group != 'none':
            actual = [len(self.split_map.get(n, [])) for n in ('train', 'val', 'test')]
            want = vals if mode == 'count' else [round(len(items) * v / 100) for v in vals]
            if any(abs(a - w) > 0 for a, w in zip(actual, want)):
                msg += f'  ※{group}単位でまとめているため指定値ちょうどにはなりません'
        if total:
            msg += '　▶ 下の「② 分割を実行」で出力します'
        self.summary_var.set(msg)
        self.run_btn.state(['!disabled'] if total else ['disabled'])

    def run(self):
        if not self.plan:
            return
        move = self.move_var.get()
        counts = ', '.join(f'{k} {len(v):,}' for k, v in self.split_map.items())
        if not messagebox.askyesno(
                '確認',
                f'{counts}\n\n{self.plan.out_dir} に'
                f'{"移動" if move else "コピー"}します。実行しますか？'):
            return

        self.run_btn.state(['disabled'])
        log, status = self.state.log, self.state.status
        log.write(f'分割開始 → {self.plan.out_dir}（{counts}）')
        stats = ops.write_plan(self.plan, classes=self.state.classes, move=move,
                               on_progress=lambda d, t: status.step(d, t, '分割中 '))
        log.write(f'  画像 {stats["images"]:,} / ラベル {stats["labels"]:,} を出力')
        for src, err in stats['errors'][:10]:
            log.write(f'  [エラー] {src}: {err}')
        status.reset()
        messagebox.showinfo('完了', f'{stats["images"]:,} 件を分割しました。')
        self._invalidate()
