"""まとめるタブ — 複数のデータセットを1つにまとめる。

散らばった収集データをアノテーション用に1つのフォルダへ集約したり、
CVAT へのアップロード用に zip にまとめたりする。
ファイル名が衝突する場合は取得元フォルダ名を頭に付けて回避する。
"""

import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk, messagebox

from .. import ops
from .common import PathRow, SummaryTable


class CollectTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.plan = None

        # --- 設定 ---
        opt = ttk.LabelFrame(self, text='まとめ方')
        opt.pack(fill='x', padx=8, pady=(8, 4))

        self.out_row = PathRow(opt, '出力フォルダ:', on_change=self._invalidate)
        self.out_row.pack(fill='x', padx=8, pady=(8, 4))

        row = ttk.Frame(opt)
        row.pack(fill='x', padx=8, pady=2)
        ttk.Label(row, text='形式:').pack(side='left')
        self.layout_var = tk.StringVar(value='flat')
        for key, label in [('flat', 'images/ labels/（アノテーション用）'),
                           ('yolo', 'train/images/（YOLO 標準・data.yaml 付き）'),
                           ('split', 'images/train/（split_dataset.py と同じ）')]:
            ttk.Radiobutton(row, text=label, value=key, variable=self.layout_var,
                            command=self._invalidate).pack(side='left', padx=(8, 0))

        row2 = ttk.Frame(opt)
        row2.pack(fill='x', padx=8, pady=2)
        self.keep_split = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text='元の train/val の分割を維持する',
                        variable=self.keep_split,
                        command=self._invalidate).pack(side='left')
        self.prefix_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text='ファイル名の先頭に取得元フォルダ名を付ける',
                        variable=self.prefix_var,
                        command=self._invalidate).pack(side='left', padx=(16, 0))
        self.unlabeled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text='ラベルのない画像も含める',
                        variable=self.unlabeled_var,
                        command=self._invalidate).pack(side='left', padx=(16, 0))

        row3 = ttk.Frame(opt)
        row3.pack(fill='x', padx=8, pady=(2, 8))
        self.move_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text='コピーではなく移動する', variable=self.move_var
                        ).pack(side='left')
        self.zip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text='まとめ終わったら zip にする', variable=self.zip_var,
                        command=self._toggle_zip).pack(side='left', padx=(16, 0))
        self.zip_row = PathRow(row3, '', mode='savezip', width=40)
        self.zip_row.pack(side='left', fill='x', expand=True, padx=(8, 0))
        self._toggle_zip()

        # --- 実行ボタン ---
        # 先に下端を確保しておく（expand する「確認」枠より後に pack すると、
        # ウィンドウが小さいときに領域が足りず表示されなくなる）
        self.run_btn = ttk.Button(self, text='② まとめを実行', command=self.run)
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

        body = ttk.Frame(prev)
        body.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self.table = SummaryTable(body, first_col='出力先', height=6)
        self.table.pack(side='left', fill='both', expand=True)

        rn = ttk.LabelFrame(body, text='名前が変わるファイル')
        rn.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.rename_tree = ttk.Treeview(rn, columns=('after',), height=6)
        self.rename_tree.heading('#0', text='変換前')
        self.rename_tree.heading('after', text='変換後')
        self.rename_tree.column('#0', width=200, stretch=True)
        self.rename_tree.column('after', width=220, stretch=True)
        self.rename_tree.pack(fill='both', expand=True, padx=6, pady=6)

    # ------------------------------------------------------------------

    def _toggle_zip(self):
        state = 'normal' if self.zip_var.get() else 'disabled'
        for child in self.zip_row.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _invalidate(self, *_):
        self.plan = None
        self.run_btn.state(['disabled'])
        self.table.clear()
        self.rename_tree.delete(*self.rename_tree.get_children())
        if self.state.items:
            self.summary_var.set('「プレビューを作成」で内容を確認してください')

    def refresh(self):
        self._invalidate()
        if not self.state.items:
            self.summary_var.set('データセットを追加してください')

    # ------------------------------------------------------------------

    def _target_items(self):
        items = self.state.items
        if not self.unlabeled_var.get():
            items = [i for i in items if i.has_label]
        return items

    def preview(self):
        items = self._target_items()
        if not items:
            messagebox.showinfo('データがありません', 'データセットを追加してください。')
            return
        out = self.out_row.get()
        if not out:
            messagebox.showinfo('出力先が未指定', '出力フォルダを指定してください。')
            return

        if self.keep_split.get():
            split_map = defaultdict(list)
            for item in items:
                split_map[item.split or 'train'].append(item)
        else:
            split_map = {'train' if self.layout_var.get() != 'flat' else '': items}

        self.plan = ops.plan_output(split_map, out, layout=self.layout_var.get(),
                                    prefix_source=self.prefix_var.get())

        counts = self.plan.counts_by_split()
        total = self.plan.n
        rows = [(f'{k}', n, n / total * 100 if total else 0)
                for k, n in counts.most_common()]
        self.table.show(rows, total=total)

        self.rename_tree.delete(*self.rename_tree.get_children())
        renamed = [(it, stem) for it, _, stem in self.plan.entries if it.stem != stem]
        for it, stem in renamed[:300]:
            self.rename_tree.insert('', 'end', text=it.stem, values=(stem,))
        if len(renamed) > 300:
            self.rename_tree.insert('', 'end', text=f'... 他 {len(renamed) - 300} 件',
                                    values=('',))

        msg = f'{total:,} 件を出力します'
        if self.plan.renamed:
            msg += f'（名前の重複により {self.plan.renamed} 件をリネーム）'
        if self.plan.skipped:
            msg += f'（画像なしのため {len(self.plan.skipped)} 件を除外）'
        if total:
            msg += '　▶ 下の「② まとめを実行」で出力します'
        self.summary_var.set(msg)
        self.run_btn.state(['!disabled'] if total else ['disabled'])

    def run(self):
        if not self.plan:
            return
        move = self.move_var.get()
        if not messagebox.askyesno(
                '確認',
                f'{self.plan.n:,} 件を {self.plan.out_dir} に'
                f'{"移動" if move else "コピー"}します。\n\n'
                + ('「移動」のため元のファイルは残りません。\n\n' if move else '')
                + '実行しますか？'):
            return

        self.run_btn.state(['disabled'])
        log, status = self.state.log, self.state.status
        log.write(f'まとめ開始 → {self.plan.out_dir}')

        stats = ops.write_plan(
            self.plan, classes=self.state.classes, move=move,
            on_progress=lambda d, t: status.step(d, t, 'まとめ中 '))

        log.write(f'  画像 {stats["images"]:,} / ラベル {stats["labels"]:,} を出力')
        for src, err in stats['errors'][:10]:
            log.write(f'  [エラー] {src}: {err}')

        if self.zip_var.get():
            zip_path = self.zip_row.get() or str(Path(self.plan.out_dir).with_suffix('.zip'))
            n, size = ops.make_zip(self.plan.out_dir, zip_path,
                                   on_progress=lambda d, t: status.step(d, t, 'zip 作成中 '))
            log.write(f'  zip 作成: {zip_path}（{n:,} ファイル / {size/1024/1024:.1f} MB）')

        status.reset()
        messagebox.showinfo('完了', f'{stats["images"]:,} 件をまとめました。')
        self._invalidate()
