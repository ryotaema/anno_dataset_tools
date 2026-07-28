"""統合GUI のメインウィンドウ。

上部で読み込んだデータセットを、下のタブ（概要 / まとめる / 分割 /
抽出・間引き / 検証 / 変換）が共通のデータ源として使う。
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from .. import dataset as ds_mod
from .common import LogPane, StatusBar


class AppState:
    """読み込み済みデータセットと、変更通知。"""

    def __init__(self):
        self.datasets = []
        self._listeners = []
        self.log = None
        self.status = None

    # --- 通知 ---
    def add_listener(self, fn):
        self._listeners.append(fn)

    def notify(self):
        for fn in self._listeners:
            fn()

    # --- データ ---
    @property
    def items(self):
        return [item for ds in self.datasets for item in ds.items]

    @property
    def classes(self):
        """複数データセットのクラス定義のうち最も項目数が多いものを使う。"""
        best = []
        for ds in self.datasets:
            if len(ds.classes) > len(best):
                best = ds.classes
        return best

    def class_conflict(self):
        """データセット間でクラスIDと名前の対応が食い違っていないか。"""
        seen = {}
        for ds in self.datasets:
            for i, name in enumerate(ds.classes):
                if i in seen and seen[i] != name:
                    return f'クラスID {i}: 「{seen[i]}」と「{name}」で食い違っています'
                seen[i] = name
        return None

    def add(self, root):
        root = Path(root).expanduser().resolve()
        if any(d.root == root for d in self.datasets):
            return None
        ds = ds_mod.load(root, progress=self._progress)
        self.datasets.append(ds)
        return ds

    def _progress(self, done, total):
        if self.status:
            self.status.step(done, total, '読み込み中 ')


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('データセット整理ツール')
        self.geometry('1120x900')
        self.minsize(940, 680)
        self.state = AppState()

        self._build_sources()
        self._build_tabs()
        self._build_footer()
        self.state.add_listener(self._refresh_sources)

    # ------------------------------------------------------- データセット一覧

    def _build_sources(self):
        frame = ttk.LabelFrame(self, text='データセット（複数読み込めます）')
        frame.pack(fill='x', padx=10, pady=(10, 4))

        cols = ('layout', 'images', 'labeled', 'classes')
        self.tree = ttk.Treeview(frame, columns=cols, height=4, selectmode='extended')
        self.tree.heading('#0', text='パス')
        for c, t, w in [('layout', '形式', 90), ('images', '画像', 80),
                        ('labeled', 'ラベル付き', 90), ('classes', 'クラス', 70)]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor='e' if c != 'layout' else 'center',
                             stretch=False)
        self.tree.column('#0', width=560, stretch=True)
        self.tree.pack(side='left', fill='both', expand=True, padx=(8, 4), pady=8)

        btns = ttk.Frame(frame)
        btns.pack(side='left', fill='y', padx=(0, 8), pady=8)
        ttk.Button(btns, text='フォルダを追加...', command=self.add_dataset).pack(fill='x')
        ttk.Button(btns, text='選択を削除',       command=self.remove_dataset).pack(fill='x', pady=(4, 0))
        ttk.Button(btns, text='再読み込み',       command=self.reload_all).pack(fill='x', pady=(4, 0))

        self.total_var = tk.StringVar(value='データセットを追加してください')
        ttk.Label(self, textvariable=self.total_var).pack(anchor='w', padx=16)

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=10, pady=6)

        from .tab_overview import OverviewTab
        from .tab_collect  import CollectTab
        from .tab_split    import SplitTab
        from .tab_subset   import SubsetTab
        from .tab_tools    import ValidateTab, ConvertTab

        self.tabs = []
        for cls, title in [(OverviewTab, '概要'),
                           (CollectTab,  'まとめる'),
                           (SplitTab,    '分割'),
                           (SubsetTab,   '抽出・間引き'),
                           (ValidateTab, '検証'),
                           (ConvertTab,  '変換')]:
            tab = cls(self.nb, self.state)
            self.nb.add(tab, text=title)
            self.tabs.append(tab)
            self.state.add_listener(tab.refresh)

    def _build_footer(self):
        self.status = StatusBar(self)
        self.status.pack(fill='x', padx=10)
        self.log = LogPane(self, height=6)
        self.log.pack(fill='x', padx=10, pady=(4, 10))
        self.state.log = self.log
        self.state.status = self.status

    # ------------------------------------------------------------- 操作

    def add_dataset(self):
        d = filedialog.askdirectory(title='データセットのフォルダを選択')
        if not d:
            return
        try:
            ds = self.state.add(d)
        except OSError as e:
            messagebox.showerror('読み込みエラー', str(e))
            return
        self.status.reset()
        if ds is None:
            self.log.write(f'既に読み込み済みです: {d}')
            return
        self.log.write(f'読み込み: {ds.root}  形式={ds.layout or "不明"}  '
                       f'画像={ds.n_images:,}  ラベル付き={ds.n_labeled:,}')
        for w in ds.warnings:
            self.log.write(f'  ⚠ {w}')
        conflict = self.state.class_conflict()
        if conflict:
            self.log.write(f'  ⚠ クラス定義の不一致 — {conflict}')
        self.state.notify()

    def remove_dataset(self):
        for iid in self.tree.selection():
            idx = self.tree.index(iid)
            del self.state.datasets[idx]
        self.state.notify()

    def reload_all(self):
        roots = [d.root for d in self.state.datasets]
        self.state.datasets.clear()
        for r in roots:
            self.state.add(r)
        self.status.reset()
        self.log.write(f'{len(roots)} 件を再読み込みしました')
        self.state.notify()

    def _refresh_sources(self):
        self.tree.delete(*self.tree.get_children())
        for ds in self.state.datasets:
            self.tree.insert('', 'end', text=str(ds.root),
                             values=(ds.layout or '?', f'{ds.n_images:,}',
                                     f'{ds.n_labeled:,}', len(ds.classes)))
        items = self.state.items
        if items:
            labeled = sum(1 for i in items if i.has_label)
            boxes = sum(i.n_boxes for i in items)
            self.total_var.set(
                f'合計 {len(items):,} 件  /  ラベル付き {labeled:,} 件  /  '
                f'BBox {boxes:,} 個  /  クラス {len(self.state.classes)} 種')
        else:
            self.total_var.set('データセットを追加してください')


def main():
    App().mainloop()
