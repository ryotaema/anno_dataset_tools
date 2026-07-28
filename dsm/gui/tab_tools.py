"""検証タブ・変換タブ — 既存スクリプト（scripts/ と cvat_conv/）をGUIから使う。"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from .. import dataset as ds_mod
from .common import PathRow

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts'
CVAT = REPO / 'cvat_conv'


def _import_validator():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import anno_validator
    return anno_validator


# ============================================================ 検証

class ValidateTab(ttk.Frame):
    """読み込み済みデータセットに anno_validator の検査をかける。"""

    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.result = None

        opt = ttk.LabelFrame(self, text='検査の設定')
        opt.pack(fill='x', padx=8, pady=(8, 4))
        row = ttk.Frame(opt)
        row.pack(fill='x', padx=8, pady=8)

        self.vars = {}
        for key, label, default, inc in [
                ('iou', '重複とみなす IoU', 0.95, 0.01),
                ('min_size', 'BBox 最小辺', 0.005, 0.001),
                ('max_size', 'BBox 最大辺', 0.99, 0.01),
        ]:
            ttk.Label(row, text=f'{label}:').pack(side='left')
            var = tk.DoubleVar(value=default)
            ttk.Spinbox(row, from_=0, to=1, increment=inc, textvariable=var, width=7
                        ).pack(side='left', padx=(4, 14))
            self.vars[key] = var
        ttk.Label(row, text='クラス最小サンプル数:').pack(side='left')
        self.vars['min_samples'] = tk.IntVar(value=10)
        ttk.Spinbox(row, from_=0, to=10000, textvariable=self.vars['min_samples'],
                    width=7).pack(side='left', padx=(4, 0))

        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=8, pady=4)
        ttk.Button(bar, text='検証を実行', command=self.run).pack(side='left')
        ttk.Button(bar, text='JSONに保存', command=self.save_json).pack(side='left', padx=(6, 0))
        self.summary_var = tk.StringVar(value='データセットを追加してください')
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=(12, 0))

        cols = ('code', 'file', 'msg')
        self.tree = ttk.Treeview(self, columns=cols, height=16)
        self.tree.heading('#0', text='レベル')
        self.tree.heading('code', text='種類')
        self.tree.heading('file', text='ファイル')
        self.tree.heading('msg', text='内容')
        self.tree.column('#0', width=90, stretch=False)
        self.tree.column('code', width=160, stretch=False)
        self.tree.column('file', width=240, stretch=False)
        self.tree.column('msg', width=460, stretch=True)
        self.tree.tag_configure('ERROR', foreground='#b00')
        self.tree.tag_configure('WARNING', foreground='#a60')
        self.tree.pack(fill='both', expand=True, padx=8, pady=(0, 8))

    def refresh(self):
        if not self.state.items:
            self.summary_var.set('データセットを追加してください')

    def run(self):
        if not self.state.datasets:
            messagebox.showinfo('データがありません', 'データセットを追加してください。')
            return
        try:
            av = _import_validator()
        except ImportError as e:
            messagebox.showerror('読み込みエラー', f'anno_validator を読み込めません:\n{e}')
            return

        classes = self.state.classes
        result = av.ValidationResult()
        log, status = self.state.log, self.state.status
        log.write('検証を開始します')

        targets = []
        for ds in self.state.datasets:
            _, pairs = ds_mod.detect_layout(ds.root)
            for img_dir, lbl_dir, split in pairs:
                if lbl_dir and lbl_dir.is_dir():
                    targets.append((lbl_dir, img_dir))

        total = sum(len([p for p in lbl.glob('*.txt') if p.name != 'classes.txt'])
                    for lbl, _ in targets)
        done = 0
        for lbl_dir, img_dir in targets:
            files = sorted(p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt')
            for path in files:
                bboxes = av.validate_file(path, classes, self.vars['min_size'].get(),
                                          self.vars['max_size'].get(), result)
                if len(bboxes) >= 2:
                    av.check_duplicate_labels(path.name, bboxes,
                                              self.vars['iou'].get(), result)
                done += 1
                if done % 100 == 0 or done == total:
                    status.step(done, total, '検証中 ')
            if img_dir and img_dir.is_dir():
                av.check_orphan_files(lbl_dir, img_dir, result)

        av.check_class_imbalance(classes, result.class_counts,
                                 self.vars['min_samples'].get(), result)
        self.result = result
        status.reset()

        self.tree.delete(*self.tree.get_children())
        order = {'ERROR': 0, 'WARNING': 1, 'INFO': 2}
        issues = sorted(result.issues, key=lambda i: order.get(i.level, 3))
        for issue in issues[:2000]:
            self.tree.insert('', 'end', text=issue.level, tags=(issue.level,),
                             values=(issue.code, issue.file, issue.message))
        if len(issues) > 2000:
            self.tree.insert('', 'end', text='', values=('', '',
                             f'... 他 {len(issues) - 2000} 件（JSONに保存して確認してください）'))

        msg = (f'{result.total_files:,} ファイル / {result.total_objects:,} BBox  —  '
               f'ERROR {result.error_count} / WARNING {result.warning_count} / '
               f'INFO {result.info_count}')
        self.summary_var.set(msg)
        log.write('  ' + msg)

    def save_json(self):
        if not self.result:
            messagebox.showinfo('結果がありません', '先に検証を実行してください。')
            return
        path = filedialog.asksaveasfilename(
            title='検証結果を保存', defaultextension='.json',
            initialfile='validation_report.json', filetypes=[('JSON', '*.json')])
        if not path:
            return
        av = _import_validator()
        av.save_report_json(self.result, Path(path))
        self.state.log.write(f'検証結果を保存しました: {path}')


# ============================================================ 変換

CONVERSIONS = [
    ('yolo2voc', 'YOLO → Pascal VOC'),
    ('voc2yolo', 'Pascal VOC → YOLO'),
    ('cvat2yolo', 'CVAT XML → YOLO'),
    ('cvat2pose', 'CVAT XML → YOLO-pose'),
]


class ConvertTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state

        top = ttk.LabelFrame(self, text='変換の種類')
        top.pack(fill='x', padx=8, pady=(8, 4))
        row = ttk.Frame(top)
        row.pack(fill='x', padx=8, pady=8)
        self.kind = tk.StringVar(value='yolo2voc')
        for key, label in CONVERSIONS:
            ttk.Radiobutton(row, text=label, value=key, variable=self.kind,
                            command=self._switch).pack(side='left', padx=(0, 14))

        self.body = ttk.LabelFrame(self, text='入出力')
        self.body.pack(fill='x', padx=8, pady=4)

        # 種類ごとの入力欄（必要なものだけ表示する）
        self.rows = {}
        self.frames = {}
        specs = {
            'yolo2voc':  [('labels', 'YOLO labels フォルダ:', 'dir'),
                          ('images', '画像フォルダ:', 'dir'),
                          ('out',    'VOC XML 出力先:', 'dir'),
                          ('classes', 'classes.txt:', 'file')],
            'voc2yolo':  [('voc',    'VOC XML フォルダ:', 'dir'),
                          ('images', '画像フォルダ:', 'dir'),
                          ('out',    'YOLO txt 出力先:', 'dir'),
                          ('classes', 'classes.txt:', 'file')],
            'cvat2yolo': [('xml',    'CVAT annotations.xml:', 'file'),
                          ('out',    'YOLO txt 出力先:', 'dir'),
                          ('classes', 'classes.txt:', 'file')],
            'cvat2pose': [('xml',    'CVAT annotations.xml:', 'file'),
                          ('out',    'YOLO-pose txt 出力先:', 'dir')],
        }
        for kind, fields in specs.items():
            frame = ttk.Frame(self.body)
            self.frames[kind] = frame
            self.rows[kind] = {}
            for key, label, mode in fields:
                r = PathRow(frame, label, mode=mode)
                r.pack(fill='x', padx=8, pady=3)
                self.rows[kind][key] = r

        self.dry_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text='dry-run（書き込まずに確認だけ）', variable=self.dry_var
                        ).pack(anchor='w', padx=16, pady=(4, 0))
        ttk.Button(self, text='変換を実行', command=self.run).pack(anchor='w', padx=8, pady=8)

        ttk.Label(self, foreground='gray', justify='left', wraplength=900,
                  text='classes.txt を空欄にすると、読み込み済みデータセットのクラス定義を使います。'
                       ).pack(anchor='w', padx=16)
        self._switch()

    def refresh(self):
        # 読み込み済みデータセットがあれば入力欄の初期値を埋めておく
        if not self.state.datasets:
            return
        root = self.state.datasets[0].root
        classes_file = next((p for p in (root / 'classes.txt', root / 'labels' / 'classes.txt')
                             if p.is_file()), None)
        for kind, rows in self.rows.items():
            if 'labels' in rows and not rows['labels'].get():
                rows['labels'].set(root / 'labels')
            if 'images' in rows and not rows['images'].get():
                rows['images'].set(root / 'images')
            if 'classes' in rows and not rows['classes'].get() and classes_file:
                rows['classes'].set(classes_file)

    def _switch(self):
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[self.kind.get()].pack(fill='x', pady=6)

    # ------------------------------------------------------------------

    def _classes_file(self, rows):
        """classes.txt のパス。未指定なら読み込み済みクラスから一時ファイルを作る。"""
        given = rows.get('classes').get() if 'classes' in rows else ''
        if given:
            return given
        if not self.state.classes:
            return ''
        tmp = Path(rows['out'].get() or '.').expanduser() / 'classes.txt'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text('\n'.join(self.state.classes) + '\n')
        self.state.log.write(f'classes.txt を生成しました: {tmp}')
        return str(tmp)

    def run(self):
        kind = self.kind.get()
        rows = self.rows[kind]
        missing = [k for k, r in rows.items() if k != 'classes' and not r.get()]
        if missing:
            messagebox.showinfo('未入力', '入出力のパスをすべて指定してください。')
            return

        log = self.state.log
        try:
            if kind in ('yolo2voc', 'voc2yolo'):
                self._run_script(kind, rows)
            else:
                self._run_cvat(kind, rows)
        except Exception as e:
            log.write(f'[エラー] {e}')
            messagebox.showerror('変換エラー', str(e))
            return
        messagebox.showinfo('完了', '変換が終わりました。詳細はログを見てください。')

    def _run_script(self, kind, rows):
        classes = self._classes_file(rows)
        if kind == 'yolo2voc':
            cmd = [sys.executable, str(SCRIPTS / 'yolo_to_pascal.py'),
                   '--yolo_labels_dir', rows['labels'].get(),
                   '--yolo_images_dir', rows['images'].get(),
                   '--voc_save_dir', rows['out'].get()]
        else:
            cmd = [sys.executable, str(SCRIPTS / 'pascal_to_yolo.py'),
                   '--voc_dir', rows['voc'].get(),
                   '--img_dir', rows['images'].get(),
                   '--yolo_save_dir', rows['out'].get()]
        if classes:
            cmd += ['--class_file', classes]
        if self.dry_var.get():
            cmd.append('--dry_run')

        log = self.state.log
        log.write('$ ' + ' '.join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # 数千行になることがあるので、先頭と末尾だけログに出す
        lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
        if len(lines) > 40:
            shown = lines[:5] + [f'... 中略 {len(lines) - 35} 行 ...'] + lines[-30:]
        else:
            shown = lines
        for line in shown:
            log.write('  ' + line)
        log.write(f'  終了コード: {proc.returncode}')

    def _run_cvat(self, kind, rows):
        if str(CVAT) not in sys.path:
            sys.path.insert(0, str(CVAT))
        log = self.state.log
        out = rows['out'].get()
        Path(out).mkdir(parents=True, exist_ok=True)

        if self.dry_var.get():
            log.write('dry-run: CVAT 変換は dry-run に対応していないため実行しません')
            return

        if kind == 'cvat2yolo':
            from cvat_to_yolo import convert_cvat_to_yolo
            classes_path = self._classes_file(rows)
            classes = ([l.strip() for l in Path(classes_path).read_text().splitlines()
                        if l.strip()] if classes_path else self.state.classes)
            if not classes:
                raise ValueError('クラス一覧が必要です（classes.txt を指定してください）')
            log.write(f'CVAT → YOLO 変換: classes={classes}')
            convert_cvat_to_yolo(rows['xml'].get(), out, classes)
        else:
            from cvat_to_yolo_pose import convert_cvat_to_yolo_pose
            log.write('CVAT → YOLO-pose 変換')
            convert_cvat_to_yolo_pose(rows['xml'].get(), out)

        n = len(list(Path(out).glob('*.txt')))
        log.write(f'  出力: {n} ファイル → {out}')
