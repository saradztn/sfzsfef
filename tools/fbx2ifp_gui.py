"""FBX -> IFP (GTA SA / MTA:SA) converter - Tk GUI for Windows.

    python fbx2ifp_gui.py            (GUI)
    python fbx2ifp_gui.py --cli DFF FBX OUT.ifp [anim_name]   (no window)

1. choose the ped DFF  -> the skeleton (32 bones) and skin bind are read from it
2. choose the FBX      -> the bones are checked against the bone map
3. Convert             -> ready ANP3 .ifp (+ automatic accuracy check)

Pure Python 3 + tkinter (included with the python.org Windows installer).
"""
import os
import queue
import re
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fbx2ifp as conv            # noqa: E402
from dff_rig import load_dff       # noqa: E402
from fbx_rig import load_rig       # noqa: E402

APP_TITLE = 'FBX -> IFP  |  GTA SA / MTA:SA'


# --------------------------------------------------------------------------
# GUI-independent logic (tested headless)
# --------------------------------------------------------------------------


def check_dff(path):
    """-> (sk, skin, info, summary_text)"""
    sk, skin, info = load_dff(path)
    need_ids = sorted({v[0] for v in conv.BONE_MAP.values()} - conv.OPTIONAL_IDS)
    missing = [b for b in need_ids if b not in skin or b not in sk]
    if missing:
        raise ValueError('DFF is missing SA bone ids: %s' % missing)
    txt = '%d bones, skin bind OK (scale %.3f)' % (info['bones'], info['scale'])
    return sk, skin, info, txt


def check_fbx(path):
    """-> (rig, summary_text, missing_list, duration_s)"""
    with open(path, 'rb') as f:
        head = f.read(23)
    if not head.startswith(b'Kaydara FBX Binary'):
        raise ValueError('This FBX is ASCII (text). Re-export it as Binary FBX '
                         '(Blender: FBX export is binary by default; Maya/Max: '
                         'set File type = Binary).')
    rig = load_rig(path)
    rr = conv.resolve_rig(rig)
    missing = rr['missing']
    n_mapped = len({v[0] for v in rr['bone_map'].values()})
    tmax = 0.0
    for c in rig['curves'].values():
        if c.times:
            tmax = max(tmax, c.times[-1])
    txt = '%s rig, clip %.2f s' % (rr['label'], tmax)
    if missing:
        txt += ' - MISSING %d bones' % len(missing)
    else:
        txt += ', %d/26 SA bones driven' % n_mapped
        opt = sorted(conv.OPTIONAL_IDS - {v[0] for v in rr['bone_map'].values()})
        if opt:
            txt += ' (no fingers/toes in FBX: kept in bind pose)'
    if rr.get('rest_src') == 'zero-rotation T-pose':
        txt += ' [no bind pose in FBX: T-pose rebuilt from joint orients]'
    elif rr.get('rest_src') == 'static pose':
        txt += ' [no bind pose in FBX: using its static pose]'
    return rig, txt, missing, tmax


def safe_anim_name(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', s or 'clip').strip('_') or 'clip'
    return s[:23]


def lua_snippet(ifp_path, anim_name, block='myclip'):
    fn = os.path.basename(ifp_path)
    return ('-- client.lua  (add  <file src="%s" />  to meta.xml)\n'
            'local ifp = engineLoadIFP("%s", "%s")\n'
            'if ifp then\n'
            '    setPedAnimation(localPlayer, "%s", "%s", -1, true, false, false)\n'
            'end\n') % (fn, fn, block, block, anim_name)


def run_job(dff_path, fbx_path, out_path, anim_name='clip', fps=30,
            up_axis='Y', validate=True, log=print, dff_data=None):
    """Full pipeline. Returns dict(out, stats, report)."""
    t0 = time.time()
    if dff_data is None:
        log('Reading DFF skeleton: %s' % os.path.basename(dff_path))
        sk, skin, info, txt = check_dff(dff_path)
        log('  ' + txt)
    else:
        sk, skin = dff_data
    anim_name = safe_anim_name(anim_name)
    stats = conv.convert(fbx_path=fbx_path, out_path=out_path,
                         anim_name=anim_name, fps=float(fps), up_axis=up_axis,
                         skel=sk, skin=skin, progress=log)
    log('Written %s (%d bytes, %d tracks, %.2f s)' %
        (out_path, stats['bytes'], stats['tracks'], stats['duration']))
    report = None
    if validate:
        import validate_ifp
        log('')
        log('Checking accuracy against the FBX ...')
        report = validate_ifp.validate(out_path, fbx_path, sk, skin, log=log,
                                       up_axis=up_axis)
    log('Done in %.1f s' % (time.time() - t0))
    return dict(out=out_path, stats=stats, report=report, anim=anim_name)


# --------------------------------------------------------------------------
# Tk GUI
# --------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    try:                                   # crisp text on high-DPI Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry('860x660')
    root.minsize(720, 540)
    style = ttk.Style()
    if 'vista' in style.theme_names():
        style.theme_use('vista')

    q = queue.Queue()
    state = {'dff': None, 'busy': False}

    v_dff = tk.StringVar()
    v_fbx = tk.StringVar()
    v_out = tk.StringVar()
    v_anim = tk.StringVar(value='clip')
    v_block = tk.StringVar(value='myclip')
    v_fps = tk.StringVar(value='30')
    v_up = tk.StringVar(value='Y')
    v_val = tk.BooleanVar(value=True)
    v_dff_st = tk.StringVar(value='-')
    v_fbx_st = tk.StringVar(value='-')
    v_status = tk.StringVar(value='Choose a DFF and an FBX')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)
    frm.columnconfigure(1, weight=1)

    def post(kind, *a):
        q.put((kind, a))

    def bg(fn, *a):
        threading.Thread(target=fn, args=a, daemon=True).start()

    # ---- DFF ----
    def load_dff_bg(path):
        try:
            sk, skin, info, txt = check_dff(path)
            post('dff_ok', (sk, skin), txt)
        except Exception as e:
            post('dff_err', str(e))

    def pick_dff():
        p = filedialog.askopenfilename(title='Ped DFF (skeleton)',
                                       filetypes=[('DFF model', '*.dff'), ('All', '*.*')])
        if p:
            v_dff.set(p)
            state['dff'] = None
            v_dff_st.set('reading ...')
            bg(load_dff_bg, p)

    # ---- FBX ----
    def load_fbx_bg(path):
        try:
            rig, txt, missing, tmax = check_fbx(path)
            post('fbx_ok', txt, missing)
        except Exception as e:
            post('fbx_err', str(e))

    def pick_fbx():
        p = filedialog.askopenfilename(title='Animation FBX',
                                       filetypes=[('FBX', '*.fbx'), ('All', '*.*')])
        if p:
            v_fbx.set(p)
            stem = os.path.splitext(os.path.basename(p))[0]
            if not v_out.get():
                v_out.set(os.path.splitext(p)[0] + '.ifp')
            if v_anim.get() in ('', 'clip'):
                v_anim.set(safe_anim_name(stem) if re.match(r'[A-Za-z]', stem) else 'clip')
            v_fbx_st.set('reading ...')
            bg(load_fbx_bg, p)

    def pick_out():
        p = filedialog.asksaveasfilename(title='Save IFP', defaultextension='.ifp',
                                         filetypes=[('IFP animation', '*.ifp')],
                                         initialfile=os.path.basename(v_out.get() or 'clip.ifp'))
        if p:
            v_out.set(p)

    r = 0
    ttk.Label(frm, text='DFF  (skeleton / العظام):').grid(row=r, column=0, sticky='w')
    ttk.Entry(frm, textvariable=v_dff).grid(row=r, column=1, sticky='we', padx=5)
    ttk.Button(frm, text='Browse...', command=pick_dff).grid(row=r, column=2)
    r += 1
    ttk.Label(frm, textvariable=v_dff_st, foreground='#555').grid(row=r, column=1, sticky='w', padx=5)
    r += 1
    ttk.Label(frm, text='FBX  (animation / الحركة):').grid(row=r, column=0, sticky='w', pady=(6, 0))
    ttk.Entry(frm, textvariable=v_fbx).grid(row=r, column=1, sticky='we', padx=5, pady=(6, 0))
    ttk.Button(frm, text='Browse...', command=pick_fbx).grid(row=r, column=2, pady=(6, 0))
    r += 1
    ttk.Label(frm, textvariable=v_fbx_st, foreground='#555').grid(row=r, column=1, sticky='w', padx=5)
    r += 1
    ttk.Label(frm, text='Output IFP:').grid(row=r, column=0, sticky='w', pady=(6, 0))
    ttk.Entry(frm, textvariable=v_out).grid(row=r, column=1, sticky='we', padx=5, pady=(6, 0))
    ttk.Button(frm, text='Save as...', command=pick_out).grid(row=r, column=2, pady=(6, 0))
    r += 1

    opt = ttk.LabelFrame(frm, text='Options', padding=6)
    opt.grid(row=r, column=0, columnspan=3, sticky='we', pady=8)
    ttk.Label(opt, text='Animation name:').grid(row=0, column=0, sticky='w')
    ttk.Entry(opt, textvariable=v_anim, width=18).grid(row=0, column=1, padx=4)
    ttk.Label(opt, text='MTA block:').grid(row=0, column=2, sticky='w', padx=(10, 0))
    ttk.Entry(opt, textvariable=v_block, width=14).grid(row=0, column=3, padx=4)
    ttk.Label(opt, text='FPS:').grid(row=0, column=4, sticky='w', padx=(10, 0))
    ttk.Combobox(opt, textvariable=v_fps, values=['30', '60'], width=4,
                 state='readonly').grid(row=0, column=5, padx=4)
    ttk.Label(opt, text='FBX up axis:').grid(row=0, column=6, sticky='w', padx=(10, 0))
    ttk.Combobox(opt, textvariable=v_up, values=['Y', 'Z'], width=3,
                 state='readonly').grid(row=0, column=7, padx=4)
    ttk.Checkbutton(opt, text='Verify accuracy after converting',
                    variable=v_val).grid(row=1, column=0, columnspan=4, sticky='w', pady=(4, 0))
    r += 1

    bar = ttk.Frame(frm)
    bar.grid(row=r, column=0, columnspan=3, sticky='we')
    btn = ttk.Button(bar, text='Convert  ->  IFP')
    btn.pack(side='left')
    btn_open = ttk.Button(bar, text='Open folder', state='disabled')
    btn_open.pack(side='left', padx=6)
    pb = ttk.Progressbar(bar, mode='indeterminate', length=200)
    pb.pack(side='left', padx=6, fill='x', expand=True)
    r += 1
    ttk.Label(frm, textvariable=v_status, font=('Segoe UI', 10, 'bold')).grid(
        row=r, column=0, columnspan=3, sticky='w', pady=(6, 2))
    r += 1

    nb = ttk.Notebook(frm)
    nb.grid(row=r, column=0, columnspan=3, sticky='nsew')
    frm.rowconfigure(r, weight=1)
    logw = tk.Text(nb, height=14, wrap='none', font=('Consolas', 9))
    luaw = tk.Text(nb, height=14, wrap='none', font=('Consolas', 10))
    nb.add(logw, text='Log')
    nb.add(luaw, text='MTA Lua')

    def log(s):
        post('log', s)

    def write_log(s):
        logw.insert('end', s + '\n')
        logw.see('end')

    def copy_lua():
        root.clipboard_clear()
        root.clipboard_append(luaw.get('1.0', 'end'))
        v_status.set('Lua code copied')
    ttk.Button(frm, text='Copy Lua', command=copy_lua).grid(row=r + 1, column=2, sticky='e', pady=4)

    def job(dff, fbx, out, anim, fps, up, val, dff_data):
        try:
            res = run_job(dff, fbx, out, anim, fps, up, val, log=log, dff_data=dff_data)
            post('done', res)
        except Exception as e:
            post('log', traceback.format_exc())
            post('fail', str(e))

    def start():
        if state['busy']:
            return
        dff, fbx, out = v_dff.get().strip(), v_fbx.get().strip(), v_out.get().strip()
        for p, what in ((dff, 'DFF'), (fbx, 'FBX')):
            if not p or not os.path.isfile(p):
                messagebox.showerror(APP_TITLE, 'Choose a valid %s file.' % what)
                return
        if not out:
            out = os.path.splitext(fbx)[0] + '.ifp'
            v_out.set(out)
        if not out.lower().endswith('.ifp'):
            out += '.ifp'
            v_out.set(out)
        state['busy'] = True
        btn.config(state='disabled')
        pb.start(12)
        logw.delete('1.0', 'end')
        v_status.set('Converting ... (about 1 minute)')
        bg(job, dff, fbx, out, v_anim.get(), int(v_fps.get()), v_up.get(),
           v_val.get(), state['dff'])

    btn.config(command=start)

    def open_folder():
        p = os.path.dirname(os.path.abspath(v_out.get()))
        try:
            os.startfile(p)          # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(['xdg-open', p])
    btn_open.config(command=open_folder)

    def finish():
        state['busy'] = False
        btn.config(state='normal')
        pb.stop()

    def poll():
        try:
            while True:
                kind, a = q.get_nowait()
                if kind == 'log':
                    write_log(a[0])
                elif kind == 'dff_ok':
                    state['dff'] = a[0]
                    v_dff_st.set('OK  ' + a[1])
                elif kind == 'dff_err':
                    v_dff_st.set('ERROR: ' + a[0])
                    messagebox.showerror(APP_TITLE, 'DFF: ' + a[0])
                elif kind == 'fbx_ok':
                    txt, missing = a
                    v_fbx_st.set(('OK  ' if not missing else 'WARNING  ') + txt)
                    if missing:
                        messagebox.showwarning(
                            APP_TITLE, 'These bones were not found in the FBX:\n\n' +
                            ', '.join(missing) + '\n\nSupported rigs: Mixamo, Newton/Rokoko (same bone names).')
                elif kind == 'fbx_err':
                    v_fbx_st.set('ERROR: ' + a[0])
                    messagebox.showerror(APP_TITLE, 'FBX: ' + a[0])
                elif kind == 'done':
                    res = a[0]
                    finish()
                    btn_open.config(state='normal')
                    rep = res['report']
                    luaw.delete('1.0', 'end')
                    luaw.insert('1.0', lua_snippet(res['out'], res['anim'],
                                                   v_block.get().strip() or 'myclip'))
                    if rep is None:
                        v_status.set('Done: ' + res['out'])
                    elif rep['passed']:
                        v_status.set('PASS  -  max error %.2f deg  ->  %s' %
                                     (max(rep['visual'], rep['segments']), res['out']))
                    else:
                        v_status.set('Written, but the accuracy check FAILED (see Log)')
                elif kind == 'fail':
                    finish()
                    v_status.set('Error: ' + a[0])
                    messagebox.showerror(APP_TITLE, a[0])
        except queue.Empty:
            pass
        root.after(100, poll)

    poll()
    root.mainloop()


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == '--cli':
        dff, fbx, out = sys.argv[2:5]
        anim = sys.argv[5] if len(sys.argv) > 5 else 'clip'
        res = run_job(dff, fbx, out, anim)
        ok = res['report'] is None or res['report']['passed']
        sys.exit(0 if ok else 1)
    run_gui()


if __name__ == '__main__':
    main()
