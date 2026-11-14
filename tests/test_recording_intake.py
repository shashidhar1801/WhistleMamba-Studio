import os
from pathlib import Path
import shutil
import pytest
from PySide6 import QtCore, QtWidgets
from whistlemamba_studio import recording_browser as rb
from whistlemamba_studio.main_window import MainWindow
DEMO=None

def test_intake_and_dialog(tmp_path, monkeypatch):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=MainWindow(); w.prefs=QtCore.QSettings(str(tmp_path/"test.ini"),QtCore.QSettings.Format.IniFormat)
    w.show();app.processEvents()
    a=tmp_path/"A clip.WaV"; b=tmp_path/"B clip.wav"
    for f in (a,b):shutil.copyfile(DEMO,f)
    monkeypatch.setenv("WHISTLEMAMBA_WINDOWS_DOWNLOADS",str(tmp_path))
    assert w.home.open_button.isVisible()
    def choose(dialog):
        assert dialog.fileMode()==QtWidgets.QFileDialog.FileMode.ExistingFiles
        assert dialog.directory().absolutePath()==str(tmp_path)
        assert str(tmp_path) in [u.toLocalFile() for u in dialog.sidebarUrls()]
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(QtWidgets.QFileDialog,"exec",choose)
    monkeypatch.setattr(QtWidgets.QFileDialog,"selectedFiles",lambda self:[str(b),str(a)])
    w.home.open_button.click();app.processEvents()
    assert w.queue_list.count()==2 and w.project.audio.path==str(b)
    assert w.run_button.isEnabled() and not w.project.analysis_complete
    assert {d.windowTitle() for d in w.panels if d.isVisible()}=={"Analysis","Display","Audio / Project","Contour editor","Recordings"}
    assert w.model_settings_toggle.isChecked()
    assert w.tracker_combo.isVisible() and [w.tracker_combo.itemText(i) for i in range(w.tracker_combo.count())]==["learned","both","heuristic"]
    assert not w.presence_check.isEnabled() and not w.stride_spin.isEnabled()
    assert all(c.isVisible() and c.isChecked() for c in (w.centerline_check,w.contours_check,w.endpoints_check,w.junctions_check))
    for panel in w.panels:panel.hide()
    w.full_layout_action.trigger();app.processEvents()
    assert all(panel.isVisible() for panel in w.panels)
    item=w.queue_list.item(1);w.queue_list.itemClicked.emit(item)
    assert w.project.audio.path==str(a)
    w._intake([tmp_path]);assert w.queue_list.count()==2
    nested=tmp_path/"nested";nested.mkdir();shutil.copyfile(DEMO,nested/"C.WaV")
    w._intake([tmp_path]);assert w.queue_list.count()==2
    w._intake([tmp_path],recursive=True);assert w.queue_list.count()==3
    w.dirty=True;monkeypatch.setattr(w,"_confirm_discard",lambda:False)
    w._open_list_item(w.queue_list.item(0));assert w.project.audio.path==str(a)
    w.dirty=False
    assert rb.local_path(r'"D:\Marine Audio\clip.WAV"')==Path('/mnt/d/Marine Audio/clip.WAV')
    assert rb.local_path("file:///C:/Marine%20Audio/clip.wav")==Path('/mnt/c/Marine Audio/clip.wav')
    prior=w.project;r=[];monkeypatch.setattr(w,'_error',lambda title,msg:r.append(msg))
    w._intake([tmp_path/'missing.wav']);assert r and w.project is prior
    assert w.prefs.value("recordings/lastFolder")==str(tmp_path)
    w.close()

