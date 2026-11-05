# Goes into base_library.zip as "pathlib.py". The original pathlib bytecode is
# kept in the zip as "_vcpatch_orig_pathlib.pyc" and run below, unchanged.
#
# Before that, it registers a tiny import finder so that any module listed in
# _VCPATCH_TARGETS loads from the vc_patches folder next to the .exe instead
# of from inside the .exe. Delete the vc_patches folder (or the .py file) and
# the app behaves exactly like stock.
import sys as _vcp_sys

_VCPATCH_TARGETS = {
    "voice_changer.RVC.embedder.OnnxContentvec": "OnnxContentvec.py",
}


class _VCPatchFinder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        fn = _VCPATCH_TARGETS.get(name)
        if fn is None:
            return None
        try:
            import os
            import importlib.util

            p = os.path.join(os.path.dirname(_vcp_sys.executable), "vc_patches", fn)
            if not os.path.isfile(p):
                return None
            print("[vc_patches] loading patched", name)
            return importlib.util.spec_from_file_location(name, p)
        except Exception as e:  # never break the app
            print("[vc_patches] patch loader error:", e)
            return None


if not any(getattr(f, "__name__", "") == "_VCPatchFinder" for f in _vcp_sys.meta_path):
    _vcp_sys.meta_path.insert(0, _VCPatchFinder)


def _vcp_run_original(g):
    import zipimport

    archive = __file__.replace("\\", "/").rsplit("/", 1)[0]
    code = zipimport.zipimporter(archive).get_code("_vcpatch_orig_pathlib")
    exec(code, g)


_vcp_run_original(globals())
del _vcp_run_original
