# Goes into base_library.zip as "pathlib.py". The original pathlib bytecode is
# kept in the zip as "_vcpatch_orig_pathlib.pyc" and run below, unchanged.
#
# Before that, it registers a tiny import finder for the vc_patches folder
# next to the .exe:
#   _VCPATCH_TARGETS: module is replaced by vc_patches/<file>
#   _VCPATCH_POST:    bundled module loads as normal, then apply(module) from
#                     vc_patches/<file> runs on it
# Missing files are skipped, so deleting vc_patches restores stock behaviour.
import sys as _vcp_sys

_VCPATCH_TARGETS = {
    "voice_changer.RVC.embedder.OnnxContentvec": "OnnxContentvec.py",
}
_VCPATCH_POST = {
    "voice_changer.RVC.RVCr2": "RVCr2_post.py",
}


def _vcp_log(*msg):
    # print() output from the frozen app often never reaches a console/log,
    # so also append to vc_patches/vc_patches.log.
    try:
        import os, time

        d = os.path.join(os.path.dirname(_vcp_sys.executable), "vc_patches")
        with open(os.path.join(d, "vc_patches.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg) + "\n")
    except Exception:
        pass
    try:
        print("[vc_patches]", *msg)
    except Exception:
        pass


def _vcp_path(fn):
    import os

    p = os.path.join(os.path.dirname(_vcp_sys.executable), "vc_patches", fn)
    return p if os.path.isfile(p) else None


class _VCPostLoader:
    def __init__(self, orig, patch_file):
        self.orig = orig
        self.patch_file = patch_file

    def create_module(self, spec):
        cm = getattr(self.orig, "create_module", None)
        return cm(spec) if cm else None

    def exec_module(self, module):
        self.orig.exec_module(module)
        try:
            with open(self.patch_file, "r", encoding="utf-8") as f:
                code = compile(f.read(), self.patch_file, "exec")
            g = {"__name__": "_vcp_post_" + module.__name__.replace(".", "_"), "__file__": self.patch_file, "log": _vcp_log}
            exec(code, g)
            g["apply"](module)
            _vcp_log("post-patched", module.__name__)
        except Exception as e:  # never break the app
            _vcp_log("post-patch failed for", module.__name__, repr(e))

    def __getattr__(self, item):  # get_code, is_package, etc.
        return getattr(self.orig, item)


class _VCPatchFinder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        try:
            fn = _VCPATCH_TARGETS.get(name)
            if fn is not None:
                p = _vcp_path(fn)
                if p is None:
                    return None
                import importlib.util

                _vcp_log("loading patched", name)
                return importlib.util.spec_from_file_location(name, p)

            fn = _VCPATCH_POST.get(name)
            if fn is not None:
                p = _vcp_path(fn)
                if p is None:
                    return None
                for finder in _vcp_sys.meta_path:
                    if finder is cls or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec is not None:
                        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                            _vcp_log("cannot post-patch (no exec_module)", name, type(spec.loader))
                            return spec
                        spec.loader = _VCPostLoader(spec.loader, p)
                        return spec
        except Exception as e:  # never break the app
            _vcp_log("patch loader error:", repr(e))
        return None


if not any(getattr(f, "__name__", "") == "_VCPatchFinder" for f in _vcp_sys.meta_path):
    _vcp_sys.meta_path.insert(0, _VCPatchFinder)
    # Flush the app's console output line by line so errors show up in logs.
    for _s in (getattr(_vcp_sys, "stdout", None), getattr(_vcp_sys, "stderr", None)):
        try:
            _s.reconfigure(line_buffering=True)
        except Exception:
            pass


def _vcp_run_original(g):
    import zipimport

    archive = __file__.replace("\\", "/").rsplit("/", 1)[0]
    code = zipimport.zipimporter(archive).get_code("_vcpatch_orig_pathlib")
    exec(code, g)


_vcp_run_original(globals())
del _vcp_run_original
