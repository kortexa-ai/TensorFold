"""The native Linear descriptors, exposed only for the Inkling development probe."""
import ctypes as C


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [("algo", Algo), ("workspace", C.c_size_t), ("state", C.c_int),
                ("waves", C.c_float), ("reserved", C.c_int * 4)]


class Linear:
    def __init__(self, rows, n, k, out_type, workspace):
        self.lib = C.CDLL("libcublasLt.so.13")
        p, i, u, z = C.c_void_p, C.c_int, C.c_uint64, C.c_size_t
        signatures = {
            "Create": [C.POINTER(p)], "Destroy": [p],
            "MatmulDescCreate": [C.POINTER(p), i, i], "MatmulDescDestroy": [p],
            "MatmulDescSetAttribute": [p, i, p, z],
            "MatrixLayoutCreate": [C.POINTER(p), i, u, u, C.c_int64], "MatrixLayoutDestroy": [p],
            "MatmulPreferenceCreate": [C.POINTER(p)], "MatmulPreferenceDestroy": [p],
            "MatmulPreferenceSetAttribute": [p, i, p, z],
            "MatmulAlgoGetHeuristic": [p, p, p, p, p, p, p, i, C.POINTER(Heuristic), C.POINTER(i)],
            "Matmul": [p, p, p, p, p, p, p, p, p, p, p, p, C.POINTER(Algo), p, z, p],
        }
        for name, args in signatures.items():
            f = getattr(self.lib, "cublasLt" + name)
            f.argtypes, f.restype = args, i
        self.lib.cublasLtGetVersion.restype = z
        self.version = self.lib.cublasLtGetVersion()
        self.owned = []
        self.handle = self.create("Create")
        self.desc = self.create("MatmulDescCreate", 68, 0)
        ta, tb, limit = i(1), i(0), u(workspace.numel())
        self.call("MatmulDescSetAttribute", self.desc, 3, C.byref(ta), C.sizeof(ta))
        self.call("MatmulDescSetAttribute", self.desc, 4, C.byref(tb), C.sizeof(tb))
        self.a = self.create("MatrixLayoutCreate", 14, k, n, k)
        self.b = self.create("MatrixLayoutCreate", 14, k, rows, k)
        self.c = self.create("MatrixLayoutCreate", out_type, n, rows, n)
        self.pref = self.create("MatmulPreferenceCreate")
        self.call("MatmulPreferenceSetAttribute", self.pref, 1, C.byref(limit), C.sizeof(limit))
        self.workspace = workspace
        self.alpha, self.beta = C.c_float(1), C.c_float(0)
        self.baseline = self.heuristics(1)[0]
        self.plans = self.heuristics(16)
        if bytes(self.baseline.algo) != bytes(self.plans[0].algo):
            raise RuntimeError("Asking for more heuristics changed the baseline plan")

    def call(self, name, *args):
        status = getattr(self.lib, "cublasLt" + name)(*args)
        if status:
            raise RuntimeError(f"cuBLASLt {name}: {status}")

    def create(self, name, *args):
        value = C.c_void_p()
        self.call(name, C.byref(value), *args)
        destroy = {"Create": "Destroy", "MatmulDescCreate": "MatmulDescDestroy",
                   "MatrixLayoutCreate": "MatrixLayoutDestroy",
                   "MatmulPreferenceCreate": "MatmulPreferenceDestroy"}[name]
        self.owned.append((destroy, value))
        return value

    def heuristics(self, count):
        found, length = (Heuristic * count)(), C.c_int()
        self.call("MatmulAlgoGetHeuristic", self.handle, self.desc, self.a, self.b, self.c, self.c,
                  self.pref, count, found, C.byref(length))
        plans = [h for h in found[:length.value] if h.state == 0 and h.workspace <= self.workspace.numel()]
        if not plans:
            raise RuntimeError("No supported plan")
        return plans

    def run(self, plan, x, w, y, stream):
        self.call("Matmul", self.handle, self.desc, C.byref(self.alpha), w.data_ptr(), self.a,
                  x.data_ptr(), self.b, C.byref(self.beta), y.data_ptr(), self.c, y.data_ptr(), self.c,
                  C.byref(plan.algo), self.workspace.data_ptr(), self.workspace.numel(), stream)

    def close(self):
        for destroy, value in reversed(self.owned):
            self.call(destroy, value)
        self.owned.clear()
