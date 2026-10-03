"""StaySearch offline pipeline."""
import sys as _sys

# The host Python ships numexpr and bottleneck builds compiled against NumPy 1.x, which print a
# long ABI error when pandas probes for them. They are optional for pandas, so hide them.
for _m in ("numexpr", "bottleneck"):
    _sys.modules.setdefault(_m, None)
