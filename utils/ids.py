from datetime import datetime

_counters = {}


def _next(prefix: str) -> str:
    """Generate sequential ID with prefix and date."""
    date = datetime.utcnow().strftime("%Y%m%d")
    key = f"{prefix}-{date}"
    _counters[key] = _counters.get(key, 0) + 1
    return f"{key}-{_counters[key]:04d}"


def batch_id():
    return _next("BATCH")


def vision_id():
    return _next("VIS")


def aroma_id():
    return _next("ARO")


def taste_id():
    return _next("TAS")


def corr_id():
    return _next("COR")


def inv_id():
    return _next("INV")


def pass_id():
    return _next("PASS")


def cert_id():
    return _next("CERT")
