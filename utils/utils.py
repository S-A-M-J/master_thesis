import json
from jax import numpy as jp


class JaxArrayEncoder(json.JSONEncoder):
    """JSON encoder that converts JAX arrays to Python lists."""
    def default(self, obj):
        if isinstance(obj, jp.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_dict(obj):
    """Convert ConfigDict, JAX arrays, and other non-serializable objects to plain Python types."""
    if hasattr(obj, 'to_dict'):
        # Handle ConfigDict-like objects
        return convert_to_dict(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__
        return convert_to_dict(vars(obj))
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_dict(v) for v in obj]
    elif isinstance(obj, jp.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if obj == float('inf'):
            return 1e308
        if obj == float('-inf'):
            return -1e308
        if obj != obj:  # NaN
            return None
    return obj