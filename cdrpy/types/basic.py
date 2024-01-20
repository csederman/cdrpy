"""

"""

from __future__ import annotations

import os
import typing as t


PathLike = t.Union[str, bytes, os.PathLike]

# json types
JSON = t.Union[int, str, float, bool, None, t.Dict[str, "JSON"], t.List["JSON"]]
JSONObject = t.Dict[str, JSON]
JSONList = t.List[JSON]
