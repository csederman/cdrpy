"""Progress bar utils."""

from __future__ import annotations

import typing as t

from tqdm import tqdm


class TQDMReportHook(tqdm):
    """Progress bar compatible with urllib."""

    def update_to(
        self,
        blocks: int = 1,
        block_size: int = 1,
        total_size: int | None = None,
    ) -> None:
        """"""
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def get_tqdm_default_params(file_name: str) -> t.Dict[str, bool | str | int]:
    """"""
    return dict(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        disable=False,
        desc=f"Downloading {file_name}",
        leave=False,
    )
