"""Progress bar utils."""

from __future__ import annotations

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
