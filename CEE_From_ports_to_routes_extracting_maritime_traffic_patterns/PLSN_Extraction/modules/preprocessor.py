import logging
from typing import Iterable

import pandas as pd


class AISPreprocessor:
    """
    Preprocesses AIS data for PLSN extraction.
    """

    def __init__(self, sog_threshold: float = 0.5, nav_status_filter: Iterable[int] = (1, 5)):
        self.sog_threshold = sog_threshold
        self.nav_status_filter = tuple(nav_status_filter)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"Required column not found. Tried: {candidates}")
        return None

    def filter_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alias kept for backward compatibility.
        """
        return self.filter_anchor_mooring(df)

    def filter_anchor_mooring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implements Eq. (1) and Eq. (2) from the paper:
        P_anchoring: status == 1 and sog < 0.5
        P_mooring:   status == 5 and sog < 0.5
        """
        self.logger.info("Preprocessing AIS data for anchoring/mooring points...")
        initial_count = len(df)

        nav_col = self._pick_column(df, ["NAVSTATUS", "Navigation Status", "STATUS"])
        sog_col = self._pick_column(df, ["SOG", "Speed Over Ground", "speed"])
        lat_col = self._pick_column(df, ["LAT", "Latitude"])
        lon_col = self._pick_column(df, ["LON", "Longitude"])
        mmsi_col = self._pick_column(df, ["MMSI"])
        time_col = self._pick_column(df, ["BASEDATETIME", "Timestamp", "DATETIME"], required=False)

        work = df.copy()
        work[nav_col] = pd.to_numeric(work[nav_col], errors="coerce")
        work[sog_col] = pd.to_numeric(work[sog_col], errors="coerce")
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")

        if time_col:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")

        # Paper constraints: anchoring/mooring status + very low speed.
        anchor_mask = (work[nav_col] == 1) & (work[sog_col] < self.sog_threshold)
        mooring_mask = (work[nav_col] == 5) & (work[sog_col] < self.sog_threshold)
        status_speed_mask = anchor_mask | mooring_mask

        quality_mask = (
            work[mmsi_col].notna()
            & work[lat_col].between(-90, 90)
            & work[lon_col].between(-180, 180)
        )
        if time_col:
            quality_mask = quality_mask & work[time_col].notna()

        filtered_df = work[status_speed_mask & quality_mask].copy()

        self.logger.info(
            "Filtered %d rows -> %d anchoring/mooring points (anchor=%d, moored=%d).",
            initial_count,
            len(filtered_df),
            int(anchor_mask.sum()),
            int(mooring_mask.sum()),
        )

        if filtered_df.empty:
            self.logger.warning(
                "No anchoring/mooring points found after filtering. Check NAVSTATUS/SOG columns or threshold."
            )

        return filtered_df
