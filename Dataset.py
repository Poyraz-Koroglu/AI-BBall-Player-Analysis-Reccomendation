from __future__ import annotations
from typing import Callable, Iterable, Optional, Union, Dict, Any
import pandas as pd
from pathlib import Path

class ExcelBasketballDataset:
    """
    Lightweight dataset wrapper for an Excel-like player stats file
    (not yet in SQL). Enforces schema and offers simple filtering/transform.

    Expected columns:
        - league (str)
        - player (str)
        - pts (float/int)
        - ast (float/int)
        - reb (float/int)
        - birth_year (int)

    Parameters
    ----------
    path : str | Path
        Path to the Excel/CSV file.
    sheet : str | int | None
        Excel sheet name/index if reading .xlsx/.xls. Ignored for CSV.
    file_type : {"excel", "csv", None}
        Force file type. If None, inferred from file extension.
    filters : Optional[Dict[str, Iterable]]
        Column-based filters. Example:
            filters={"league": ["NBA", "EuroLeague"]}
    transform : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]
        A function applied to each row dict on access (e.g., to tensors).
    strict : bool
        If True, raise on missing columns. If False, fill missing with NaN.
    dtype_overrides : Optional[Dict[str, str]]
        Pandas dtype strings to coerce specific columns, e.g., {"birth_year": "Int64"}.
    dropna_rows : bool
        If True, drop rows with NA in required columns after loading/coercion.

    Notes
    -----
    - Uses pandas.read_excel(engine="openpyxl") for .xlsx and engine="xlrd" for .xls by default.
    - For CSV, uses pandas.read_csv.
    """
    REQUIRED_COLS = ["league", "player", "pts", "ast", "reb", "birth_year"]

    def __init__(
        self,
        path: Union[str, Path],
        sheet: Optional[Union[str, int]] = None,
        file_type: Optional[str] = None,
        filters: Optional[Dict[str, Iterable]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        strict: bool = True,
        dtype_overrides: Optional[Dict[str, str]] = None,
        dropna_rows: bool = False,
    ):
        self.path = Path(path)
        self.sheet = sheet
        self.file_type = file_type or self._infer_type(self.path)
        self.filters = filters or {}
        self.transform = transform
        self.strict = strict
        self.dtype_overrides = dtype_overrides or {}
        self.dropna_rows = dropna_rows

        self._df = self._load()
        self._validate_and_cast()
        self._clean_strings()
        self._apply_filters()
        if self.dropna_rows:
            self._df = self._df.dropna(subset=self.REQUIRED_COLS)

        # Reset index for stable __getitem__
        self._df = self._df.reset_index(drop=True)

    @staticmethod
    def _infer_type(path: Path) -> str:
        ext = path.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            return "excel"
        elif ext in [".csv", ".txt"]:
            return "csv"
        raise ValueError(f"Could not infer file type from extension '{ext}'. Provide file_type='excel' or 'csv'.")

    def _load(self) -> pd.DataFrame:
        if self.file_type == "excel":
            ext = self.path.suffix.lower()
            if ext == ".xlsx":
                return pd.read_excel(self.path, sheet_name=self.sheet, engine="openpyxl")
            elif ext == ".xls":
                return pd.read_excel(self.path, sheet_name=self.sheet, engine="xlrd")
            else:
                # Fallback if someone forces excel but extension is odd
                return pd.read_excel(self.path, sheet_name=self.sheet)
        elif self.file_type == "csv":
            # Good default for UTF-8; adjust if needed
            return pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported file_type: {self.file_type}")

    def _validate_and_cast(self):
        missing = [c for c in self.REQUIRED_COLS if c not in self._df.columns]
        if missing and self.strict:
            raise ValueError(f"Missing required columns: {missing}. Found: {list(self._df.columns)}")
        # If not strict, create missing with NaN
        for c in missing:
            self._df[c] = pd.NA

        # Dtype defaults
        default_dtypes = {
            "league": "string",
            "player": "string",
            "pts": "float64",
            "ast": "float64",
            "reb": "float64",
            "birth_year": "Int64",  # nullable integer
        }
        # Merge overrides
        dtypes = {**default_dtypes, **self.dtype_overrides}

        # Coerce numerics safely
        for col, dt in dtypes.items():
            if col not in self._df.columns:
                continue
            if col in ["pts", "ast", "reb"]:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
            elif col == "birth_year":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
            elif col in ["league", "player"]:
                self._df[col] = self._df[col].astype("string")
            else:
                # generic cast
                try:
                    self._df[col] = self._df[col].astype(dt)
                except Exception:
                    pass

    def _clean_strings(self):
        for col in ["league", "player"]:
            if col in self._df.columns:
                self._df[col] = self._df[col].astype("string").str.strip()

    def _apply_filters(self):
        df = self._df
        for col, allowed in self.filters.items():
            if col in df.columns:
                df = df[df[col].isin(set(allowed))]
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._df.iloc[idx]
        item = {
            "league": row.get("league"),
            "player": row.get("player"),
            "pts": float(row.get("pts")) if pd.notna(row.get("pts")) else None,
            "ast": float(row.get("ast")) if pd.notna(row.get("ast")) else None,
            "reb": float(row.get("reb")) if pd.notna(row.get("reb")) else None,
            "birth_year": int(row.get("birth_year")) if pd.notna(row.get("birth_year")) else None,
        }
        if self.transform:
            item = self.transform(item)
        return item

    # Convenience accessors
    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the internal pandas DataFrame (read-only usage recommended)."""
        return self._df.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for dataframe()."""
        return self.dataframe

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    def unique_leagues(self) -> list[str]:
        return sorted([x for x in self._df["league"].dropna().unique().tolist()])

    def describe_stats(self) -> pd.DataFrame:
        """Quick numeric summary of pts/ast/reb."""
        cols = [c for c in ["pts", "ast", "reb"] if c in self._df.columns]
        return self._df[cols].describe()