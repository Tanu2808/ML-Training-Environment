import pandas as pd


class PreprocessingPipeline:
    """A minimal preprocessing pipeline that keeps the framework usable immediately."""

    def fit(self, df: pd.DataFrame):
        self.columns = list(df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
