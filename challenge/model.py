from typing import List, Tuple, Union

import pandas as pd
import xgboost as xgb


FEATURE_COLUMNS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        features = features.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        if target_column is None:
            return features

        data = data.copy()
        if target_column not in data.columns:
            data[target_column] = self._get_delay_target(data)

        return features, data[[target_column]]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        negative_count = int((target.iloc[:, 0] == 0).sum())
        positive_count = int((target.iloc[:, 0] == 1).sum())
        scale_pos_weight = negative_count / positive_count

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        self._model.fit(features, target.iloc[:, 0])

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * features.shape[0]

        return [int(prediction) for prediction in self._model.predict(features)]

    @staticmethod
    def _get_delay_target(data: pd.DataFrame) -> pd.Series:
        fecha_o = pd.to_datetime(data["Fecha-O"])
        fecha_i = pd.to_datetime(data["Fecha-I"])
        min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60
        return (min_diff > 15).astype(int)
