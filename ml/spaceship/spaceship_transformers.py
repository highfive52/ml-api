import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def to_str(X):
    return X.astype(str)


class LogicalImputeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Impute missing values for Cabin and Name with logical defaults
        X["Cabin"] = X["Cabin"].fillna("Unknown/0/U")
        X["Name"] = X["Name"].fillna("Unknown Unknown")

        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        # Rule 1: CryoSleep NaN + zero spending → CryoSleep = 1
        cond_null = X["CryoSleep"].isnull()
        cond_zero_spend = X[spend_cols].sum(axis=1) == 0
        rule1 = cond_null & cond_zero_spend
        X.loc[rule1, "CryoSleep"] = 1

        # Rule 2: CryoSleep NaN + any spending > 0 → CryoSleep = 0
        cond_null = X["CryoSleep"].isnull()
        cond_spend = X[spend_cols].sum(axis=1) > 0
        rule2 = cond_null & cond_spend
        X.loc[rule2, "CryoSleep"] = 0

        # Rule 3: CryoSleep = 1 + spending NaN → spending = 0
        cond_cryo_true = X["CryoSleep"] == 1
        for col in spend_cols:
            X.loc[cond_cryo_true, col] = X.loc[cond_cryo_true, col].fillna(0)

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


class FeatureEngineer(BaseEstimator, TransformerMixin):

    NUMERIC_FEATURES = [
        "Age",
        "GroupSize",
        "IsAlone",
        "TotalSpend",
        "LuxurySpend",
        "BasicSpend",
        "SpendVariance",
        "IsAwakeZeroSpender",
    ]
    ORDINAL_FEATURES = ["Cabin_Num_Binned", "AgeBin"]
    CATEGORICAL_FEATURES = [
        "Deck",
        "Side",
        "CryoSleep",
        "VIP",
        "HomePlanet",
        "Destination",
    ]

    OUTPUT_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES

    def fit(self, X, y=None):

        cabin_split = X["Cabin"].str.split("/", expand=True)
        cabin_num = pd.to_numeric(cabin_split[1], errors="coerce")

        # store bin edges
        self.cabin_bins_ = pd.qcut(cabin_num, q=3, retbins=True, duplicates="drop")[1]

        return self

    def transform(self, X):
        X = X.copy().reset_index(drop=True)

        # 2.1 PassengerId features
        # Extract GroupId and IndividualId from PassengerId and calculate GroupSize
        X["GroupId"] = X["PassengerId"].str[:4]
        X["IndividualId"] = X["PassengerId"].str[5:].astype(int)

        X["GroupSize"] = (
            X.groupby("GroupId")["IndividualId"].transform("max").to_numpy()
        )

        # 2.1 Cabin features
        # Create IsAlone feature based on GroupSize
        X["IsAlone"] = (X["GroupSize"] == 1).astype(int).to_numpy()

        # Split Cabin into Deck and Side
        cabin_split = X["Cabin"].str.split("/", expand=True)
        X["Deck"] = cabin_split[0].to_numpy()
        X["Cabin_Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
        X["Side"] = cabin_split[2].to_numpy()
        X["Cabin_Num_Binned"] = pd.cut(
            X["Cabin_Num"],
            bins=self.cabin_bins_,
            labels=["Low", "Mid", "High"][: len(self.cabin_bins_) - 1],
            include_lowest=True,
        )

        # 2.3 Name features
        X["LastName"] = X["Name"].str.split(" ").str[1]
        X["FamilySize"] = X.groupby("LastName")["LastName"].transform("count")

        # 2.4 Age features
        # Age bins
        X["AgeBin"] = pd.cut(
            X["Age"],
            bins=[0, 12, 17, 25, 65, 200],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            include_lowest=True,
        )

        # 2.5 Spend features
        # Create CombinedServices as the sum of RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck
        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        X["TotalSpend"] = X[spend_cols].sum(axis=1)
        X["BasicSpend"] = X[["FoodCourt", "ShoppingMall"]].sum(axis=1)
        X["LuxurySpend"] = X[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
        X["SpendVariance"] = X[spend_cols].var(axis=1)
        X["IsAwakeZeroSpender"] = (
            (X["CryoSleep"] == False) & (X["TotalSpend"] == 0)
        ).astype(int)

        return X[self.OUTPUT_FEATURES]


class Log1pTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        skew_vals = X.skew().abs()
        self.skewed_cols_ = skew_vals[skew_vals > 2.0].index.tolist()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.skewed_cols_:
            X[col] = np.log1p(np.clip(X[col], 0, None))

        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)
