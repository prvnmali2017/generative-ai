from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def transformer_fn():

    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "origin_encoder",
                            OneHotEncoder(
                                categories="auto", sparse=False, handle_unknown="ignore"
                            ),
                            ["origin"],
                        ),
                        (
                            "dest_encoder",
                            OneHotEncoder(
                                categories="auto", sparse=False, handle_unknown="ignore"
                            ),
                            ["dest"],
                        ),
                        (
                            "carrier_encoder",
                            OneHotEncoder(
                                categories="auto", sparse=False, handle_unknown="ignore"
                            ),
                            ["carrier"],
                        ),
                    ]
                ),
            ),
        ]
    )