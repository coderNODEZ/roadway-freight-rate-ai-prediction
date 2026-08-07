import pandas as pd
from fastai.tabular.all import *
import torch
from torch.nn import SmoothL1Loss
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


import FeatureEngineering.feature_engr_constants as fec
import FeatureEngineering.MissingColumnsPrediction.missing_columns_prediction_history as mcph
import Utils.visualization as vis



def predict_market_index_column_fastai():
    df = pd.read_csv("FeatureEngineering/MissingColumnsPrediction/missing_columns_feature_engr_output.csv")

    target_column = fec.missing_columns_target

    if isinstance(target_column, list):
        if len(target_column) != 1:
            raise ValueError(
                "Exactly one target column is required."
            )

        target_column = target_column[0]

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    observed_target_df = df.loc[
        df[target_column].notna()
    ].reset_index(drop=True)

    if len(observed_target_df) <= 8_000:
        raise ValueError(
            "At least 8,000 rows with known market_index "
            "values are required."
        )    


    testing_indices = observed_target_df.sample(
        n=8_000,
        random_state=42,
    ).index

    df_testing = observed_target_df.loc[
        testing_indices
    ].reset_index(drop=True)

    df_training = observed_target_df.drop(
        index=testing_indices,
    ).reset_index(drop=True)

    prohibited_features = {
        target_column,
        "quote_signal",
        "posted_rate",
    }

    categorical_columns = list(
        fec.missing_columns_categorical_columns
    )

    continuous_columns = list(
        fec.missing_columns_continuous_columns
    )

    configured_features = set(
        categorical_columns
    ) | set(
        continuous_columns
    )

    leaked_features = sorted(
        prohibited_features & configured_features
    )

    if leaked_features:
        raise ValueError(
            f"Target/leakage columns are configured as features: "
            f"{leaked_features}"
        )    

    procs = [FillMissing, Categorify, Normalize]

    splits = RandomSplitter(
        valid_pct=0.2,
        seed=42,
    )(range_of(df_training))   

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch."
        )

    device = torch.device("cuda:0")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")    

    df_training = df_training.copy(deep=True)
    df_testing = df_testing.copy(deep=True)

    columns_used = (
        categorical_columns
        + continuous_columns
        + [target_column]
    )

    for column in columns_used:
        df_training[column] = (
            df_training[column]
            .to_numpy(copy=True)
        )

        df_testing[column] = (
            df_testing[column]
            .to_numpy(copy=True)
        )     


    dls = TabularDataLoaders.from_df(
        df_training,
        procs=procs,
        cat_names=categorical_columns,
        cont_names=continuous_columns,
        y_names=target_column,
        y_block=RegressionBlock(),
        splits=splits,
        bs=256,
        device=device,
    )

    learn = tabular_learner(
        dls,
        metrics=mae,
        loss_func=SmoothL1Loss(),
    )    

    learn.model.to(device)

    print(
        "Model device:",
        next(learn.model.parameters()).device,
    )

    print(
        "DataLoader device:",
        learn.dls.device,
    )    



    learn.fit_one_cycle(40, 1e-3)

    print("#############################################")
    print("# FastAI tabular learner is predicting test data")

    y_test = pd.to_numeric(
        df_testing[target_column],
        errors="coerce",
    ).to_numpy()

    x_test = df_testing.drop(
        columns=[
            target_column,
            "quote_signal",
            "posted_rate",
        ],
        errors="ignore",
    ).copy()

    test_dl = learn.dls.test_dl(
        x_test,
        with_labels=False,
    )

    predictions, _ = learn.get_preds(
        dl=test_dl,
    )

    predicted_market_index = (
        predictions.squeeze()
        .cpu()
        .numpy()
    )

    df_testing["predicted_market_index"] = (
        predicted_market_index
    )

    test_mae = mean_absolute_error(
        y_test,
        predicted_market_index,
    )

    test_rmse = mean_squared_error(
        y_test,
        predicted_market_index,
    ) ** 0.5

    test_r2 = r2_score(
        y_test,
        predicted_market_index,
    )

    print("# FastAI prediction is complete")
    print(f"Test MAE:  {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test R²:   {test_r2:.6f}")
    print("#############################################")

    print("Evaluating overfit/generalization")
    def evaluate_partition(
        partition: pd.DataFrame,
    ) -> dict[str, float]:
        y_actual = pd.to_numeric(
            partition[target_column],
            errors="coerce",
        ).to_numpy()

        features = partition.drop(
            columns=[
                target_column,
                "quote_signal",
                "posted_rate",
            ],
            errors="ignore",
        ).copy()

        partition_dl = learn.dls.test_dl(
            features,
            with_labels=False,
        )

        partition_predictions, _ = learn.get_preds(
            dl=partition_dl,
        )

        y_predicted = (
            partition_predictions.squeeze()
            .cpu()
            .numpy()
        )

        return {
            "mae": mean_absolute_error(
                y_actual,
                y_predicted,
            ),
            "rmse": mean_squared_error(
                y_actual,
                y_predicted,
            ) ** 0.5,
            "r2": r2_score(
                y_actual,
                y_predicted,
            ),
        }

    train_indices, validation_indices = splits

    training_metrics = evaluate_partition(
        df_training.iloc[train_indices]
    )

    validation_metrics = evaluate_partition(
        df_training.iloc[validation_indices]
    )

    testing_metrics = evaluate_partition(
        df_testing
    )   


    validation_mae_gap = (
        validation_metrics["mae"]
        - training_metrics["mae"]
    )

    test_mae_gap = (
        testing_metrics["mae"]
        - training_metrics["mae"]
    )

    validation_rmse_gap = (
        validation_metrics["rmse"]
        - training_metrics["rmse"]
    )

    test_rmse_gap = (
        testing_metrics["rmse"]
        - training_metrics["rmse"]
    )

    test_r2_gap = (
        training_metrics["r2"]
        - testing_metrics["r2"]
    )         

    print("#############################################")
    print("# FastAI regression performance")
    print(
        f"Training:   "
        f"MAE={training_metrics['mae']:.6f}, "
        f"RMSE={training_metrics['rmse']:.6f}, "
        f"R²={training_metrics['r2']:.6f}"
    )
    print(
        f"Validation: "
        f"MAE={validation_metrics['mae']:.6f}, "
        f"RMSE={validation_metrics['rmse']:.6f}, "
        f"R²={validation_metrics['r2']:.6f}"
    )
    print(
        f"Testing:    "
        f"MAE={testing_metrics['mae']:.6f}, "
        f"RMSE={testing_metrics['rmse']:.6f}, "
        f"R²={testing_metrics['r2']:.6f}"
    )

    print("# Generalization gaps")
    print(f"Validation MAE gap:  {validation_mae_gap:.6f}")
    print(f"Test MAE gap:        {test_mae_gap:.6f}")
    print(f"Validation RMSE gap: {validation_rmse_gap:.6f}")
    print(f"Test RMSE gap:       {test_rmse_gap:.6f}")
    print(f"Test R² gap:         {test_r2_gap:.6f}")
    print("#############################################")    
    print("#############################################")   

    vis.plot_regression_performance(
        model_name="FastAI",
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        testing_metrics=testing_metrics,
    )      

    mcph.record_prediction_history(
        model="FastAI_TabularLearner",
        training_mae=training_metrics["mae"],
        validation_mae=validation_metrics["mae"],
        testing_mae=testing_metrics["mae"],
        training_rmse=training_metrics["rmse"],
        validation_rmse=validation_metrics["rmse"],
        testing_rmse=testing_metrics["rmse"],
        training_r2=training_metrics["r2"],
        validation_r2=validation_metrics["r2"],
        testing_r2=testing_metrics["r2"],
    )     



    # return df_testing, learn    

    # use for the actual df that is missing the column
    # print("# Predicting rows with missing market_index")

    # if not missing_target_df.empty:
    #     missing_x = missing_target_df.drop(
    #         columns=[
    #             target_column,
    #             "quote_signal",
    #             "posted_rate",
    #         ],
    #         errors="ignore",
    #     ).copy()

    #     missing_dl = learn.dls.test_dl(
    #         missing_x,
    #         with_labels=False,
    #     )

    #     missing_predictions, _ = learn.get_preds(
    #         dl=missing_dl,
    #     )

    #     missing_target_df[target_column] = (
    #         missing_predictions.squeeze()
    #         .cpu()
    #         .numpy()
    #     )

    # print(
    #     f"Missing market_index values predicted: "
    #     f"{len(missing_target_df):,}"
    # )    



  
