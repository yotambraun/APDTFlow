def infer_forecaster(forecaster, new_x, forecast_horizon, device):
    """
    Run inference using the forecaster's predict method.
    """
    preds, pred_logvars = forecaster.predict(new_x, forecast_horizon, device)
    return preds, pred_logvars
