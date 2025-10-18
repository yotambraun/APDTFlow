def train_forecaster(forecaster, train_loader, num_epochs, learning_rate, device):
    """
    Train the given forecaster using its train_model method.
    """
    forecaster.train_model(train_loader, num_epochs, learning_rate, device)
    return forecaster
