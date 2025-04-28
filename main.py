from Handler import DataHandler, ModelHandler
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    base_path = './Data/Final_Dataset/'
    class_names = torch.load(base_path + "class_map.pt")    
    
    # Initialize data handler and model handler
    logging.info("Initializing data handler and model handler.")
    data_handler = DataHandler(base_path, 4)
    
    logging.info("Loading data for model handler.")
    model_handler = ModelHandler(
        model_name='efficientnet_b0',
        dataloaders=data_handler.load_data(),
        config_path='./config.json',
        )

    logging.info("Starting model training.")
    # Train the model
    model, history = model_handler.train_model(10, use_amp=True)
    
    logging.info("Testing the model.")
    test_results = model_handler.test_model(class_names=class_names)
    logging.info("Test results: %s", test_results)  
    
if __name__ == "__main__":
    main()
    
    