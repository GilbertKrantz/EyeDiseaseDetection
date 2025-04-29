from Handler import DataHandler, ModelHandler
import torch
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    base_path = "./Data/Final_Dataset/"
    class_names = torch.load(base_path + "class_map.pt")

    model_names = [
        "efficientnet_b0",
        "mobilenetv3_small_100",
        "resnet18",
        "mobilevit_xxs",
        "convnext_tiny",
    ]

    # Initialize data handler and model handler
    logging.info("Initializing data handler and model handler.")
    data_handler = DataHandler(base_path, 32)  # Re-enable data handler initialization

    for model_name in model_names:
        save_dir = f"./model_outputs/{model_name}/"
        logging.info(f"Training model {model_name}.")
        model_handler = ModelHandler(
            model_name=model_name,
            dataloaders=data_handler.load_data(),
            config_path="./config.json",
        )
        logging.info("Starting model training.")

        model, history = model_handler.train_model(100, use_amp=True, save_dir=save_dir)
        logging.info("Testing the model.")
        test_results = model_handler.test_model(
            class_names=class_names, save_dir=save_dir
        )
        logging.info("Model training completed.")
        logging.info("Test results: %s", test_results)

        # Clean VRAM
        del model
        del history
        del test_results

        torch.cuda.empty_cache()

    # logging.info("Loading data for model handler.")
    # model_handler = ModelHandler(
    #     model_name='efficientnet_b0',
    #     dataloaders=data_handler.load_data(),
    #     config_path='./config.json',
    #     )

    # logging.info("Starting model training.")
    # # Train the model
    # model, history = model_handler.train_model(10, use_amp=True)

    # logging.info("Testing the model.")
    # test_results = model_handler.test_model(class_names=class_names)
    # logging.info("Test results: %s", test_results)


if __name__ == "__main__":
    main()
