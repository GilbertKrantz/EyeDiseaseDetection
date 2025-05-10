from Handler import DataHandler, ModelHandler

from util.Validator import validate_dataset

import torch
import logging
import gc

import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_SIZE = 8

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Train and test models.")
parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for training and testing.",
)
parser.add_argument(
    "--validate_dataset",
    action="store_true",
    help="Validate the dataset during training.",
)


def main():
    base_path = "./Data/Min_Tensor_Dataset/"
    class_names = torch.load(base_path + "class_map.pt")

    model_names = [
        "mobilenetv4_conv_small.e1200_r224_in1k",
        "tinynet_e.in1k",
        "efficientnet_b0",
    ]

    # Initialize data handler and model handler
    logging.info("Initializing data handler and model handler.")
    data_handler = DataHandler(
        base_path, BATCH_SIZE, use_weighted_sampler=False
    )  # Re-enable data handler initialization

    if parser.parse_args().validate_dataset:
        try:
            # Load data and validate
            logging.info("Loading data.")
            data_loaders = data_handler.load_data()
            logging.info("Data loaded successfully.")
            logging.info("Validating dataset.")
            logging.info("Validating training data.")
            validate_dataset(
                data_loaders["train"],
                class_names,
                num_batches=BATCH_SIZE,
                display_samples=False,
            )

            logging.info("Validating validation data.")
            validate_dataset(
                data_loaders["val"],
                class_names,
                num_batches=BATCH_SIZE,
                display_samples=False,
            )
            logging.info("Dataset validation completed.")

            # Clean up data handler
            del data_loaders

            # start garbage collection
            gc.collect()

            logging.info("Garbage collection completed.")

        except Exception as e:
            logging.error(f"Error occurred while loading or validating data: {e}")
            return

    for model_name in model_names:
        try:
            save_dir = f"./model_outputs/{model_name}/"
            logging.info(f"Training model {model_name}.")
            model_handler = ModelHandler(
                model_name=model_name,
                dataloaders=data_handler.load_data(),
                config_path="./config.json",
            )
            logging.info("Starting model training.")

            model, history = model_handler.train_model(
                100,
                use_amp=True,
                save_dir=save_dir,
                # class_weights=data_handler.get_loss_class_weights(),
            )
            logging.info("Testing the model.")
            test_results = model_handler.test_model(
                class_names=class_names, save_dir=save_dir
            )
            logging.info("Model training completed.")
            logging.info("Test results: %s", test_results)
        except Exception as e:
            logging.error(f"Error occurred while training {model_name}: {e}")
            continue

        # Clean VRAM
        del model
        del history
        del test_results

        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Cleared VRAM.")

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
