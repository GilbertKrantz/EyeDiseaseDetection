import torch
from torch import device
import torch.nn as nn
import timm

# Set device
DEVICE = device("cuda" if torch.cuda.is_available() else "cpu")


class EyeDetectionModels(object):
    """
    A class to create and configure various deep learning models for eye detection tasks.
    """

    def __init__(
        self,
        num_classes: int,
        freeze_layers: bool = True,
        device: device = DEVICE,
    ):
        """
        Initialize the EyeDetectionModels class.
        This class provides methods to create and configure various deep learning models for eye detection.
        """
        # Initialize the model creator
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.device = device
        self.models = {
            "mobilenetv4": self.get_model_mobilenetv4,
            "levit": self.get_model_levit,
            "efficientvit": self.get_model_efficientvit,
            "gernet": self.get_model_gernet,
            "regnetx": self.get_model_regnetx,
        }

    # Model architecture functions
    @staticmethod
    def _get_feature_blocks(model: nn.Module) -> nn.ModuleList:
        """
        Utility: locate the main feature blocks container in a timm model.
        Returns a list-like module of blocks.
        """
        for attr in ("features", "blocks", "layers", "stem"):  # common container names
            if hasattr(model, attr):
                return getattr(model, attr)
        # fallback: collect all children except classifier/head
        return list(model.children())[:-1]

    @staticmethod
    def _freeze_except_last_n(blocks: nn.ModuleList, n: int) -> None:
        total = len(blocks)
        for idx, block in enumerate(blocks):
            requires = idx >= total - n
            for p in block.parameters():
                p.requires_grad = requires

    def get_model_mobilenetv4(self) -> nn.Module:
        model = timm.create_model(
            "mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True
        )
        if self.freeze_layers:
            blocks = self._get_feature_blocks(model)
            self._freeze_except_last_n(blocks, 2)
        # replace classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        return model.to(self.device)

    def get_model_levit(self) -> nn.Module:
        model = timm.create_model("levit_128s.fb_dist_in1k", pretrained=True)
        if self.freeze_layers:
            blocks = self._get_feature_blocks(model)
            self._freeze_except_last_n(blocks, 2)
        # Attempt to extract in_features from model.head or classifier
        in_features = 384
        model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        model.head_dist = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        return model.to(self.device)

    def get_model_efficientvit(self) -> nn.Module:
        model = timm.create_model("efficientvit_m1.r224_in1k", pretrained=True)
        if self.freeze_layers:
            blocks = self._get_feature_blocks(model)
            self._freeze_except_last_n(blocks, 2)
        # handle different head naming
        in_features = 192
        model.head.linear = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        return model.to(self.device)

    def get_model_gernet(self) -> nn.Module:
        """
        Load and configure a GENet (General and Efficient Network) model with customizable classifier.

        Returns:
            Configured GENet model
        """
        model = timm.create_model("gernet_s.idstcv_in1k", pretrained=True)

        if self.freeze_layers:
            # For GENet, we need to specifically handle its structure
            # It typically has a 'stem' and 'stages' structure
            if hasattr(model, "stem") and hasattr(model, "stages"):
                # Freeze stem completely
                for param in model.stem.parameters():
                    param.requires_grad = False

                # Freeze all stages except the last two
                stages = list(model.stages.children())
                total_stages = len(stages)
                for i, stage in enumerate(stages):
                    requires_grad = i >= total_stages - 2
                    for param in stage.parameters():
                        param.requires_grad = requires_grad
            else:
                # Fallback to generic approach
                blocks = self._get_feature_blocks(model)
                self._freeze_except_last_n(blocks, 2)

        # Replace classifier
        in_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        return model.to(self.device)

    def get_model_regnetx(self) -> nn.Module:
        """
        Load and configure a RegNetX model with customizable classifier.

        Returns:
            Configured RegNetX model
        """
        model = timm.create_model("regnetx_008.tv2_in1k", pretrained=True)

        if self.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

            # RegNetX typically has 'stem' + 'trunk' structure in timm
            if hasattr(model, "trunk"):
                # Unfreeze final stages of the trunk
                trunk_blocks = list(model.trunk.children())
                # Unfreeze approximately last 25% of trunk blocks
                unfreeze_from = max(0, int(len(trunk_blocks) * 0.75))
                for i in range(unfreeze_from, len(trunk_blocks)):
                    for param in trunk_blocks[i].parameters():
                        param.requires_grad = True

            # Always unfreeze the classifier/head for fine-tuning
            for param in model.head.parameters():
                param.requires_grad = True

        # Replace classifier
        in_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes),
        )
        return model.to(self.device)
