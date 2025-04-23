import norse.torch.functional
import norse.torch.functional.reset
import norse.torch.module
import norse.torch.module.lif_box
import torch
import torch.nn as nn
import norse
import cv2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = norse.torch.LICell(norse.torch.LIParameters(0.1), dt=1)
        self.state = None
        self.submodule = norse.torch.SequentialState(self.conv, 
                                                     self.bn,
                                                     self.activation
                                                     )

    def forward(self, x):
        z, _ = self.submodule(x)
        return z

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPDarknetTiny(nn.Module):
    """
    CSPDarknet for YOLOX-Tiny: Lightweight backbone with fewer blocks and channels.
    """
    def __init__(self):
        super().__init__()
        base_channels = 16  # Reduced base channels for Tiny version
        self.stem = ConvBlock(3, base_channels, kernel_size=3, stride=1, padding=1)

        # Stages with reduced depth for YOLOX-Tiny
        self.stage1 = self._make_stage(base_channels, base_channels * 2, num_blocks=1)
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks=2)
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, num_blocks=2)
        self.stage4 = self._make_stage(base_channels * 8, base_channels * 16, num_blocks=1)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.stem(x)
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            x = stage(x)
            features.append(x)
        return features[-3:]  # Return only the last three feature maps for PANet


class PANetTiny(nn.Module):
    """
    PANet for YOLOX-Tiny: Path Aggregation Network with reduced channels.
    """
    def __init__(self, in_channels_list):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_layers = nn.ModuleList([
            ConvBlock(ch, ch // 2, kernel_size=1, stride=1, padding=0)
            for ch in in_channels_list
        ])
        self.fusion_layers = nn.ModuleList([
            ConvBlock(in_channels_list[i] // 2 + (in_channels_list[i + 1] // 2 if i + 1 < len(in_channels_list) else 0),
                    in_channels_list[i] // 2, kernel_size=3, stride=1, padding=1)
            for i in range(len(in_channels_list))
        ])

    def forward(self, features):
        fpn_outs = [self.reduce_layers[-1](features[-1])]
        # print(f"First fusion output shape: {fpn_outs[-1].shape}")
        for i in range(len(features) - 2, -1, -1):
            upsampled = self.upsample(fpn_outs[-1])
            reduced = self.reduce_layers[i](features[i])
            fpn_outs.append(self.fusion_layers[i](torch.cat([reduced, upsampled], dim=1)))
            # print(f"Fusion output shape at level {i}: {fpn_outs[-1].shape}")
        return fpn_outs[::-1]



class YOLOXHeadTiny(nn.Module):
    def __init__(self, num_classes, in_channels_list, num_anchors=1):
        super().__init__()
        # 1x1 convolution to align the channels of each feature map
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)  # Align all channels to 128
            for in_channels in in_channels_list
        ])
        
        # Classification and regression convolution layers
        self.cls_convs = nn.ModuleList([nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        ) for _ in range(3)])
        self.reg_convs = nn.ModuleList([nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        ) for _ in range(3)])
        
        # Prediction layers (for class, regression, and objectness)
        self.cls_pred = nn.Conv2d(128, num_anchors * num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(128, num_anchors * 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(128, num_anchors * 1, kernel_size=1)

    def forward(self, features):
        outputs = []
        for i, x in enumerate(features):
            # Align the channels using 1x1 convolution
            x = self.align_convs[i](x)

            # Classification and regression convolutions
            cls_x = self.cls_convs[i](x)
            reg_x = self.reg_convs[i](x)

            # Predictions
            cls_pred = nn.functional.max_pool2d(self.cls_pred(cls_x), (3, 4))  # (B, num_anchors * num_classes, H, W) self.cls_pred(cls_x) 
            reg_pred = nn.functional.max_pool2d(self.reg_pred(reg_x), (3, 4))  # (B, num_anchors * 4, H, W) self.reg_pred(reg_x) 
            obj_pred = nn.functional.max_pool2d(self.obj_pred(reg_x), (3, 4))  # (B, num_anchors * 1, H, W) self.obj_pred(reg_x) 
            
            # print(cls_pred.shape)
            # Reshape predictions to have consistent tensor dimensions
            B, _, H, W = reg_pred.shape
            num_anchors = reg_pred.shape[1] // 4

            # Reshape to (B, H, W, num_anchors, dims)
            cls_pred = cls_pred.view(B, num_anchors, -1, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, num_anchors, num_classes)
            reg_pred = reg_pred.view(B, num_anchors, 4, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, num_anchors, 4)
            obj_pred = obj_pred.view(B, num_anchors, 1, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, num_anchors, 1)

            # Concatenate along the last dimension to form (x, y, w, h, obj conf, class conf)
            output = torch.cat([reg_pred, obj_pred, cls_pred], dim=-1)  # (B, H, W, num_anchors, 5 + num_classes)
            outputs.append(output)

        # Concatenate outputs from all feature maps along the spatial dimensions (H and W)
        outputs = torch.cat([o.view(B, -1, o.shape[-1]) for o in outputs], dim=1)  # (B, total_anchors, 5 + num_classes)

        return outputs

class YOLOXTiny(nn.Module):
    """
    YOLOX-Tiny: Full Model combining CSPDarknetTiny, PANetTiny, and YOLOXHeadTiny.
    """
    def __init__(self, num_classes, num_anchors=1):
        super().__init__()
        self.backbone = CSPDarknetTiny()
        self.neck = PANetTiny([64, 128, 256])  # Adjust channels to match backbone
        self.head = YOLOXHeadTiny(num_classes, [32, 64, 128], num_anchors)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(x, 2)
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs


def reset_model_states(model):
    """Recursively reset LIF states in the model."""
    for layer in model.children():
        if isinstance(layer, ConvBlock):
            layer.reset_state()
        elif len(list(layer.children())) > 0:  # If the layer contains other submodules
            reset_model_states(layer)




import torch
import torch.nn as nn
from norse.torch import SequentialState

class YOLOSimple(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super(YOLOSimple, self).__init__()
        # Backbone: CSPDarknet-like layers
        self.backbone = SequentialState(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.6, alpha=10), dt=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.6, alpha=10), dt=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.6, alpha=10), dt=1),
        )
        
        # Neck: Feature pyramid (simplified)
        self.neck = SequentialState(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.6, alpha=10), dt=1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.6, alpha=10), dt=1),
        )
        
        # Head: Predict bounding boxes, confidence, and classes
        self.head = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)
    
    def forward(self, x):
        x, _ = self.backbone(x)
        x, _ = self.neck(x)
        x = self.head(x)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOBase(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, bbox_per_cell=2):
        super(YOLOBase, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.bbox_per_cell = bbox_per_cell
        self.output_size = grid_size * grid_size * (num_classes + bbox_per_cell * 5)
        self.output_size = 13 * 12*12

        # Define the model (simple CNN for demonstration)
        self.conv_layers = SequentialState(
            nn.Conv2d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.4), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.2), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.15), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.125), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.1), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.1), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.1), dt=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=7, stride=2, padding=3, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.1), dt=1),

        )
        self.fc_layers = SequentialState(
            nn.Flatten(),
            nn.Linear(6144, 4096, bias=False),
            norse.torch.LIBoxCell(norse.torch.LIBoxParameters(tau_mem_inv=0.1), dt=1),
            nn.Linear(4096, self.output_size, bias=False),
        )

    def forward(self, x):
        x, _ = self.conv_layers(x)
        x, _ = self.fc_layers(x)
        return x.view(-1, 13, 12*12)
        # return x.view(-1, self.grid_size, self.grid_size, self.num_classes + self.bbox_per_cell * 5)



# ------------------------------------------------ YOLOv8 ------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Convolutional Block
# --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=4, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.activation = norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(0.1, alpha=2, v_th=0.1, reset_method=norse.torch.functional.reset.reset_subtract), dt=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = norse.torch.LIBoxCell(norse.torch.LIBoxParameters(0.1), dt=1)
        self.state = None
        self.submodule = norse.torch.SequentialState(self.conv, 
                                                     self.bn,
                                                     self.activation
                                                     )

    def forward(self, x):
        z, _ = self.submodule(x)
        return z


# ----------------------
# CSP Block (Tiny Version)
# ----------------------
class CSPBlockTiny(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlockTiny, self).__init__()
        # self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=15, stride=1, padding=7)
        self.conv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(out_channels // 2, out_channels // 2, kernel_size=15, stride=1, padding=7)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # return x1
        return self.conv3(torch.cat([x1, x2], dim=1))


# ---------------------
# CSPDarknet Backbone
# ---------------------
class CSPDarknetTiny(nn.Module):
    def __init__(self):
        super(CSPDarknetTiny, self).__init__()
        self.conv1 = ConvBlock(2, 16, kernel_size=15, stride=1, padding=7)  # Adjusted for 2 input channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage1 = CSPBlockTiny(8, 16)
        self.stage2 = CSPBlockTiny(16, 32)
        self.stage3 = CSPBlockTiny(32, 64)
        self.stage4 = CSPBlockTiny(64, 128)
        self.stage5 = CSPBlockTiny(128, 256)
    
    def forward(self, x):
        x1 = self.pool(self.conv1(x))  # Small objects
        x2 = self.pool(self.stage2(x1))  # Medium objects
        x3 = self.pool(self.stage3(x2))  # Large objects
        x4 = self.pool(self.stage4(x3))  # Largest objects
        x5 = self.pool(self.stage5(x4))  # Largestest objects
        return x3, x4, x5  # Return multi-scale features


# -----------------
# PANet Neck
# -----------------
class PANetTiny(nn.Module):
    def __init__(self):
        super(PANetTiny, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x2, x3, x4):
        # Top-down pathway
        x3_upsampled = self.upsample(self.conv1(x4)) + x3
        x2_upsampled = self.upsample(self.conv2(x3_upsampled)) + x2
        
        return x2_upsampled, x3_upsampled, x4



# ----------------------
# YOLOv8 Detection Head
# ----------------------
class DetectionHeadTiny(nn.Module):
    def __init__(self, num_classes):
        super(DetectionHeadTiny, self).__init__()
        self.num_classes = num_classes
        
        # Separate detection heads for each scale
        self.head_small = nn.Conv2d(64, 4 + 1 + num_classes, kernel_size=1, bias=False)  # Small objects
        self.head_medium = nn.Conv2d(128, 4 + 1 + num_classes, kernel_size=1, bias=False)  # Medium objects
        self.head_large = nn.Conv2d(256, 4 + 1 + num_classes, kernel_size=1, bias=False)  # Large objects
    
    def forward(self, x2, x3, x4):
        out_small = self.head_small(x2)  # Small objects
        out_medium = self.head_medium(x3)  # Medium objects
        out_large = self.head_large(x4)  # Large objects
        return [out_small, out_medium, out_large]


# ------------------------
# YOLOv8-Tiny Model
# ------------------------
class YOLOv8Tiny(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Tiny, self).__init__()
        self.backbone = CSPDarknetTiny()
        self.neck = PANetTiny()
        self.head = DetectionHeadTiny(num_classes)
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Backbone
        x2, x3, x4 = self.backbone(x)

        # Neck
        x2, x3, x4 = self.neck(x2, x3, x4)
        
        # Detection Head
        outputs = self.head(x2, x3, x4)
        # outputs = self.head(self.pool(x2), self.pool(x3), self.pool(x4))
        batch = outputs[0].shape[0]
        output = [out.view(batch, 5 + self.num_classes, -1) for out in outputs]
        return torch.cat(output, dim = -1)
    

# ----------------------
# Model Initialization
# ----------------------
if __name__ == "__main__":
    # Example with 20 classes (e.g., Pascal VOC dataset)
    model = YOLOv8Tiny(num_classes=20)
    
    # Dummy input for (batch size=1, channels=2, height=640, width=480)
    dummy_input = torch.randn(1, 2, 640, 480)
    
    # Forward pass
    outputs = model(dummy_input)
    
    # Output shapes for different scales
    for i, output in enumerate(outputs):
        print(f"Output {i} shape:", output.shape)




# # ------------------------------------------------ YOLOv8 ------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --------------------
# # Convolutional Block
# # --------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=9, stride=4, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         # self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = norse.torch.LIFCell(norse.torch.LIFParameters(0.1, alpha=2), dt=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         # self.activation = norse.torch.LICell(norse.torch.LIParameters(0.5), dt=1)
#         self.state = None
#         self.submodule = norse.torch.SequentialState(self.conv, 
#                                                      self.bn,
#                                                      self.activation
#                                                      )

#     def forward(self, x):
#         z, _ = self.submodule(x)
#         return z


# # ----------------------
# # CSP Block (Tiny Version)
# # ----------------------
# class CSPBlockTiny(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(CSPBlockTiny, self).__init__()
#         self.conv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=2, padding=0)
#         self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
#         self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#     def forward(self, x):
#         # x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         return x2


# # ---------------------
# # CSPDarknet Backbone
# # ---------------------
# class CSPDarknetTiny(nn.Module):
#     def __init__(self):
#         super(CSPDarknetTiny, self).__init__()
#         self.conv1 = ConvBlock(2, 16, kernel_size=7, stride=2, padding=3)  # Adjusted for 2 input channels
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stage1 = CSPBlockTiny(8, 16)
#         self.stage2 = CSPBlockTiny(16, 32)
#         self.stage3 = CSPBlockTiny(32, 64)
#         self.stage4 = CSPBlockTiny(64, 128)
#         self.stage5 = CSPBlockTiny(128, 256)
    
#     def forward(self, x):
#         x1 = self.conv1(x)  # Small objects
#         x2 = self.stage2(x1)  # Medium objects
#         x3 = self.stage3(x2)  # Large objects
#         x4 = self.stage4(x3)  # Largest objects
#         x5 = self.stage5(x4)  # Largestest objects
#         return x3, x4, x5  # Return multi-scale features


# # -----------------
# # PANet Neck
# # -----------------
# class PANetTiny(nn.Module):
#     def __init__(self):
#         super(PANetTiny, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.conv1 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
#         self.conv2 = ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, x2, x3, x4):
#         # Top-down pathway
#         x3_upsampled = self.upsample(self.conv1(x4)) + x3
#         x2_upsampled = self.upsample(self.conv2(x3_upsampled)) + x2
        
#         return x2_upsampled, x3_upsampled, x4


# # ----------------------
# # YOLOv8 Detection Head
# # ----------------------
# class DetectionHeadTiny(nn.Module):
#     def __init__(self, num_classes):
#         super(DetectionHeadTiny, self).__init__()
#         self.num_classes = num_classes
        
#         self.head_small_1 = nn.Linear(64 * 80 * 60, 100)  # Small objects
#         self.head_small = nn.Linear(100, (4 + 1 + num_classes) * 80 * 60)  # Small objects
#         self.head_medium_1 = nn.Linear(128 * 30 * 40, 100)  # Medium objects
#         self.head_medium = nn.Linear(100, (4 + 1 + num_classes) * 30 * 40)  # Medium objects
#         self.head_large_1 = nn.Linear(256 * 15 * 20, 100)  # Large objects
#         self.head_large = nn.Linear(100, (4 + 1 + num_classes) * 15 * 20)  # Large objects

#     def forward(self, x2, x3, x4):
#         # Flatten spatial dimensions (H and W) into a single dimension
#         batch_size, c_small, h_small, w_small = x2.shape
#         batch_size, c_medium, h_medium, w_medium = x3.shape
#         batch_size, c_large, h_large, w_large = x4.shape

#         x2_flat = x2.view(batch_size, -1)
#         x3_flat = x3.view(batch_size, -1)
#         x4_flat = x4.view(batch_size, -1)

#         # Process through linear layers
#         out_small_1 = self.head_small_1(x2_flat)
#         out_small = self.head_small(out_small_1)
#         out_medium_1 = self.head_medium_1(x3_flat)
#         out_medium = self.head_medium(out_medium_1)
#         out_large_1 = self.head_large_1(x4_flat)
#         out_large = self.head_large(out_large_1)

#         # Reshape back to the original spatial dimensions with updated channels
#         out_small = out_small.view(batch_size, 4 + 1 + self.num_classes, h_small, w_small)
#         out_medium = out_medium.view(batch_size, 4 + 1 + self.num_classes, h_medium, w_medium)
#         out_large = out_large.view(batch_size, 4 + 1 + self.num_classes, h_large, w_large)

#         return [out_small, out_medium, out_large]

# # ------------------------
# # YOLOv8-Tiny Model
# # ------------------------
# class YOLOv8Tiny(nn.Module):
#     def __init__(self, num_classes):
#         super(YOLOv8Tiny, self).__init__()
#         self.backbone = CSPDarknetTiny()
#         self.neck = PANetTiny()
#         self.head = DetectionHeadTiny(num_classes)
#         self.num_classes = num_classes
    
#     def forward(self, x):
#         # Backbone
#         x2, x3, x4 = self.backbone(x)

#         # Neck
#         x2, x3, x4 = self.neck(x2, x3, x4)
        
#         # Detection Head
#         outputs = self.head(x2, x3, x4)
#         batch = outputs[0].shape[0]
#         output = [out.view(batch, 5 + self.num_classes, -1) for out in outputs]
#         return torch.cat(output, dim = -1)
    
