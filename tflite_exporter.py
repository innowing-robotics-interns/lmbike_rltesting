import torch as th
from torch import nn
from stable_baselines3 import PPO
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# ---------------------------------------------------------
# Concept 1: Extracting the Deterministic Actor
# ---------------------------------------------------------
class OnnxablePolicy(nn.Module):
    """
    A wrapper class that strips away the PPO Critic network and the 
    random probability distribution, leaving ONLY the deterministic Actor.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # 1. Extract features (for our simple MLP, this just passes the data through)
        features = self.policy.extract_features(observation)
        # 2. Pass the data through the Actor's hidden layers
        latent_pi, _ = self.policy.mlp_extractor(features)
        # 3. Output the exact optimal action (the mean of the distribution)
        return self.policy.action_net(latent_pi)

def main():
    print("--- Phase 3: Exporting to TensorFlow Lite ---")
    
    # 1. Load the trained PyTorch Model from Phase 2
    print("Loading PyTorch model...")
    model = PPO.load("balancing_robot_model.zip")
    onnx_policy = OnnxablePolicy(model.policy)

    # 2. Export to ONNX
    print("Exporting to ONNX format...")
    # We create a "dummy" input shaped [1 batch, 4 state variables] to show 
    # the exporter what the data flowing through the network looks like.
    dummy_input = th.randn(1, 4) 
    th.onnx.export(
        onnx_policy,
        dummy_input,
        "balancing_robot_model.onnx",
        opset_version=11,
        input_names=["observation"],
        output_names=["action"]
    )

    # 3. Convert ONNX to TensorFlow SavedModel
    print("Converting ONNX to TensorFlow...")
    onnx_model = onnx.load("balancing_robot_model.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("tf_model_dir")

    # 4. Convert TensorFlow to TFLite with Quantization
    print("Converting to TFLite and applying float16 Quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model("tf_model_dir")
    
    # Enable optimizations to shrink the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Tell the converter to use 16-bit floats instead of 32-bit
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    # Save the final embedded-ready file
    tflite_filename = "balancing_robot_model.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
        
    file_size_kb = os.path.getsize(tflite_filename) / 1024
    print(f"\nSuccess! Exported {tflite_filename}")
    print(f"Final Model Size: {file_size_kb:.2f} KB")
    print("This file is now ready to be loaded onto the Pixhawk C++ environment!")

if __name__ == "__main__":
    main()