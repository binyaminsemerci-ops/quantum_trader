#!/usr/bin/env python3
"""Inspect structure of N-HiTS and PatchTST .pth files"""
import torch
import json
import sys

def inspect_model(model_path, meta_path, name):
    print(f"\n{'='*60}")
    print(f"  {name} MODEL INSPECTION")
    print(f"{'='*60}")
    
    # Load model file
    try:
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"\n✅ Loaded: {model_path}")
        print(f"Type: {type(loaded)}")
        print(f"Is dict: {isinstance(loaded, dict)}")
        
        if isinstance(loaded, dict):
            print(f"\n📋 Dictionary Keys: {list(loaded.keys())}")
            
            # Check for common checkpoint patterns
            if 'model_state_dict' in loaded:
                print("   ⚠️  Found 'model_state_dict' - this is a checkpoint format")
                state_dict = loaded['model_state_dict']
            elif 'state_dict' in loaded:
                print("   ⚠️  Found 'state_dict' - this is a checkpoint format")
                state_dict = loaded['state_dict']
            else:
                # The dict itself might be the state_dict
                print("   📦 Dict appears to be the state_dict directly (OrderedDict)")
                state_dict = loaded
            
            # Show layer structure
            print(f"\n🧱 State Dict Structure:")
            print(f"   Total parameters: {len(state_dict)}")
            print(f"   First 10 layers:")
            for i, (key, tensor) in enumerate(list(state_dict.items())[:10]):
                print(f"      {i+1}. {key}: shape={tensor.shape}")
            if len(state_dict) > 10:
                print(f"      ... and {len(state_dict) - 10} more layers")
                
        else:
            print(f"\n⚠️  Not a dict - Type: {type(loaded).__name__}")
            print(f"Has .state_dict(): {hasattr(loaded, 'state_dict')}")
            print(f"Has .predict(): {hasattr(loaded, 'predict')}")
            print(f"Has .forward(): {hasattr(loaded, 'forward')}")
            
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return
    
    # Load metadata
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n📄 Metadata from {meta_path}:")
        for key, value in meta.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"   {key}: [{value[0]}, {value[1]}, ... {len(value)} items total]")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"\n⚠️  Could not load metadata: {e}")

if __name__ == "__main__":
    # N-HiTS
    inspect_model(
        "models/nhits_v20260205_231109_v5.pth",
        "models/nhits_v20260205_231109_v5_meta.json",
        "N-HiTS"
    )
    
    # PatchTST
    inspect_model(
        "models/patchtst_v20260205_231204_v5.pth",
        "models/patchtst_v20260205_231204_v5_meta.json",
        "PatchTST"
    )
    
    print(f"\n{'='*60}")
    print("  CONCLUSION")
    print(f"{'='*60}")
    print("\nTo use these models, we need to:")
    print("1. Define the model architecture (neural network class)")
    print("2. Instantiate the model: model = ModelClass()")
    print("3. Load the state_dict: model.load_state_dict(loaded)")
    print("4. Set to eval mode: model.eval()")
    print("\n")
