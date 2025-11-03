from vllm import LLM, SamplingParams
import sys
import os

# os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'

# 1. Path to your model
# model_path = "facebook/opt-125m"  # Small test model (~250MB)
# model_path = "Qwen/Qwen2.5-7B-Instruct"  # Original (requires ~40GB GPU memory)
model_path = "Qwen/Qwen3-4B-Instruct-2507"

# 2. Initialize vLLM with your model
try:
    print(f"Initializing vLLM with model: {model_path}")
    print("This may take a few minutes on first run (downloading model)...")
    
    # Add configuration for better compatibility
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,  # Increased from 0.8
        max_model_len=27152,  # Set to fit available KV cache memory
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )
    
    print("✓ vLLM initialized successfully!")
    
    
    
    
except RuntimeError as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Model not found or insufficient GPU memory")

    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
    print("Run 'python diagnose_vllm.py' for more details")
    sys.exit(1)

# 3. Choose sampling parameters (temperature, top_p)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

# 4. Example prompt(s)
prompts = ["Hello, my name is", "What is generative AI?"]

# 5. Generate output
try:
    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("\n" + "=" * 70)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated Text: {output.outputs[0].text!r}")
        print("-" * 70)
    print("\n✓ Generation completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during generation: {e}")
    sys.exit(1)
