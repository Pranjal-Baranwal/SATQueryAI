import base64
import io

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "satqueryai-cross-modal"

VOLUME_NAME = "satqueryai-extracted-v2"

ADAPTER_PATH = "/data/SATQuery_FINAL_RSCoVLM_output/final_adapter"

BASE_MODEL = "Qingyun/RSCoVLM-7B-2512"


# ============================================================
# MODAL IMAGE
# ============================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "pillow",
        "fastapi",
        "qwen-vl-utils",
    )
)


# ============================================================
# MODAL APP
# ============================================================

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME)


# ============================================================
# VLM SERVER
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=1800,
    scaledown_window=300,
)
class CrossModalVLM:

    @modal.enter()
    def load_model(self):
        print("=" * 60)
        print("Loading SATQueryAI fine-tuned VLM")
        print("=" * 60)

        import torch

        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
        )

        from peft import PeftModel

        print(f"Base model: {BASE_MODEL}")
        print(f"Adapter: {ADAPTER_PATH}")

        # ----------------------------------------------------
        # 4-bit configuration
        # ----------------------------------------------------

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        # ----------------------------------------------------
        # Base model
        # ----------------------------------------------------

        print("Loading base model...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # ----------------------------------------------------
        # LoRA adapter
        # ----------------------------------------------------

        print("Loading fine-tuned LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            self.model,
            ADAPTER_PATH,
            is_trainable=False,
        )

        self.model.eval()

        print("=" * 60)
        print("MODEL READY")
        print("=" * 60)


    # ========================================================
    # INFERENCE ENDPOINT
    # ========================================================

    @modal.fastapi_endpoint(method="POST")
    def predict(self, request: dict):

        import torch
        from PIL import Image

        # ----------------------------------------------------
        # Validate request
        # ----------------------------------------------------

        if "sar_image" not in request:
            return {
                "success": False,
                "error": "Missing sar_image",
            }

        if "optical_image" not in request:
            return {
                "success": False,
                "error": "Missing optical_image",
            }

        # ----------------------------------------------------
        # Decode images
        # ----------------------------------------------------

        try:
            sar_bytes = base64.b64decode(request["sar_image"])
            optical_bytes = base64.b64decode(request["optical_image"])

            sar_image = Image.open(
                io.BytesIO(sar_bytes)
            ).convert("RGB")

            optical_image = Image.open(
                io.BytesIO(optical_bytes)
            ).convert("RGB")

        except Exception as e:
            return {
                "success": False,
                "error": f"Image decoding failed: {str(e)}",
            }

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        user_question = request.get(
            "query",
            "Identify the land-cover classes present in this scene.",
        )

        prompt = f"""
Analyze the Sentinel-1 SAR and Sentinel-2
optical imagery together.

Identify the land-cover classes present
in this scene.

Use both modalities together when interpreting
the scene.

USER QUESTION:
{user_question}
""".strip()

        # ----------------------------------------------------
        # Qwen2.5-VL multimodal message
        # ----------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # ----------------------------------------------------
        # Prepare model input
        #
        # IMPORTANT:
        # Image order MUST match training:
        #
        #   1. SAR
        #   2. Optical
        # ----------------------------------------------------

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[
                sar_image,
                optical_image,
            ],
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to GPU
        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
            )

        # ----------------------------------------------------
        # Remove input tokens
        # ----------------------------------------------------

        input_length = inputs["input_ids"].shape[1]

        generated_ids = generated_ids[
            :, input_length:
        ]

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return {
            "success": True,
            "answer": answer,
        }


# ============================================================
# LOCAL TEST ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():

    print()
    print("=" * 60)
    print("SATQueryAI Modal inference server")
    print("=" * 60)
    print()
    print("This file exposes the fine-tuned Cross-Modal VLM.")
    print()