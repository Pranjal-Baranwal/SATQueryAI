import base64
import io

import modal
from fastapi import Request
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qingyun/RSCoVLM-7B-2512"


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
        "pillow",
        "fastapi",
        "qwen-vl-utils",
    )
)


# ============================================================
# MODAL APP
# ============================================================

app = modal.App("satqueryai-base-models")


# ============================================================
# BASE MODEL SERVICE
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    scaledown_window=300,
)
class BaseVLM:

    @modal.enter()
    def load_model(self):

        import torch

        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print("=" * 60)
        print("LOADING BASE RSCoVLM-7B")
        print("=" * 60)
        print(f"Model: {MODEL_ID}")
        print("GPU: Modal A10G")
        print("Adapter: NONE")
        print("=" * 60)

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        )

        self.model.eval()

        print("=" * 60)
        print("BASE MODEL READY")
        print("=" * 60)


    # ========================================================
    # IMAGE DECODER
    # ========================================================

    @staticmethod
    def decode_image(encoded):

        image_bytes = base64.b64decode(encoded)

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image


    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        images,
        messages,
        max_new_tokens=256,
    ):

        import torch

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to("cuda")
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_token_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = generated_ids[
            :,
            input_token_length:
        ]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return answer.strip()


    # ========================================================
    # SIA
    # ========================================================

    @modal.fastapi_endpoint(method="POST")
    async def sia(self, request: Request):

        data = await request.json()

        image = self.decode_image(
            data["image"]
        )

        query = data.get(
            "query",
            "",
        )

        prompt = data.get(
            "prompt",
            query,
        )

        system_prompt = data.get(
            "system_prompt",
            "",
        )

        messages = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        )

        answer = self.generate(
            images=[image],
            messages=messages,
            max_new_tokens=data.get(
                "max_new_tokens",
                256,
            ),
        )

        return {
            "success": True,
            "model": MODEL_ID,
            "adapter": None,
            "device": "Modal A10G",
            "answer": answer,
        }


    # ========================================================
    # BTA
    # ========================================================

    @modal.fastapi_endpoint(method="POST")
    async def bta(self, request: Request):

        data = await request.json()

        image_t1 = self.decode_image(
            data["image_t1"]
        )

        image_t2 = self.decode_image(
            data["image_t2"]
        )

        query = data.get(
            "query",
            "",
        )

        prompt = data.get(
            "prompt",
            query,
        )

        system_prompt = data.get(
            "system_prompt",
            "",
        )

        messages = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        )

        answer = self.generate(
            images=[
                image_t1,
                image_t2,
            ],
            messages=messages,
            max_new_tokens=data.get(
                "max_new_tokens",
                128,
            ),
        )

        return {
            "success": True,
            "model": MODEL_ID,
            "adapter": None,
            "device": "Modal A10G",
            "answer": answer,
        }