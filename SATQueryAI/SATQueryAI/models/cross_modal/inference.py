

import os
import time
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from models.cross_modal.fusion import (
    CrossModalFusion,
)


load_dotenv()



MODEL_ID = os.getenv(
    "SIA_MODEL_ID",
    "Qingyun/RSCoVLM-7B-2512"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 96

MAX_IMAGE_SIZE = 1024



QUANTIZATION_CONFIG = None

if DEVICE == "cuda":

    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class CrossModalInference:

    def __init__(
        self,
        model_id=MODEL_ID,
        device=DEVICE
    ):

        self.model_id = model_id
        self.device = device

        self.model = None
        self.processor = None

        self.fusion_engine = (
            CrossModalFusion()
        )

        self.loaded = False


    def load_model(self):

        if self.loaded:
            return

        print(
            f"\nLoading Cross-Modal model: "
            f"{self.model_id}"
        )

        print(
            f"Device: {self.device}"
        )


        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
        )


        if self.device == "cuda":

            print(
                "Using 4-bit quantization."
            )

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                f"VRAM: "
                f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
            )

            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.model_id,
                    quantization_config=(
                        QUANTIZATION_CONFIG
                    ),
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=True
                )
            )

        else:

            print(
                "WARNING: CUDA unavailable."
            )

            print(
                "Loading Cross-Modal model on CPU."
            )

            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.model_id,
                    dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            )

        self.model.eval()

        self.loaded = True

        print(
            "\nCross-Modal model loaded successfully."
        )


    @staticmethod
    def array_to_image(
        data
    ):
        """
        Convert normalized CHW array into RGB PIL image.
        """

        data = np.asarray(
            data
        ).astype(
            np.float32
        )


        if data.shape[0] == 1:

            band = data[0]

            rgb = np.stack(
                [
                    band,
                    band,
                    band
                ],
                axis=-1
            )


        elif data.shape[0] == 2:

            rgb = np.stack(
                [
                    data[0],
                    data[1],
                    data[1]
                ],
                axis=-1
            )


        else:

            rgb = np.transpose(
                data[:3],
                (1, 2, 0)
            )

        rgb = np.clip(
            rgb * 255.0,
            0,
            255
        ).astype(
            np.uint8
        )

        image = Image.fromarray(
            rgb,
            mode="RGB"
        )

        image.thumbnail(
            (
                MAX_IMAGE_SIZE,
                MAX_IMAGE_SIZE
            ),
            Image.Resampling.LANCZOS
        )

        return image


    @staticmethod
    def create_fused_visualization(
        optical,
        sar
    ):
        """
        Create a visual Optical + SAR representation.
        """

        optical = np.asarray(
            optical
        ).astype(
            np.float32
        )

        sar = np.asarray(
            sar
        ).astype(
            np.float32
        )


        if optical.shape[0] >= 3:

            optical_rgb = np.transpose(
                optical[:3],
                (1, 2, 0)
            )

        else:

            optical_rgb = np.repeat(
                optical[0:1],
                3,
                axis=0
            )

            optical_rgb = np.transpose(
                optical_rgb,
                (1, 2, 0)
            )


        sar_band = sar[0]

        sar_rgb = np.stack(
            [
                sar_band,
                sar_band,
                sar_band
            ],
            axis=-1
        )


        height = min(
            optical_rgb.shape[0],
            sar_rgb.shape[0]
        )

        width = min(
            optical_rgb.shape[1],
            sar_rgb.shape[1]
        )

        optical_rgb = (
            optical_rgb[
                :height,
                :width
            ]
        )

        sar_rgb = (
            sar_rgb[
                :height,
                :width
            ]
        )


        fused = (
            0.5 * optical_rgb
            +
            0.5 * sar_rgb
        )

        fused = np.clip(
            fused * 255.0,
            0,
            255
        ).astype(
            np.uint8
        )

        image = Image.fromarray(
            fused,
            mode="RGB"
        )

        image.thumbnail(
            (
                MAX_IMAGE_SIZE,
                MAX_IMAGE_SIZE
            ),
            Image.Resampling.LANCZOS
        )

        return image


    @staticmethod
    def create_prompt(
        query
    ):

        return f"""
You are an expert remote-sensing analyst.

You are given three visual inputs describing the same
geographic region:

IMAGE 1:
Optical satellite imagery.

IMAGE 2:
Synthetic Aperture Radar (SAR) imagery.

IMAGE 3:
A fused Optical + SAR representation.

Analyze all three inputs together.

Optical imagery can provide information about:
- Vegetation
- Buildings
- Roads
- Water bodies
- Land cover
- Visible surface characteristics

SAR imagery can provide information about:
- Surface structure
- Built-up areas
- Surface roughness
- Radar backscatter patterns
- Structural features
- Features that may be obscured in optical imagery

Use agreement between Optical and SAR observations as
stronger supporting evidence.

Do not claim that a feature exists solely because of weak
evidence in one modality.

Clearly distinguish between:
- Direct observations
- Cross-modal interpretation
- Uncertain conclusions

Do not invent coordinates, locations, dates, or measurements.

Answer the user's question concisely.

USER QUESTION:
{query}
"""

    def prepare_inputs(
        self,
        optical_path,
        sar_path
    ):

        print(
            "\n========== CROSS-MODAL FUSION =========="
        )

        fusion_result = (
            self.fusion_engine.process(
                optical_path=optical_path,
                sar_path=sar_path
            )
        )

        optical = (
            fusion_result["optical"]
        )

        sar = (
            fusion_result["sar"]
        )

        optical_image = (
            self.array_to_image(
                optical
            )
        )


        sar_image = (
            self.array_to_image(
                sar
            )
        )


        fused_image = (
            self.create_fused_visualization(
                optical,
                sar
            )
        )


        fused_path = (
            Path("data/output")
            /
            "cross_modal_vlm_fusion.png"
        )

        fused_image.save(
            fused_path
        )

        print(
            f"Fused VLM image saved to: "
            f"{fused_path}"
        )

        return (
            optical_image,
            sar_image,
            fused_image,
            fusion_result
        )


    def analyze(
        self,
        optical_path,
        sar_path,
        query,
        max_new_tokens=MAX_NEW_TOKENS
    ):

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        total_start = time.time()

        model_start = time.time()

        self.load_model()

        model_time = (
            time.time()
            -
            model_start
        )


        fusion_start = time.time()

        (
            optical_image,
            sar_image,
            fused_image,
            fusion_result
        ) = self.prepare_inputs(
            optical_path,
            sar_path
        )

        fusion_time = (
            time.time()
            -
            fusion_start
        )


        prompt = self.create_prompt(
            query
        )


        messages = [
            {
                "role": "user",
                "content": [

                    {
                        "type": "text",
                        "text": (
                            "IMAGE 1 - OPTICAL"
                        )
                    },

                    {
                        "type": "image"
                    },

                    {
                        "type": "text",
                        "text": (
                            "IMAGE 2 - SAR"
                        )
                    },

                    {
                        "type": "image"
                    },

                    {
                        "type": "text",
                        "text": (
                            "IMAGE 3 - OPTICAL + SAR "
                            "FUSION"
                        )
                    },

                    {
                        "type": "image"
                    },

                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]


        processing_start = time.time()

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = (
            self.processor(
                text=[text],
                images=[
                    optical_image,
                    sar_image,
                    fused_image
                ],
                padding=True,
                return_tensors="pt"
            )
        )

        if self.device == "cuda":

            inputs = {
                key: value.to("cuda")
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

        processing_time = (
            time.time()
            -
            processing_start
        )


        print(
            "\nGenerating Cross-Modal response..."
        )

        generation_start = time.time()

        if self.device == "cuda":

            torch.cuda.synchronize()

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            )

        if self.device == "cuda":

            torch.cuda.synchronize()

        generation_time = (
            time.time()
            -
            generation_start
        )


        input_token_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = (
            generated_ids[
                :,
                input_token_length:
            ]
        )


        answer = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        )

        answer = answer.strip()

        gpu_memory = None

        if self.device == "cuda":

            gpu_memory = round(
                torch.cuda.memory_allocated()
                /
                (1024 ** 3),
                2
            )


        total_time = (
            time.time()
            -
            total_start
        )


        print(
            "\n========== CROSS-MODAL TIMING =========="
        )

        print(
            f"Model loading : "
            f"{model_time:.2f} sec"
        )

        print(
            f"Fusion        : "
            f"{fusion_time:.2f} sec"
        )

        print(
            f"Processing    : "
            f"{processing_time:.2f} sec"
        )

        print(
            f"Generation    : "
            f"{generation_time:.2f} sec"
        )

        print(
            f"Total         : "
            f"{total_time:.2f} sec"
        )

        print(
            "========================================="
        )

        return {
            "model": self.model_id,
            "query": query,
            "answer": answer,
            "device": self.device,
            "gpu_memory_gb": gpu_memory,
            "fusion_statistics": (
                fusion_result["statistics"]
            ),
            "generation_time_seconds": round(
                generation_time,
                2
            ),
            "total_time_seconds": round(
                total_time,
                2
            )
        }

    def compare(
        self,
        optical_path,
        sar_path,
        query=(
            "What information can be identified "
            "by combining the optical and SAR imagery?"
        )
    ):

        result = self.analyze(
            optical_path=optical_path,
            sar_path=sar_path,
            query=query
        )

        return result["answer"]


    def status(self):

        return {
            "model_id": self.model_id,
            "device": self.device,
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "loaded": self.loaded
        }


    def unload_model(self):

        if self.model is not None:

            del self.model

        if self.processor is not None:

            del self.processor

        self.model = None
        self.processor = None

        self.loaded = False

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            "Cross-Modal model unloaded."
        )



cross_modal_engine = (
    CrossModalInference()
)



def analyze_cross_modal(
    optical_path,
    sar_path,
    query
):

    return cross_modal_engine.analyze(
        optical_path=optical_path,
        sar_path=sar_path,
        query=query
    )


if __name__ == "__main__":

    optical_path = (
        "data/input/T1.png"
    )

    sar_path = (
        "data/input/Sample.tif"
    )

    query = (
        "What information can be identified "
        "by combining the optical and SAR imagery?"
    )

    print(
        "\n========== CROSS-MODAL TEST =========="
    )

    print(
        f"Model : {MODEL_ID}"
    )

    print(
        f"Device: {DEVICE}"
    )

    try:

        result = analyze_cross_modal(
            optical_path=optical_path,
            sar_path=sar_path,
            query=query
        )

        print(
            "\nCross-Modal Answer:"
        )

        print(
            result["answer"]
        )

        print(
            "\nModel:"
        )

        print(
            result["model"]
        )

        print(
            "\nDevice:"
        )

        print(
            result["device"]
        )

        if result["gpu_memory_gb"] is not None:

            print(
                f"\nGPU memory allocated: "
                f"{result['gpu_memory_gb']} GB"
            )

        print(
            "\n=============================="
        )

    except Exception as e:

        print(
            f"\nCross-Modal inference error: {e}"
        )