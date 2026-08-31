import os
import time
from pathlib import Path

import torch
import rasterio
import numpy as np

from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from models.bta.prompts import (
    SYSTEM_PROMPT,
    build_general_prompt,
    build_change_detection_prompt,
    build_urban_change_prompt,
    build_vegetation_change_prompt,
    build_water_change_prompt,
    build_infrastructure_change_prompt,
    build_environmental_change_prompt,
    build_change_summary_prompt,
    build_no_change_verification_prompt,
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

MAX_IMAGE_SIZE = 1024

DEFAULT_MAX_NEW_TOKENS = 128

DEFAULT_TEMPERATURE = 0.0



QUANTIZATION_CONFIG = None

if DEVICE == "cuda":

    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )



class BTAInference:

    def __init__(
        self,
        model_id=MODEL_ID,
        device=DEVICE
    ):

        self.model_id = model_id
        self.device = device

        self.model = None
        self.processor = None

        self.loaded = False



    def load_model(self):

        if self.loaded:
            return

        print(
            f"\nLoading BTA model: {self.model_id}"
        )

        print(
            f"Device: {self.device}"
        )


        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
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
                    quantization_config=QUANTIZATION_CONFIG,
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
                "Loading BTA model on CPU."
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
            "\nBTA model loaded successfully."
        )



    @staticmethod
    def load_geotiff(
        image_path,
        max_size=MAX_IMAGE_SIZE
    ):
        """
        Read a GeoTIFF or normal image and convert it to RGB.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        suffix = image_path.suffix.lower()


        if suffix in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp"
        ]:

            print(
                f"Reading standard image: {image_path}"
            )

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


        else:

            print(
                f"Reading GeoTIFF: {image_path}"
            )

            with rasterio.open(
                image_path
            ) as src:

                band_count = src.count


                if band_count == 1:

                    data = src.read(1)

                    data = data.astype(
                        np.float32
                    )

                    data = np.nan_to_num(
                        data,
                        nan=0.0
                    )

                    low = np.percentile(
                        data,
                        2
                    )

                    high = np.percentile(
                        data,
                        98
                    )

                    if high > low:

                        data = (
                            (data - low)
                            /
                            (high - low)
                            *
                            255.0
                        )

                    else:

                        data = np.zeros_like(
                            data
                        )

                    data = np.clip(
                        data,
                        0,
                        255
                    ).astype(
                        np.uint8
                    )

                    rgb = np.stack(
                        [
                            data,
                            data,
                            data
                        ],
                        axis=-1
                    )


                else:

                    bands = src.read(
                        [1, 2, 3]
                    ).astype(
                        np.float32
                    )

                    rgb_bands = []

                    for band in bands:

                        band = np.nan_to_num(
                            band,
                            nan=0.0
                        )

                        low = np.percentile(
                            band,
                            2
                        )

                        high = np.percentile(
                            band,
                            98
                        )

                        if high > low:

                            band = (
                                (band - low)
                                /
                                (high - low)
                                *
                                255.0
                            )

                        else:

                            band = np.zeros_like(
                                band
                            )

                        band = np.clip(
                            band,
                            0,
                            255
                        )

                        rgb_bands.append(
                            band.astype(
                                np.uint8
                            )
                        )

                    rgb = np.stack(
                        rgb_bands,
                        axis=-1
                    )

            image = Image.fromarray(
                rgb,
                mode="RGB"
            )

        original_size = image.size

        image.thumbnail(
            (
                max_size,
                max_size
            ),
            Image.Resampling.LANCZOS
        )

        print(
            f"Original size: "
            f"{original_size[0]} x {original_size[1]}"
        )

        print(
            f"VLM size: "
            f"{image.width} x {image.height}"
        )

        return image



    @staticmethod
    def build_prompt(
        query,
        analysis_type="general"
    ):

        analysis_type = (
            analysis_type
            .lower()
            .strip()
        )

        if analysis_type == "general":

            return build_general_prompt(
                query
            )

        elif analysis_type == "change_detection":

            return build_change_detection_prompt(
                query
            )

        elif analysis_type == "urban":

            return build_urban_change_prompt(
                query
            )

        elif analysis_type == "vegetation":

            return build_vegetation_change_prompt(
                query
            )

        elif analysis_type == "water":

            return build_water_change_prompt(
                query
            )

        elif analysis_type == "infrastructure":

            return build_infrastructure_change_prompt(
                query
            )

        elif analysis_type == "environmental":

            return build_environmental_change_prompt(
                query
            )

        elif analysis_type == "summary":

            return build_change_summary_prompt()

        elif analysis_type == "no_change":

            return build_no_change_verification_prompt()

        else:

            raise ValueError(
                f"Unsupported BTA analysis type: "
                f"{analysis_type}"
            )


    @staticmethod
    def create_messages(
        prompt
    ):

        return [

            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
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



    def analyze(
        self,
        image_t1_path,
        image_t2_path,
        query,
        analysis_type="general",
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE
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
            - model_start
        )


        image_start = time.time()

        print(
            "\n========== T1 =========="
        )

        image_t1 = self.load_geotiff(
            image_t1_path
        )

        print(
            "\n========== T2 =========="
        )

        image_t2 = self.load_geotiff(
            image_t2_path
        )

        image_time = (
            time.time()
            - image_start
        )


        prompt = self.build_prompt(
            query=query,
            analysis_type=analysis_type
        )


        messages = self.create_messages(
            prompt
        )

        processing_start = time.time()

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[
                image_t1,
                image_t2
            ],
            padding=True,
            return_tensors="pt"
        )

        processing_time = (
            time.time()
            - processing_start
        )

        if self.device == "cuda":

            inputs = {
                key: value.to("cuda")
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

        if self.device == "cuda":

            torch.cuda.synchronize()

        print(
            "\nGenerating BTA response..."
        )

        generation_start = time.time()

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        if self.device == "cuda":

            torch.cuda.synchronize()

        generation_time = (
            time.time()
            - generation_start
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
            clean_up_tokenization_spaces=True
        )[0]

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
            - total_start
        )

        print(
            "\n========== BTA TIMING =========="
        )

        print(
            f"Model loading : "
            f"{model_time:.2f} sec"
        )

        print(
            f"Image loading : "
            f"{image_time:.2f} sec"
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
            "================================"
        )


        return {

            "model": self.model_id,

            "analysis_type": analysis_type,

            "image_t1": str(
                image_t1_path
            ),

            "image_t2": str(
                image_t2_path
            ),

            "query": query,

            "answer": answer,

            "device": self.device,

            "gpu_memory_gb": gpu_memory,

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
        image_t1_path,
        image_t2_path,
        query=(
            "What significant changes occurred "
            "between the two satellite images?"
        )
    ):

        result = self.analyze(

            image_t1_path=image_t1_path,

            image_t2_path=image_t2_path,

            query=query,

            analysis_type="general"
        )

        return result["answer"]



    def status(self):

        return {

            "model_id": self.model_id,

            "device": self.device,

            "cuda_available":
                torch.cuda.is_available(),

            "loaded":
                self.loaded
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
            "BTA model unloaded."
        )


bta_engine = BTAInference()



def analyze_change(
    image_t1_path,
    image_t2_path,
    query,
    analysis_type="general"
):

    return bta_engine.analyze(

        image_t1_path=image_t1_path,

        image_t2_path=image_t2_path,

        query=query,

        analysis_type=analysis_type
    )



def run_bta(
    t1_path,
    t2_path,
    query,
    analysis_type="general"
):
    """
    Interface used by routing.graph.

    This is only a wrapper around the existing BTA engine.
    """

    return bta_engine.analyze(

        image_t1_path=t1_path,

        image_t2_path=t2_path,

        query=query,

        analysis_type=analysis_type
    )



if __name__ == "__main__":

    image_t1_path = (
        "data/input/T1.png"
    )

    image_t2_path = (
        "data/input/T2.png"
    )

    query = (
        "What significant changes occurred "
        "between the two images?"
    )

    print(
        "\n========== BTA TEST =========="
    )

    print(
        f"Model : {MODEL_ID}"
    )

    print(
        f"Device: {DEVICE}"
    )

    try:

        result = analyze_change(

            image_t1_path=image_t1_path,

            image_t2_path=image_t2_path,

            query=query,

            analysis_type="general"
        )

        print(
            "\nBTA Answer:"
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
            "\n==============================\n"
        )

    except Exception as e:

        print(
            f"\nBTA inference error: {e}"
        )