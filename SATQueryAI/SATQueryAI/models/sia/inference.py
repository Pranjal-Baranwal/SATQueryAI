import os
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from models.sia.prompts import (
    SYSTEM_PROMPT,
    build_general_prompt,
    build_scene_description_prompt,
    build_object_prompt,
    build_land_cover_prompt,
    build_vegetation_prompt,
    build_water_prompt,
    build_urban_prompt,
    build_change_aware_prompt,
    build_grounding_prompt,
)

from models.sia.grounding import (
    parse_grounding_response,
    boxes_to_mask,
)

from geospatial.visualization import (
    SpatialVisualizer,
)


load_dotenv()



MODEL_ID = os.getenv(
    "SIA_MODEL_ID",
    "Qingyun/RSCoVLM-7B-2512"
)

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"



QUANTIZATION_CONFIG = None

if CUDA_AVAILABLE:

    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )



class SIAInference:
    """
    Single Image Analysis inference engine.

    Uses a pretrained remote-sensing Vision-Language Model
    with 4-bit quantization for memory-efficient inference.
    """

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
        """
        Load the pretrained VLM.

        On CUDA:
            Uses 4-bit quantization.

        On CPU:
            Uses float32.
        """

        if self.loaded:
            return

        print(
            f"\nLoading SIA model: {self.model_id}"
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
                "WARNING: CUDA is unavailable."
            )

            print(
                "Loading model on CPU."
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
            "\nSIA model loaded successfully."
        )


    @staticmethod
    def validate_image(image_path):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        supported_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff"
        ]

        if image_path.suffix.lower() not in supported_formats:

            raise ValueError(
                "Unsupported image format. "
                f"Supported formats: {supported_formats}"
            )

        return image_path

    @staticmethod
    def load_image(image_path):

        image_path = SIAInference.validate_image(
            image_path
        )

        image = Image.open(
            image_path
        )

        image = image.convert(
            "RGB"
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

        elif analysis_type == "scene":

            return build_scene_description_prompt()

        elif analysis_type == "object":

            return build_object_prompt(
                query
            )

        elif analysis_type == "land_cover":

            return build_land_cover_prompt(
                query
            )

        elif analysis_type == "vegetation":

            return build_vegetation_prompt(
                query
            )

        elif analysis_type == "water":

            return build_water_prompt(
                query
            )

        elif analysis_type == "urban":

            return build_urban_prompt(
                query
            )

        elif analysis_type == "change_aware":

            return build_change_aware_prompt(
                query
            )

        else:

            raise ValueError(
                f"Unsupported analysis type: "
                f"{analysis_type}"
            )


    @staticmethod
    def create_messages(prompt):

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
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

    def analyze(
        self,
        image_path,
        query,
        analysis_type="general",
        max_new_tokens=256,
        temperature=0.2
    ):
        """
        Perform Single Image Analysis.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        self.load_model()


        image = self.load_image(
            image_path
        )


        prompt = self.build_prompt(
            query=query,
            analysis_type=analysis_type
        )


        messages = self.create_messages(
            prompt
        )


        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        if self.device == "cuda":

            inputs = {
                key: value.to("cuda")
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

        else:

            inputs = {
                key: value.to("cpu")
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }


        print(
            "\nGenerating SIA response..."
        )

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs
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
                / (1024 ** 3),
                2
            )
        return {
            "model": self.model_id,
            "analysis_type": analysis_type,
            "query": query,
            "answer": answer,
            "device": self.device,
            "gpu_memory_gb": gpu_memory
        }


    def ask(
        self,
        image_path,
        query
    ):

        result = self.analyze(
            image_path=image_path,
            query=query,
            analysis_type="general"
        )

        return result["answer"]


    def ground(
        self,
        image_path,
        query,
        max_new_tokens=512,
        output_path=None
    ):
        """
        Localize the region(s) requested in `query` and return
        an annotated image highlighting them.

        The model is asked to return normalized bounding boxes
        as JSON (see models.sia.prompts.build_grounding_prompt).
        Boxes are rasterized into a mask and rendered with
        geospatial.visualization.SpatialVisualizer so the same
        highlight styling used for change masks elsewhere in
        the app is reused here.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        self.load_model()

        image = self.load_image(
            image_path
        )

        image_width, image_height = image.size

        prompt = build_grounding_prompt(
            query
        )

        messages = self.create_messages(
            prompt
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        input_token_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = generated_ids[
            :,
            input_token_length:
        ]

        raw_response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        objects = parse_grounding_response(
            raw_response,
            image_width=image_width,
            image_height=image_height
        )

        if objects:

            mask = boxes_to_mask(
                objects,
                image_width=image_width,
                image_height=image_height
            )

            visualizer = SpatialVisualizer(
                image_path
            )

            visualizer.draw_mask(mask)
            visualizer.draw_mask_boundary(mask)

            if output_path is None:

                stem = Path(image_path).stem

                output_path = (
                    Path("data/output")
                    /
                    f"grounding_{stem}.png"
                )

            output_path = visualizer.save(
                output_path
            )

            answer = (
                f"Found {len(objects)} region(s) matching "
                f"'{query.strip()}': "
                + ", ".join(
                    f"{item['label']} ({item['confidence']} confidence)"
                    for item in objects
                )
            )

        else:

            output_path = None

            answer = (
                f"No region matching '{query.strip()}' "
                "could be confidently localized in the image."
            )

        return {
            "model": self.model_id,
            "query": query,
            "answer": answer,
            "raw_response": raw_response,
            "objects": objects,
            "image_width": image_width,
            "image_height": image_height,
            "image_path": (
                str(output_path)
                if output_path
                else None
            ),
        }


    def status(self):

        return {
            "model_id": self.model_id,
            "device": self.device,
            "cuda_available": CUDA_AVAILABLE,
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
            "SIA model unloaded."
        )



sia_engine = SIAInference()


def analyze_image(
    image_path,
    query,
    analysis_type="general"
):

    return sia_engine.analyze(
        image_path=image_path,
        query=query,
        analysis_type=analysis_type
    )


def run_sia(
    image_path,
    query,
    analysis_type="general"
):
    """
    Execute SIA through the routing pipeline.

    This function is used by routing.graph / app.py.
    """

    return sia_engine.analyze(
        image_path=image_path,
        query=query,
        analysis_type=analysis_type
    )


def ground_image(
    image_path,
    query,
    output_path=None
):
    """
    Convenience wrapper around SIAInference.ground().
    """

    return sia_engine.ground(
        image_path=image_path,
        query=query,
        output_path=output_path
    )


def run_grounding(
    image_path,
    query
):
    """
    Execute visual grounding through the routing pipeline.

    This function is used by routing.graph / app.py.
    """

    return sia_engine.ground(
        image_path=image_path,
        query=query
    )



if __name__ == "__main__":

    image_path = "data/input/T1.png"

    query = (
        "Describe the major features visible "
        "in this satellite image."
    )

    print(
        "\n========== SIA TEST =========="
    )

    print(
        f"Model : {MODEL_ID}"
    )

    print(
        f"Device: {DEVICE}"
    )

    try:

        result = analyze_image(
            image_path=image_path,
            query=query,
            analysis_type="general"
        )

        print(
            "\nModel Answer:"
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
            f"\nSIA inference error: {e}"
        )