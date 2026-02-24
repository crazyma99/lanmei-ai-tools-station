import os
import json
import shutil
import random
import piexif
from PIL import Image, ImageChops, ImageEnhance
from PIL.PngImagePlugin import PngInfo

__all__ = [
    "get_exif_data",
    "remove_exif",
    "modify_exif",
    "create_thumbnail",
    "detect_aigc_from_exif",
    "strip_aigc_metadata",
    "add_grain",
    "deep_clean_image",
]


def _apply_deep_clean(img, intensity=0.0):
    if img.mode in ("P", "1", "LA"):
        img = img.convert("RGBA")
    img = img.copy()
    w_orig, h_orig = img.size
    angle = random.uniform(-0.25, 0.25)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    w_rot, h_rot = img.size
    crop_w = int(w_orig * 0.99)
    crop_h = int(h_orig * 0.99)
    left = (w_rot - crop_w) // 2
    top = (h_rot - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))
    w_curr, h_curr = img.size
    new_w = int(w_curr * 0.998)
    new_h = int(h_curr * 0.998)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    w_curr, h_curr = img.size
    if w_curr > 4 and h_curr > 4:
        img = img.crop((1, 1, w_curr, h_curr))
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.99, 1.01)
    img = enhancer.enhance(factor)
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(0.99, 1.01)
    img = enhancer.enhance(factor)
    applied_intensity = max(0.1, intensity)
    img = add_grain(img, applied_intensity)
    return img


def deep_clean_image(image_path, output_path, intensity=0.0):
    try:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with Image.open(image_path) as img:
            processed_img = _apply_deep_clean(img, intensity)
            ext = os.path.splitext(output_path)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                if processed_img.mode == "RGBA":
                    processed_img = processed_img.convert("RGB")
                processed_img.save(
                    output_path, format="JPEG", quality=98, subsampling=0
                )
            elif ext in [".png"]:
                processed_img.save(output_path, format="PNG", optimize=True)
            elif ext in [".webp"]:
                processed_img.save(output_path, format="WEBP", quality=100)
            else:
                processed_img.save(output_path)
        return True
    except Exception as e:
        print(f"Deep clean failed: {e}")
        return False


def add_grain(image, intensity=0.1):
    if intensity <= 0:
        return image
    sigma = max(0.5, intensity * 0.2)
    base_res = 1600
    w, h = image.size
    long_edge = max(w, h)
    scale_factor = max(1.0, long_edge / base_res)
    noise_w = int(w / scale_factor)
    noise_h = int(h / scale_factor)
    try:
        if image.mode == "RGB":
            ycbcr = image.convert("YCbCr")
            y, cb, cr = ycbcr.split()
            noise_small = Image.effect_noise((noise_w, noise_h), sigma)
            noise_y = noise_small.resize((w, h), Image.Resampling.BILINEAR)
            y_with_noise = ImageChops.overlay(y, noise_y)
            merged = Image.merge("YCbCr", (y_with_noise, cb, cr))
            return merged.convert("RGB")
        if image.mode == "L":
            noise_small = Image.effect_noise((noise_w, noise_h), sigma)
            noise = noise_small.resize((w, h), Image.Resampling.BILINEAR)
            return ImageChops.overlay(image, noise)
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            rgb = Image.merge("RGB", (r, g, b))
            noisy_rgb = add_grain(rgb, intensity)
            r2, g2, b2 = noisy_rgb.split()
            return Image.merge("RGBA", (r2, g2, b2, a))
        rgb = image.convert("RGB")
        return add_grain(rgb, intensity)
    except Exception as e:
        print(f"Error adding grain: {e}")
        return image


def get_exif_data(image_path):
    try:
        readable_exif = {}
        with Image.open(image_path) as img:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                try:
                    exif_dict = piexif.load(exif_bytes)
                    for ifd in ("0th", "Exif", "GPS", "1st"):
                        if ifd in exif_dict:
                            readable_exif[ifd] = {}
                            for tag in exif_dict[ifd]:
                                try:
                                    tag_name = piexif.TAGS[ifd][tag]["name"]
                                    value = exif_dict[ifd][tag]
                                    if isinstance(value, bytes):
                                        if tag_name == "UserComment":
                                            try:
                                                prefix = value[:8]
                                                rest = value[8:]
                                                if prefix.startswith(b"ASCII"):
                                                    value = rest.decode(
                                                        "ascii", errors="ignore"
                                                    )
                                                elif prefix.startswith(b"UNICODE"):
                                                    value = rest.decode(
                                                        "utf-16", errors="ignore"
                                                    )
                                                elif prefix.startswith(b"JIS"):
                                                    try:
                                                        value = rest.decode(
                                                            "shift_jis",
                                                            errors="ignore",
                                                        )
                                                    except Exception:
                                                        value = rest.decode(
                                                            "utf-8", errors="ignore"
                                                        )
                                                else:
                                                    value = value.decode(
                                                        "utf-8", errors="ignore"
                                                    )
                                            except Exception:
                                                try:
                                                    value = value.decode(
                                                        "utf-8", errors="ignore"
                                                    )
                                                except Exception:
                                                    value = f"<bytes: {len(value)}>"
                                        else:
                                            try:
                                                value = value.decode("utf-8")
                                            except Exception:
                                                value = f"<bytes: {len(value)}>"
                                    readable_exif[ifd][tag_name] = value
                                except KeyError:
                                    continue
                except Exception as e:
                    print(f"Error parsing EXIF bytes: {e}")
            if (img.format or "").lower() == "png":
                png_info = {}
                for k, v in img.info.items():
                    if k == "exif":
                        continue
                    if isinstance(v, (str, int, float, bool, type(None))):
                        png_info[k] = v
                    else:
                        png_info[k] = str(v)
                if png_info:
                    readable_exif["PNG Info"] = png_info
            if hasattr(img, "getxmp"):
                try:
                    xmp_data = img.getxmp()
                    if xmp_data:
                        readable_exif["XMP"] = xmp_data
                except Exception as e:
                    print(f"Error getting XMP: {e}")
        return readable_exif
    except Exception as e:
        print(f"Error reading EXIF: {e}")
        return {}


def remove_exif(image_path, output_path, add_noise=False, noise_intensity=0):
    try:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if add_noise:
            with Image.open(image_path) as img:
                processed_img = add_grain(img, noise_intensity)
                fmt = (img.format or "").upper()
                if fmt == "JPEG":
                    processed_img.save(output_path, quality=95)
                elif fmt == "PNG":
                    pnginfo = PngInfo()
                    processed_img.save(
                        output_path, format="PNG", pnginfo=pnginfo, optimize=True
                    )
                elif fmt == "WEBP":
                    processed_img.save(output_path, format="WEBP", lossless=True)
                else:
                    base = processed_img
                    if base.mode in ("P", "1"):
                        base = base.convert("RGB")
                    base.save(output_path, quality=100)
            return True
        is_jpeg = False
        try:
            with Image.open(image_path) as img:
                if img.format == "JPEG":
                    is_jpeg = True
        except Exception:
            pass
        if is_jpeg:
            shutil.copy(image_path, output_path)
            try:
                piexif.remove(output_path)
                return True
            except Exception as e:
                print(f"piexif remove failed: {e}, falling back to PIL")
        with Image.open(image_path) as img:
            fmt = (img.format or "").upper()
            if fmt == "PNG":
                pnginfo = PngInfo()
                img.save(output_path, format="PNG", pnginfo=pnginfo, optimize=True)
            elif fmt == "WEBP":
                img.save(output_path, format="WEBP", lossless=True)
            else:
                base = img
                if img.mode in ("P", "1"):
                    base = img.convert("RGB")
                base.save(output_path, quality=100, subsampling=0)
        return True
    except Exception as e:
        print(f"Error removing EXIF: {e}")
        return False


def modify_exif(
    image_path,
    output_path,
    exif_json_path=None,
    preset_data=None,
    convert_to_jpg=False,
    add_noise=False,
    noise_intensity=0,
    deep_clean=False,
):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if exif_json_path:
            with open(exif_json_path, "r", encoding="utf-8") as f:
                target_exif = json.load(f)
        elif preset_data:
            target_exif = preset_data
        else:
            return False
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        def to_tuple(val):
            if isinstance(val, list):
                return tuple(to_tuple(i) for i in val)
            return val

        def convert_value(tag_type, value):
            if tag_type == 2:
                if isinstance(value, str):
                    return value.encode("utf-8")
            elif tag_type in (5, 10):
                return to_tuple(value)
            elif tag_type == 7:
                if isinstance(value, str):
                    return value.encode("utf-8")
            return value

        def map_keys_to_id(ifd_name, data_dict):
            mapped = {}
            if ifd_name not in piexif.TAGS:
                return {}
            name_to_id = {
                info["name"]: tag for tag, info in piexif.TAGS[ifd_name].items()
            }
            tag_types = {
                tag: info.get("type") for tag, info in piexif.TAGS[ifd_name].items()
            }
            for k, v in data_dict.items():
                if k in name_to_id:
                    tag_id = name_to_id[k]
                    tag_type = tag_types.get(tag_id)
                    try:
                        converted_v = convert_value(tag_type, v)
                        mapped[tag_id] = converted_v
                    except Exception as conv_e:
                        print(f"Warning: Failed to convert tag {k}: {conv_e}")
                        mapped[tag_id] = v
            return mapped

        if "0th" in target_exif:
            exif_dict["0th"] = map_keys_to_id("0th", target_exif["0th"])
        if "Exif" in target_exif:
            exif_dict["Exif"] = map_keys_to_id("Exif", target_exif["Exif"])
        if "GPS" in target_exif:
            exif_dict["GPS"] = map_keys_to_id("GPS", target_exif["GPS"])
        exif_bytes = piexif.dump(exif_dict)
        is_jpeg = False
        if not convert_to_jpg and not add_noise and not deep_clean:
            try:
                with Image.open(image_path) as img:
                    if img.format == "JPEG":
                        is_jpeg = True
            except Exception:
                pass
        if convert_to_jpg or add_noise or deep_clean:
            with Image.open(image_path) as img:
                processed_img = img
                if deep_clean:
                    processed_img = _apply_deep_clean(img, noise_intensity)
                elif add_noise:
                    processed_img = add_grain(img, noise_intensity)
                if convert_to_jpg:
                    if processed_img.mode != "RGB":
                        processed_img = processed_img.convert("RGB")
                    processed_img.save(
                        output_path, "JPEG", exif=exif_bytes, quality=95
                    )
                else:
                    fmt = (img.format or "JPEG").upper()
                    if fmt == "JPEG":
                        if processed_img.mode != "RGB":
                            processed_img = processed_img.convert("RGB")
                        processed_img.save(
                            output_path, "JPEG", exif=exif_bytes, quality=95
                        )
                    else:
                        processed_img.save(output_path, exif=exif_bytes, quality=100)
        elif is_jpeg:
            shutil.copy(image_path, output_path)
            piexif.insert(exif_bytes, output_path)
        else:
            with Image.open(image_path) as img:
                img.save(output_path, exif=exif_bytes, quality=100, subsampling=0)
        return True
    except Exception as e:
        print(f"Error modifying EXIF: {e}")
        return False


def create_thumbnail(image_path, output_path, size=(200, 200)):
    try:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with Image.open(image_path) as img:
            img.thumbnail(size)
            img.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return False


def detect_aigc_from_exif(exif_data):
    try:
        keywords = [
            "ai generated",
            "ai-generated",
            "aigc",
            "midjourney",
            "stable diffusion",
            "comfyui",
            "dall-e",
            "dalle",
            "firefly",
            "novelai",
            "runway",
            "ideogram",
            "leonardo",
            "generated by",
            "sdxl",
            "flux",
            "controlnet",
            "lora",
        ]
        cn_keys = ["ai生成", "由ai生成", "aigc生成", "人工智能生成"]

        def check_text(text):
            if not isinstance(text, str):
                return None
            lower_text = text.lower()
            for kw in keywords:
                if kw in lower_text:
                    return kw
            for kw in cn_keys:
                if kw in lower_text:
                    return kw
            return None

        if isinstance(exif_data, dict):
            exif_ifd = exif_data.get("Exif", {})
            zero_ifd = exif_data.get("0th", {})
            if "UserComment" in exif_ifd:
                match = check_text(exif_ifd["UserComment"])
                if match:
                    return {
                        "is_aigc": True,
                        "matched": match,
                        "source": "UserComment",
                    }
            if "ImageDescription" in zero_ifd:
                match = check_text(zero_ifd["ImageDescription"])
                if match:
                    return {
                        "is_aigc": True,
                        "matched": match,
                        "source": "ImageDescription",
                    }
            if "Software" in zero_ifd:
                match = check_text(zero_ifd["Software"])
                if match:
                    return {
                        "is_aigc": True,
                        "matched": match,
                        "source": "Software",
                    }
        png_info = exif_data.get("PNG Info", {})
        if isinstance(png_info, dict):
            if "parameters" in png_info:
                match = check_text(png_info["parameters"])
                if match:
                    return {
                        "is_aigc": True,
                        "matched": match,
                        "source": "PNG Parameters",
                    }
                if "Steps:" in png_info["parameters"]:
                    return {
                        "is_aigc": True,
                        "matched": "Steps:",
                        "source": "PNG Parameters",
                    }
            for k, v in png_info.items():
                match = check_text(str(v))
                if match:
                    return {
                        "is_aigc": True,
                        "matched": match,
                        "source": f"PNG {k}",
                    }
        xmp = exif_data.get("XMP")
        if xmp:
            match = check_text(str(xmp))
            if match:
                return {"is_aigc": True, "matched": match, "source": "XMP"}
        return {"is_aigc": False}
    except Exception as e:
        print(f"Error detecting AIGC: {e}")
        return {"is_aigc": False}


def strip_aigc_metadata(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)
            fmt = img.format or "JPEG"
            if fmt == "JPEG":
                pass
        return True
    except Exception:
        return False

