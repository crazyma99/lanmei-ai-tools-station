import os
import tempfile
from typing import Callable, Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import BISE_ONNX_PATH, BISE_MODEL_PATH, BISE_NET_DIR, FUSION_OUTPUT_DIR


def ensure_fusion_output_dir() -> None:
    FUSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DeviceWrapper:
    def __init__(self) -> None:
        self.providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in self.providers:
            self.device_type = "cuda"
        elif "CoreMLExecutionProvider" in self.providers:
            self.device_type = "coreml"
        else:
            self.device_type = "cpu"

    @property
    def label(self) -> str:
        mapping = {"cuda": "CUDA(GPU)", "coreml": "CoreML(GPU)", "cpu": "CPU"}
        return mapping.get(self.device_type, self.device_type.upper())


DEVICE_WRAPPER = DeviceWrapper()
FACE_ONNX_SESSION: Optional[ort.InferenceSession] = None


def get_face_onnx_session() -> Optional[ort.InferenceSession]:
    global FACE_ONNX_SESSION
    if FACE_ONNX_SESSION is not None:
        return FACE_ONNX_SESSION
    if not BISE_ONNX_PATH.exists():
        return None
    
    providers = []
    if DEVICE_WRAPPER.device_type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif DEVICE_WRAPPER.device_type == "coreml":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(BISE_ONNX_PATH), providers=providers)
    except Exception as e:
        print(f"加载 BiSeNet ONNX 模型失败: {e}")
        return None
    FACE_ONNX_SESSION = sess
    return sess


def get_bisenet_parsing(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img_norm = (img_norm - mean) / std
    img_chw = img_norm.transpose(2, 0, 1)
    
    sess = get_face_onnx_session()
    if sess is None:
        return None

    try:
        inp = img_chw[None, ...].astype("float32")
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: inp})[0]
        if out.ndim == 4:
            logits = out[0]
            parsing = logits.argmax(0).astype(np.uint8)
            parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
            return parsing
    except Exception as e:
        print(f"BiSeNet ONNX 推理失败: {e}")
    
    return None


def get_face_mask_bisenet(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    parsing = get_bisenet_parsing(img_bgr)
    if parsing is None:
        return None
    face_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17], dtype=np.uint8)
    mask = np.isin(parsing, face_ids).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def get_person_mask_bisenet(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    parsing = get_bisenet_parsing(img_bgr)
    if parsing is None:
        return None
    keep_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 18], dtype=np.uint8)
    mask = np.isin(parsing, keep_ids).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def inpaint_face_region(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = img_bgr.shape[:2]
    face_mask = get_face_mask_bisenet(img_bgr)
    if face_mask is None or np.count_nonzero(face_mask) == 0:
        return img_bgr, None
    mask = (face_mask > 0).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpaint_mask = mask.astype("uint8")
    filled = cv2.inpaint(img_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    return filled, inpaint_mask


def color_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source_lab = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target.astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")
    kernel_stat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_stat = cv2.erode(mask, kernel_stat, iterations=2)
    source_pixels = source_lab[mask_stat > 0]
    target_pixels = target_lab[mask_stat > 0]
    if len(source_pixels) < 100 or len(target_pixels) < 100:
        return source
    s_mean, s_std = cv2.meanStdDev(source_lab, mask=mask_stat)
    t_mean, t_std = cv2.meanStdDev(target_lab, mask=mask_stat)
    s_mean = s_mean.flatten()
    s_std = s_std.flatten()
    t_mean = t_mean.flatten()
    t_std = t_std.flatten()
    res_lab = source_lab.copy()
    l_scale = float(np.clip(t_std[0] / (s_std[0] + 1e-5), 0.8, 1.2))
    res_lab[:, :, 0] = (source_lab[:, :, 0] - s_mean[0]) * l_scale + t_mean[0]
    for i in range(1, 3):
        ab_scale = float(np.clip(t_std[i] / (s_std[i] + 1e-5), 0.5, 1.5))
        res_lab[:, :, i] = (source_lab[:, :, i] - s_mean[i]) * ab_scale + t_mean[i]
        res_lab[:, :, i] = res_lab[:, :, i] * 0.7 + source_lab[:, :, i] * 0.3
    res_lab = np.clip(res_lab, 0, 255).astype("uint8")
    transfer = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
    result = source.copy()
    result[mask > 0] = transfer[mask > 0]
    return result


def blend_images_smart(
    img_source_nobg: np.ndarray,
    img_target: np.ndarray,
    progress: Optional[Callable[[float, str], None]] = None,
    mode: str = "poisson_normal",
    use_color: bool = True,
    face_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], str]:
    if progress:
        progress(0.4, "正在进行特征对齐...")
    img1_src_bgr = img_source_nobg[:, :, :3]
    img1_mask = img_source_nobg[:, :, 3]
    img2_target_bgr = img_target
    if progress:
        progress(0.45, "正在计算特征点...")
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(img1_src_bgr, None)
    kp2, des2 = sift.detectAndCompute(img2_target_bgr, None)
    if des1 is None or des2 is None:
        return None, "特征描述符提取失败，图像可能缺乏纹理。"
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 20:
        return None, "有效匹配点太少，无法精准对齐。"
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    if progress:
        progress(0.5, "正在计算相似变换矩阵...")
    m_affine, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m_affine is None:
        return None, "无法计算对齐矩阵。"
    m_mat = np.vstack([m_affine, [0, 0, 1]])
    scale_x = float(np.sqrt(m_mat[0, 0] ** 2 + m_mat[0, 1] ** 2))
    if scale_x < 0.1 or scale_x > 10.0:
        return None, f"检测到异常缩放比例 ({scale_x:.2f})。"
    h_bg, w_bg = img2_target_bgr.shape[:2]
    if progress:
        progress(0.55, f"正在进行图像变换 ({w_bg}x{h_bg})...")
    img1_warped = cv2.warpPerspective(img1_src_bgr, m_mat, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)
    mask_warped = cv2.warpPerspective(img1_mask, m_mat, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img1_warped_sharpened = cv2.filter2D(img1_warped, -1, kernel_sharpen)
    img1_warped = cv2.addWeighted(img1_warped, 0.8, img1_warped_sharpened, 0.2, 0)
    _, mask_binary = cv2.threshold(mask_warped, 128, 255, cv2.THRESH_BINARY)
    kernel_clean = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_clean)
    mask_binary = cv2.GaussianBlur(mask_binary, (5, 5), 0)
    _, mask_binary = cv2.threshold(mask_binary, 128, 255, cv2.THRESH_BINARY)
    face_mask_warped = None
    if face_mask is not None:
        try:
            fm = face_mask
            if fm.shape[:2] != img1_src_bgr.shape[:2]:
                fm = cv2.resize(fm, (img1_src_bgr.shape[1], img1_src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            face_mask_warped = cv2.warpPerspective(fm, m_mat, (w_bg, h_bg), flags=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"BiSeNet 面部 mask 变换失败: {e}")
            face_mask_warped = None
    if use_color:
        if progress:
            progress(0.65, "正在进行色彩协调...")
        img1_warped_transferred = color_transfer(img1_warped, img2_target_bgr, mask_binary)
    else:
        img1_warped_transferred = img1_warped.copy()
    if progress:
        progress(0.8, "正在进行泊松融合...")
    y_indices, x_indices = np.where(mask_binary > 0)
    if len(x_indices) == 0:
        return None, "对齐后图像在视野外。"
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    kernel_shrink = np.ones((3, 3), np.uint8)
    mask_for_clone = cv2.erode(mask_binary, kernel_shrink, iterations=2)
    mask_for_clone[0:10, :] = 0
    mask_for_clone[-10:, :] = 0
    mask_for_clone[:, 0:10] = 0
    mask_for_clone[:, -10:] = 0
    seamless = None
    if mode.startswith("poisson"):
        clone_flag = cv2.NORMAL_CLONE if mode == "poisson_normal" else cv2.MIXED_CLONE
        try:
            seamless = cv2.seamlessClone(img1_warped_transferred, img2_target_bgr, mask_for_clone, center, clone_flag)
        except Exception as e:
            print(f"泊松融合失败: {e}")
            seamless = img2_target_bgr.copy()
            mask_inv = cv2.bitwise_not(mask_binary)
            img2_bg = cv2.bitwise_and(seamless, seamless, mask=mask_inv)
            img1_fg = cv2.bitwise_and(img1_warped_transferred, img1_warped_transferred, mask=mask_binary)
            seamless = cv2.add(img2_bg, img1_fg)
    else:
        seamless = img2_target_bgr.copy()
    if progress:
        progress(0.9, "正在进行边缘羽化与最终合成...")
    kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_main = cv2.erode(mask_binary, kernel_feather, iterations=3)
    feather_amount = int(min(w_bg, h_bg) * 0.008)
    if feather_amount < 11:
        feather_amount = 11
    if feather_amount % 2 == 0:
        feather_amount += 1
    mask_feathered = cv2.GaussianBlur(mask_main, (feather_amount, feather_amount), 0)
    kernel_size_protect = int(min(w_bg, h_bg) * 0.02)
    if kernel_size_protect % 2 == 0:
        kernel_size_protect += 1
    kernel_protect = np.ones((kernel_size_protect, kernel_size_protect), np.uint8)
    mask_eroded = cv2.erode(mask_main, kernel_protect, iterations=1)
    if face_mask_warped is not None:
        _, face_bin = cv2.threshold(face_mask_warped, 128, 255, cv2.THRESH_BINARY)
        kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        face_bin = cv2.dilate(face_bin, kernel_face, iterations=1)
        mask_eroded = cv2.bitwise_or(mask_eroded, face_bin)
    blur_radius_protect = int(min(w_bg, h_bg) * 0.08)
    if blur_radius_protect % 2 == 0:
        blur_radius_protect += 1
    mask_blur_protect = cv2.GaussianBlur(mask_eroded, (blur_radius_protect, blur_radius_protect), 0)
    alpha_edge = mask_feathered.astype(float) / 255.0
    alpha_edge = np.stack([alpha_edge] * 3, axis=2)
    alpha_protect_single = mask_blur_protect.astype(float) / 255.0
    core_strong = (alpha_protect_single >= 0.6).astype(np.float32)
    alpha_protect_single = alpha_protect_single * (1.0 - core_strong) + core_strong
    alpha_protect = np.stack([alpha_protect_single] * 3, axis=2)
    blended_inner = img1_warped_transferred.astype(float) * alpha_protect + seamless.astype(float) * (1 - alpha_protect)
    final_output = (blended_inner * alpha_edge + img2_target_bgr.astype(float) * (1 - alpha_edge)).astype(np.uint8)
    tag = {
        ("poisson_normal", True): "泊松-标准+协调",
        ("poisson_mixed", True): "泊松-混合+协调",
        ("poisson_normal", False): "泊松-标准",
        ("poisson_mixed", False): "泊松-混合",
        ("alpha", True): "羽化-Alpha+协调",
        ("alpha", False): "羽化-Alpha",
    }.get((mode, use_color), "融合结果")
    return final_output, f"处理成功：{tag}"


def run_fusion(
    img1_pil: Image.Image,
    img2_pil: Image.Image,
    output_dir: Path,
    progress: Optional[Callable[[float, str], None]] = None,
    add_watermark: bool = False,
) -> Dict[str, Any]:
    if img1_pil is None or img2_pil is None:
        return {"message": "请先上传两张图片。", "variants": []}
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if progress:
        progress(0.0, "正在初始化...")
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)
    face_mask = None
    try:
        face_mask = get_face_mask_bisenet(img1)
    except Exception as e:
        print(f"BiSeNet 面部分割失败: {e}")
    h, w = img2.shape[:2]
    target_max = 6048
    if max(h, w) != target_max:
        scale = target_max / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if progress:
            progress(0.1, f"正在调整底图分辨率至 {target_max}px...")
        img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img2.shape[:2]
    if progress:
        progress(0.15, "正在对底图原有人脸做内容填充...")
    img2_filled, bg_face_mask = inpaint_face_region(img2)
    if bg_face_mask is not None:
        img2 = img2_filled
    if progress:
        progress(0.2, "正在进行前景分割...")
    person_mask = get_person_mask_bisenet(img1)
    if person_mask is None:
        alpha = np.full(img1.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = person_mask
    if face_mask is not None:
        h_fg, w_fg = img1.shape[:2]
        if face_mask.shape[:2] != (h_fg, w_fg):
            face_mask_resized = cv2.resize(face_mask, (w_fg, h_fg), interpolation=cv2.INTER_NEAREST)
        else:
            face_mask_resized = face_mask
        kernel_face_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        face_safe = cv2.dilate(face_mask_resized, kernel_face_safe, iterations=2)
        alpha = cv2.bitwise_or(alpha, face_safe)
    img1_nobg_cv = np.dstack([img1, alpha])
    if progress:
        progress(0.4, "正在开始对齐与融合...")
    variants_spec = [
        ("算法：泊松-标准", dict(mode="poisson_normal", use_color=True)),
    ]
    before_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    results: List[Dict[str, Any]] = []

    def save_variant_image(img_bgr: np.ndarray, tag: str) -> Dict[str, Any]:
        name_prefix = f"fusion_{tag}"
        png_path = output_dir / f"{name_prefix}.png"
        jpg_path = output_dir / f"{name_prefix}.jpg"
        wm_path = output_dir / f"{name_prefix}_watermark.jpg"
        
        cv2.imwrite(str(png_path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        quality = 99
        while quality > 50:
            cv2.imwrite(str(jpg_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if os.path.getsize(jpg_path) < 10 * 1024 * 1024:
                break
            quality -= 5
            
        # 生成水印
        if add_watermark:
            wm_text = "Lanmei AI Portrait Studio"
            wm_img = img_bgr.copy()
            h, w = wm_img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_scale = max(w, h) / 1600.0
            font_scale = max(base_scale, 0.8)
            thickness = int(max(w, h) / 900)
            if thickness < 1:
                thickness = 1
            
            scale_canvas = 1.6
            big_h = int(h * scale_canvas)
            big_w = int(w * scale_canvas)
            overlay_big = np.zeros((big_h, big_w, 3), dtype=wm_img.dtype)
            text_size, _ = cv2.getTextSize(wm_text, font, font_scale, thickness)
            tw, th = text_size
            step_x = int(tw * 1.2)
            step_y = int(th * 2.2)
            if step_x < 80: step_x = 80
            if step_y < 60: step_y = 60
            for y in range(0, big_h + step_y, step_y):
                for x in range(-big_w // 2, big_w + step_x, step_x):
                    cv2.putText(overlay_big, wm_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            center_big = (big_w // 2, big_h // 2)
            rot_mat = cv2.getRotationMatrix2D(center_big, 45, 1.0)
            overlay_big = cv2.warpAffine(overlay_big, rot_mat, (big_w, big_h))
            y0 = (big_h - h) // 2
            x0 = (big_w - w) // 2
            overlay = overlay_big[y0:y0 + h, x0:x0 + w]
            alpha = 0.15
            wm_img = cv2.addWeighted(overlay, alpha, wm_img, 1 - alpha, 0)
            cv2.imwrite(str(wm_path), wm_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "png": png_path.name,
            "jpg": jpg_path.name,
            "watermark": wm_path.name if add_watermark else None,
        }

    for idx, (title, params) in enumerate(variants_spec):
        if progress:
            progress(0.5 + idx * 0.15, f"正在生成 {title} ...")
        res_img, msg = blend_images_smart(
            img1_nobg_cv,
            img2,
            progress=progress,
            face_mask=face_mask,
            **params,
        )
        if res_img is None:
            results.append(
                {
                    "title": title,
                    "success": False,
                    "message": msg,
                }
            )
            continue
        out_h, out_w = res_img.shape[:2]
        urls = save_variant_image(res_img, f"v{idx+1}")
        # 这里不返回 heavy data，只返回 url 和 meta
        results.append(
            {
                "title": title,
                "success": True,
                "message": msg,
                "resolution": {"width": int(out_w), "height": int(out_h)},
                "urls": urls,
            }
        )
    if progress:
        progress(1.0, "全部结果生成完成")
    return {"device": DEVICE_WRAPPER.label, "variants": results}


def preview_segmentation(img1_pil: Image.Image) -> Optional[Image.Image]:
    if img1_pil is None:
        return None
    img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2BGR)
    mask = get_face_mask_bisenet(img1)
    if mask is None:
        return None
    overlay = img1.copy()
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
    preview = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(preview)
