mod fft;
mod model;

#[cfg(all(feature = "directml", target_os = "windows"))]
use anyhow::Context;
use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, GrayImage, RgbImage, imageops};
use koharu_runtime::RuntimeManager;
use tracing::instrument;

use crate::{
    device,
    inpainting::{
        HdStrategyConfig, InpaintForward, apply_bubble_fill, binarize_mask, extract_alpha,
        restore_alpha_channel, run_inpaint, run_inpaint_with_windows,
    },
    loading,
    types::TextRegion,
};

const HF_REPO: &str = "mayocream/lama-manga";
#[cfg(all(feature = "directml", target_os = "windows"))]
const ONNX_HF_REPO: &str = "mayocream/lama-manga-onnx";
const BLOCK_WINDOW_RATIO: f64 = 1.7;
const BLOCK_WINDOW_ASPECT_RATIO: f64 = 1.0;

type Xyxy = [u32; 4];

koharu_runtime::declare_hf_model_package!(
    id: "model:lama:weights",
    repo: "mayocream/lama-manga",
    file: "lama-manga.safetensors",
    bootstrap: false,
    order: 130,
);

#[cfg(all(feature = "directml", target_os = "windows"))]
koharu_runtime::declare_hf_model_package!(
    id: "model:lama:onnx",
    repo: "mayocream/lama-manga-onnx",
    file: "lama-manga.onnx",
    bootstrap: false,
    order: 129,
);

pub struct Lama {
    backend: LamaBackend,
    device: Device,
}

enum LamaBackend {
    Candle(model::Lama),
    #[cfg(all(feature = "vulkan-ncnn", target_os = "windows"))]
    Ncnn(crate::comic_text_detector::ncnn::NcnnDetector),
    #[cfg(all(feature = "directml", target_os = "windows"))]
    DirectMl(std::sync::Mutex<ort::session::Session>),
}

impl Lama {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        #[cfg(all(feature = "vulkan-ncnn", target_os = "windows"))]
        if !cpu {
            match crate::comic_text_detector::ncnn::NcnnDetector::load_model(
                "lama_256",
                "lama-manga",
            ) {
                Ok(ncnn) => {
                    return Ok(Self {
                        backend: LamaBackend::Ncnn(ncnn),
                        device: Device::Cpu,
                    });
                }
                Err(error) => tracing::warn!(%error, "ncnn Vulkan LaMa backend unavailable"),
            }
        }
        #[cfg(all(feature = "directml", target_os = "windows"))]
        if !cpu {
            let model_path = runtime
                .downloads()
                .huggingface_model(ONNX_HF_REPO, "lama-manga.onnx")
                .await?;
            let session = (|| -> ort::Result<ort::session::Session> {
                let builder = ort::session::Session::builder()?;
                let device_id = std::env::var("KOHARU_DIRECTML_DEVICE_ID")
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                let mut builder =
                    builder.with_execution_providers([ort::ep::DirectML::default()
                        .with_device_id(device_id)
                        .build()
                        .error_on_failure()])?;
                builder.commit_from_file(&model_path)
            })();
            match session {
                Ok(session) => {
                    tracing::info!("using DirectML for LaMa inpainting");
                    return Ok(Self {
                        backend: LamaBackend::DirectMl(std::sync::Mutex::new(session)),
                        device: Device::Cpu,
                    });
                }
                Err(error) => {
                    tracing::warn!(%error, "DirectML initialization failed; falling back to Candle");
                }
            }
        }

        let device = device(cpu)?;
        let weights_path = runtime
            .downloads()
            .huggingface_model(HF_REPO, "lama-manga.safetensors")
            .await?;
        let model = loading::load_buffered_safetensors_path(&weights_path, &device, |vb| {
            model::Lama::load(&vb)
        })?;

        Ok(Self {
            backend: LamaBackend::Candle(model),
            device,
        })
    }

    /// Run inpainting with the manga-tuned default strategy (Crop, 800/128/1280).
    #[instrument(level = "debug", skip_all)]
    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        bubble_mask: &DynamicImage,
    ) -> Result<DynamicImage> {
        self.inference_with_config_and_blocks(
            image,
            mask,
            bubble_mask,
            None,
            &HdStrategyConfig::lama_default(),
        )
    }

    /// Run inpainting with scene text regions as crop-planning hints. LaMa
    /// uses these to build larger semantic windows than raw mask contours.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_blocks(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        bubble_mask: &DynamicImage,
        text_blocks: &[TextRegion],
    ) -> Result<DynamicImage> {
        self.inference_with_config_and_blocks(
            image,
            mask,
            bubble_mask,
            Some(text_blocks),
            &HdStrategyConfig::lama_default(),
        )
    }

    /// Run inpainting with a caller-supplied [`HdStrategyConfig`]. Use this to
    /// pick a different strategy (Original / Resize) or tune the trigger /
    /// margin / resize-limit for GPUs with less VRAM.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_config(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        bubble_mask: &DynamicImage,
        cfg: &HdStrategyConfig,
    ) -> Result<DynamicImage> {
        self.inference_with_config_and_blocks(image, mask, bubble_mask, None, cfg)
    }

    /// Variant of [`Self::inference_with_config`] that also accepts text
    /// regions for crop planning.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_config_and_blocks(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        bubble_mask: &DynamicImage,
        text_blocks: Option<&[TextRegion]>,
        cfg: &HdStrategyConfig,
    ) -> Result<DynamicImage> {
        if image.dimensions() != mask.dimensions() || image.dimensions() != bubble_mask.dimensions()
        {
            bail!(
                "image/mask/bubble dimensions dismatch: image is {:?}, mask is {:?}, bubble is {:?}",
                image.dimensions(),
                mask.dimensions(),
                bubble_mask.dimensions()
            );
        }

        let binary_mask = binarize_mask(mask);
        let bubble_mask = bubble_mask.to_luma8();
        let image_rgb = image.to_rgb8();
        let crop_windows = text_blocks
            .filter(|blocks| !blocks.is_empty())
            .map(|blocks| crop_windows_from_text_blocks(blocks, image.width(), image.height()))
            .filter(|windows| !windows.is_empty());
        let forward = LamaForward { lama: self };
        let output_rgb = if let Some(windows) = crop_windows.as_deref() {
            tracing::debug!(
                text_block_count = text_blocks.map_or(0, <[TextRegion]>::len),
                crop_window_count = windows.len(),
                "lama text-aware crop planning"
            );
            run_inpaint_with_windows(
                &forward,
                &image_rgb,
                &binary_mask,
                Some(&bubble_mask),
                cfg,
                Some(windows),
            )?
        } else {
            run_inpaint(&forward, &image_rgb, &binary_mask, Some(&bubble_mask), cfg)?
        };

        if image.color().has_alpha() {
            let original_alpha = image.to_rgba8();
            let alpha = extract_alpha(&original_alpha);
            let output = restore_alpha_channel(&output_rgb, &alpha, &binary_mask);
            Ok(DynamicImage::ImageRgba8(output))
        } else {
            Ok(DynamicImage::ImageRgb8(output_rgb))
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        match &self.backend {
            LamaBackend::Candle(model) => model.forward(image, mask),
            #[cfg(all(feature = "vulkan-ncnn", target_os = "windows"))]
            LamaBackend::Ncnn(_) => bail!("ncnn tensors are handled by inference_model"),
            #[cfg(all(feature = "directml", target_os = "windows"))]
            LamaBackend::DirectMl(_) => bail!("DirectML tensors are handled by inference_model"),
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn inference_model(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        #[cfg(all(feature = "vulkan-ncnn", target_os = "windows"))]
        if let LamaBackend::Ncnn(ncnn) = &self.backend {
            return ncnn_inference(ncnn, image, mask);
        }
        #[cfg(all(feature = "directml", target_os = "windows"))]
        if let LamaBackend::DirectMl(session) = &self.backend {
            return directml_inference(session, image, mask);
        }
        let (image_tensor, mask_tensor) = self.preprocess(image, mask)?;
        let output = self.forward(&image_tensor, &mask_tensor)?;
        self.postprocess(&output)
    }

    #[instrument(level = "debug", skip_all)]
    fn preprocess(&self, image: &RgbImage, mask: &GrayImage) -> Result<(Tensor, Tensor)> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let rgb = image.clone().into_raw();
        let luma = mask.clone().into_raw();

        let image_tensor = (Tensor::from_vec(rgb, (1, h, w, 3), &self.device)?
            .permute((0, 3, 1, 2))?
            .to_dtype(DType::F32)?
            * (1. / 255.))?;

        let mask_tensor = Tensor::from_vec(luma, (1, h, w, 1), &self.device)?
            .permute((0, 3, 1, 2))?
            .to_dtype(DType::F32)?
            .gt(1.0f32)?;

        Ok((image_tensor, mask_tensor))
    }

    #[instrument(level = "debug", skip_all)]
    fn postprocess(&self, output: &Tensor) -> Result<RgbImage> {
        let output = output.squeeze(0)?;
        let (channels, height, width) = output.dims3()?;
        if channels != 3 {
            bail!("expected 3 channels in output, got {channels}");
        }
        let output = (output * 255.)?
            .clamp(0., 255.)?
            .permute((1, 2, 0))?
            .to_dtype(DType::U8)?;
        let raw: Vec<u8> = output.flatten_all()?.to_vec1()?;
        RgbImage::from_raw(width as u32, height as u32, raw)
            .ok_or_else(|| anyhow::anyhow!("failed to create image buffer from model output"))
    }
}

#[cfg(all(feature = "vulkan-ncnn", target_os = "windows"))]
fn ncnn_inference(
    ncnn: &crate::comic_text_detector::ncnn::NcnnDetector,
    image: &RgbImage,
    mask: &GrayImage,
) -> Result<RgbImage> {
    const SIZE: u32 = 256;
    let original_size = image.dimensions();
    let resized_image = if original_size == (SIZE, SIZE) {
        image.clone()
    } else {
        imageops::resize(image, SIZE, SIZE, imageops::FilterType::Triangle)
    };
    let resized_mask = if mask.dimensions() == (SIZE, SIZE) {
        mask.clone()
    } else {
        imageops::resize(mask, SIZE, SIZE, imageops::FilterType::Nearest)
    };
    let pixels = (SIZE * SIZE) as usize;
    let mut image_chw = vec![0.0f32; pixels * 3];
    for (index, pixel) in resized_image.pixels().enumerate() {
        image_chw[index] = f32::from(pixel[0]) / 255.0;
        image_chw[pixels + index] = f32::from(pixel[1]) / 255.0;
        image_chw[pixels * 2 + index] = f32::from(pixel[2]) / 255.0;
    }
    let mut mask_chw = resized_mask
        .pixels()
        .map(|pixel| if pixel[0] > 1 { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    let output = ncnn.forward_two_inputs(
        &mut image_chw,
        3,
        &mut mask_chw,
        1,
        SIZE as i32,
        SIZE as i32,
        "out0",
    )?;
    if output.len() != pixels * 3 {
        bail!("unexpected ncnn LaMa output size: {}", output.len());
    }
    let mut raw = vec![0u8; pixels * 3];
    for index in 0..pixels {
        raw[index * 3] = (output[index] * 255.0).clamp(0.0, 255.0) as u8;
        raw[index * 3 + 1] = (output[pixels + index] * 255.0).clamp(0.0, 255.0) as u8;
        raw[index * 3 + 2] = (output[pixels * 2 + index] * 255.0).clamp(0.0, 255.0) as u8;
    }
    let result = RgbImage::from_raw(SIZE, SIZE, raw)
        .ok_or_else(|| anyhow::anyhow!("failed to create ncnn LaMa output image"))?;
    if original_size == (SIZE, SIZE) {
        Ok(result)
    } else {
        Ok(imageops::resize(
            &result,
            original_size.0,
            original_size.1,
            imageops::FilterType::Triangle,
        ))
    }
}

#[cfg(all(feature = "directml", target_os = "windows"))]
fn directml_inference(
    session: &std::sync::Mutex<ort::session::Session>,
    image: &RgbImage,
    mask: &GrayImage,
) -> Result<RgbImage> {
    let (width, height) = image.dimensions();
    let pixels = (width as usize) * (height as usize);
    let mut image_chw = vec![0.0f32; pixels * 3];
    for (index, pixel) in image.pixels().enumerate() {
        image_chw[index] = f32::from(pixel[0]) / 255.0;
        image_chw[pixels + index] = f32::from(pixel[1]) / 255.0;
        image_chw[pixels * 2 + index] = f32::from(pixel[2]) / 255.0;
    }
    let mask_chw = mask
        .pixels()
        .map(|pixel| if pixel[0] > 1 { 1.0f32 } else { 0.0 })
        .collect::<Vec<_>>();

    let image_tensor =
        ort::value::Tensor::from_array(([1usize, 3, height as usize, width as usize], image_chw))?;
    let mask_tensor =
        ort::value::Tensor::from_array(([1usize, 1, height as usize, width as usize], mask_chw))?;
    let mut session = session
        .lock()
        .map_err(|_| anyhow::anyhow!("DirectML session lock is poisoned"))?;
    let outputs = session.run(ort::inputs!["image" => image_tensor, "mask" => mask_tensor])?;
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("LaMa DirectML output is not an f32 tensor")?;
    if &shape[..] != [1, 3, height as i64, width as i64].as_slice() {
        bail!("unexpected LaMa DirectML output shape: {shape:?}");
    }

    let mut raw = vec![0u8; pixels * 3];
    for index in 0..pixels {
        raw[index * 3] = (values[index] * 255.0).clamp(0.0, 255.0) as u8;
        raw[index * 3 + 1] = (values[pixels + index] * 255.0).clamp(0.0, 255.0) as u8;
        raw[index * 3 + 2] = (values[pixels * 2 + index] * 255.0).clamp(0.0, 255.0) as u8;
    }
    RgbImage::from_raw(width, height, raw)
        .ok_or_else(|| anyhow::anyhow!("failed to create DirectML output image"))
}

/// [`InpaintForward`] impl used by the HD-strategy dispatcher. Applies the
/// balloon-fill fast path on a per-crop basis before falling back to the
/// model forward — flat-background speech bubbles skip the model entirely.
struct LamaForward<'a> {
    lama: &'a Lama,
}

impl InpaintForward for LamaForward<'_> {
    fn forward(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        bubble_mask: Option<&GrayImage>,
    ) -> Result<RgbImage> {
        if mask.pixels().all(|p| p.0[0] == 0) {
            return Ok(image.clone());
        }

        let (image, mask) = if let Some(bubble_mask) = bubble_mask {
            let filled = apply_bubble_fill(image, mask, bubble_mask);
            tracing::debug!(
                filled_pixels = filled.filled_pixels,
                "lama bubble fill fast path"
            );
            (filled.image, filled.remaining_mask)
        } else {
            (image.clone(), mask.clone())
        };

        if mask.pixels().all(|p| p.0[0] == 0) {
            return Ok(image);
        }
        self.lama.inference_model(&image, &mask)
    }
}

fn crop_windows_from_text_blocks(text_blocks: &[TextRegion], width: u32, height: u32) -> Vec<Xyxy> {
    let mut windows = Vec::with_capacity(text_blocks.len());
    for block in text_blocks {
        let Some(block_box) = block_xyxy(block, width, height) else {
            continue;
        };
        let window = enlarge_window(
            block_box,
            width,
            height,
            BLOCK_WINDOW_RATIO,
            BLOCK_WINDOW_ASPECT_RATIO,
        );
        if window[2] > window[0] && window[3] > window[1] {
            windows.push(window);
        }
    }
    merge_overlapping_windows(windows)
}

fn block_xyxy(block: &TextRegion, width: u32, height: u32) -> Option<Xyxy> {
    let x1 = block.x.floor().max(0.0) as u32;
    let y1 = block.y.floor().max(0.0) as u32;
    let x2 = (block.x + block.width).ceil().max(block.x.floor()) as u32;
    let y2 = (block.y + block.height).ceil().max(block.y.floor()) as u32;

    let x1 = x1.min(width);
    let y1 = y1.min(height);
    let x2 = x2.min(width);
    let y2 = y2.min(height);

    if x2 <= x1 || y2 <= y1 {
        return None;
    }

    Some([x1, y1, x2, y2])
}

fn enlarge_window(rect: Xyxy, im_w: u32, im_h: u32, ratio: f64, aspect_ratio: f64) -> Xyxy {
    debug_assert!(ratio > 1.0);

    let [x1, y1, x2, y2] = rect;
    let w = f64::from(x2.saturating_sub(x1));
    let h = f64::from(y2.saturating_sub(y1));
    if w <= 0.0 || h <= 0.0 || aspect_ratio <= 0.0 {
        return [0, 0, 0, 0];
    }

    let a = aspect_ratio;
    let b = w + h * aspect_ratio;
    let c = (1.0 - ratio) * w * h;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let delta = ((-b + discriminant.sqrt()) / (2.0 * a) / 2.0).round();
    let mut delta_h = delta.max(0.0) as u32;
    let mut delta_w = (delta * aspect_ratio).round().max(0.0) as u32;

    delta_w = delta_w.min(x1).min(im_w.saturating_sub(x2));
    delta_h = delta_h.min(y1).min(im_h.saturating_sub(y2));

    [
        x1.saturating_sub(delta_w),
        y1.saturating_sub(delta_h),
        (x2 + delta_w).min(im_w),
        (y2 + delta_h).min(im_h),
    ]
}

fn merge_overlapping_windows(mut windows: Vec<Xyxy>) -> Vec<Xyxy> {
    windows.sort_by_key(|window| (window[0], window[1], window[2], window[3]));
    let mut merged = Vec::with_capacity(windows.len());
    for window in windows {
        merge_window_into(&mut merged, window);
    }
    merged.sort_by_key(|window| (window[0], window[1], window[2], window[3]));
    merged
}

fn merge_window_into(merged: &mut Vec<Xyxy>, mut window: Xyxy) {
    loop {
        let Some(index) = merged
            .iter()
            .position(|candidate| windows_touch_or_overlap(*candidate, window))
        else {
            merged.push(window);
            return;
        };
        window = union_xyxy(merged.swap_remove(index), window);
    }
}

fn windows_touch_or_overlap(a: Xyxy, b: Xyxy) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

fn union_xyxy(a: Xyxy, b: Xyxy) -> Xyxy {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

#[cfg(test)]
mod tests {
    use crate::inpainting::restore_alpha_channel;
    use crate::types::TextRegion;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    use super::{crop_windows_from_text_blocks, enlarge_window};

    const ALPHA_RING_RADIUS: u8 = 7;

    #[test]
    fn rgba_alpha_restore_uses_surrounding_ring() {
        let image = RgbImage::from_pixel(32, 32, Rgb([20, 30, 40]));
        let mut alpha = GrayImage::from_pixel(32, 32, Luma([255]));
        let mut mask = GrayImage::new(32, 32);

        for y in 10..22 {
            for x in 10..22 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        for y in (10 - u32::from(ALPHA_RING_RADIUS))..(22 + u32::from(ALPHA_RING_RADIUS)) {
            for x in (10 - u32::from(ALPHA_RING_RADIUS))..(22 + u32::from(ALPHA_RING_RADIUS)) {
                if x < 32 && y < 32 && mask.get_pixel(x, y).0[0] == 0 {
                    alpha.put_pixel(x, y, Luma([64]));
                }
            }
        }

        let restored = restore_alpha_channel(&image, &alpha, &mask);
        assert_eq!(restored.get_pixel(15, 15).0[3], 64);
        assert_eq!(restored.get_pixel(2, 2).0[3], 255);
    }

    #[test]
    fn enlarge_window_matches_ratio_1_7_reference() {
        let enlarged = enlarge_window([10, 20, 50, 60], 200, 150, 1.7, 1.0);
        assert_eq!(enlarged, [4, 14, 56, 66]);
    }

    #[test]
    fn crop_windows_merge_overlapping_text_blocks() {
        let windows = crop_windows_from_text_blocks(
            &[
                TextRegion {
                    x: 100.0,
                    y: 100.0,
                    width: 40.0,
                    height: 40.0,
                    ..TextRegion::default()
                },
                TextRegion {
                    x: 145.0,
                    y: 105.0,
                    width: 40.0,
                    height: 40.0,
                    ..TextRegion::default()
                },
            ],
            512,
            512,
        );

        assert_eq!(windows.len(), 1);
        assert!(windows[0][0] <= 100);
        assert!(windows[0][1] <= 100);
        assert!(windows[0][2] >= 185);
        assert!(windows[0][3] >= 145);
    }
}
