use std::{
    ffi::{CString, c_void},
    path::{Path, PathBuf},
    ptr,
};

use anyhow::{Context, Result, bail};
use libloading::Library;

type Handle = *mut c_void;

pub struct NcnnDetector {
    api: Api,
    net: Handle,
}

unsafe impl Send for NcnnDetector {}
unsafe impl Sync for NcnnDetector {}

struct Api {
    net_destroy: unsafe extern "C" fn(Handle),
    extractor_create: unsafe extern "C" fn(Handle) -> Handle,
    extractor_destroy: unsafe extern "C" fn(Handle),
    extractor_input: unsafe extern "C" fn(Handle, *const i8, Handle) -> i32,
    extractor_extract: unsafe extern "C" fn(Handle, *const i8, *mut Handle) -> i32,
    mat_create_external_3d: unsafe extern "C" fn(i32, i32, i32, *mut c_void, Handle) -> Handle,
    mat_create: unsafe extern "C" fn() -> Handle,
    mat_destroy: unsafe extern "C" fn(Handle),
    mat_w: unsafe extern "C" fn(Handle) -> i32,
    mat_h: unsafe extern "C" fn(Handle) -> i32,
    mat_c: unsafe extern "C" fn(Handle) -> i32,
    mat_channel: unsafe extern "C" fn(Handle, i32) -> *mut c_void,
    _lib: Library,
}

impl NcnnDetector {
    pub fn load() -> Result<Self> {
        Self::load_model("comictextdetector", "comic text detection")
    }

    pub(crate) fn load_model(stem: &str, label: &str) -> Result<Self> {
        let root = std::env::var_os("KOHARU_NCNN_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_WORKSPACE_DIR")).join("temp/vulkan-pilot")
            });
        let dll = root.join("ncnn.dll");
        let param = root.join(format!("{stem}.ncnn.param"));
        let bin = root.join(format!("{stem}.ncnn.bin"));
        for path in [&dll, &param, &bin] {
            if !path.is_file() {
                bail!("missing ncnn pilot artifact: {}", path.display());
            }
        }
        let result = unsafe { Self::load_inner(&dll, &param, &bin) };
        if result.is_ok() {
            tracing::info!(backend = "ncnn-vulkan", engine = label, "using GPU backend");
        }
        result
    }

    unsafe fn load_inner(dll: &Path, param: &Path, bin: &Path) -> Result<Self> {
        let lib =
            unsafe { Library::new(dll) }.with_context(|| format!("load {}", dll.display()))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) }?
            };
        }
        let net_create = sym!("ncnn_net_create", unsafe extern "C" fn() -> Handle);
        let net_set_option = sym!("ncnn_net_set_option", unsafe extern "C" fn(Handle, Handle));
        let net_set_device = sym!(
            "ncnn_net_set_vulkan_device",
            unsafe extern "C" fn(Handle, i32)
        );
        let load_param = sym!(
            "ncnn_net_load_param",
            unsafe extern "C" fn(Handle, *const i8) -> i32
        );
        let load_model = sym!(
            "ncnn_net_load_model",
            unsafe extern "C" fn(Handle, *const i8) -> i32
        );
        let option_create = sym!("ncnn_option_create", unsafe extern "C" fn() -> Handle);
        let option_destroy = sym!("ncnn_option_destroy", unsafe extern "C" fn(Handle));
        let option_vulkan = sym!(
            "ncnn_option_set_use_vulkan_compute",
            unsafe extern "C" fn(Handle, i32)
        );
        let option_fp16 = sym!(
            "ncnn_option_set_use_fp16_storage",
            unsafe extern "C" fn(Handle, i32)
        );
        let net = unsafe { net_create() };
        let opt = unsafe { option_create() };
        unsafe {
            option_vulkan(opt, 1);
            option_fp16(opt, 1);
            net_set_option(net, opt);
            net_set_device(net, 0);
            option_destroy(opt);
        }
        let p = CString::new(param.to_string_lossy().as_bytes())?;
        let b = CString::new(bin.to_string_lossy().as_bytes())?;
        if unsafe { load_param(net, p.as_ptr()) } != 0
            || unsafe { load_model(net, b.as_ptr()) } != 0
        {
            bail!("ncnn model load failed");
        }
        let api = Api {
            net_destroy: sym!("ncnn_net_destroy", unsafe extern "C" fn(Handle)),
            extractor_create: sym!(
                "ncnn_extractor_create",
                unsafe extern "C" fn(Handle) -> Handle
            ),
            extractor_destroy: sym!("ncnn_extractor_destroy", unsafe extern "C" fn(Handle)),
            extractor_input: sym!(
                "ncnn_extractor_input",
                unsafe extern "C" fn(Handle, *const i8, Handle) -> i32
            ),
            extractor_extract: sym!(
                "ncnn_extractor_extract",
                unsafe extern "C" fn(Handle, *const i8, *mut Handle) -> i32
            ),
            mat_create_external_3d: sym!(
                "ncnn_mat_create_external_3d",
                unsafe extern "C" fn(i32, i32, i32, *mut c_void, Handle) -> Handle
            ),
            mat_create: sym!("ncnn_mat_create", unsafe extern "C" fn() -> Handle),
            mat_destroy: sym!("ncnn_mat_destroy", unsafe extern "C" fn(Handle)),
            mat_w: sym!("ncnn_mat_get_w", unsafe extern "C" fn(Handle) -> i32),
            mat_h: sym!("ncnn_mat_get_h", unsafe extern "C" fn(Handle) -> i32),
            mat_c: sym!("ncnn_mat_get_c", unsafe extern "C" fn(Handle) -> i32),
            mat_channel: sym!(
                "ncnn_mat_get_channel_data",
                unsafe extern "C" fn(Handle, i32) -> *mut c_void
            ),
            _lib: lib,
        };
        Ok(Self { api, net })
    }

    pub fn forward(&self, input: &mut [f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let outputs = self.forward_outputs(input, 1024, 1024, &["out0", "out1", "out2"])?;
        Ok((outputs[0].clone(), outputs[1].clone(), outputs[2].clone()))
    }

    pub(crate) fn forward_outputs(
        &self,
        input: &mut [f32],
        width: i32,
        height: i32,
        output_names: &[&str],
    ) -> Result<Vec<Vec<f32>>> {
        let ex = unsafe { (self.api.extractor_create)(self.net) };
        let mat = unsafe {
            (self.api.mat_create_external_3d)(
                width,
                height,
                3,
                input.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
        };
        let name = CString::new("in0")?;
        if unsafe { (self.api.extractor_input)(ex, name.as_ptr(), mat) } != 0 {
            bail!("ncnn input failed");
        }
        let result = output_names
            .iter()
            .map(|name| self.extract(ex, name))
            .collect::<Result<Vec<_>>>();
        unsafe {
            (self.api.mat_destroy)(mat);
            (self.api.extractor_destroy)(ex);
        }
        result
    }

    pub(crate) fn forward_two_inputs(
        &self,
        first: &mut [f32],
        first_channels: i32,
        second: &mut [f32],
        second_channels: i32,
        width: i32,
        height: i32,
        output_name: &str,
    ) -> Result<Vec<f32>> {
        let ex = unsafe { (self.api.extractor_create)(self.net) };
        let first_mat = unsafe {
            (self.api.mat_create_external_3d)(
                width,
                height,
                first_channels,
                first.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
        };
        let second_mat = unsafe {
            (self.api.mat_create_external_3d)(
                width,
                height,
                second_channels,
                second.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
        };
        let first_name = CString::new("in0")?;
        let second_name = CString::new("in1")?;
        let result = if unsafe { (self.api.extractor_input)(ex, first_name.as_ptr(), first_mat) }
            != 0
            || unsafe { (self.api.extractor_input)(ex, second_name.as_ptr(), second_mat) } != 0
        {
            Err(anyhow::anyhow!("ncnn input failed"))
        } else {
            self.extract(ex, output_name)
        };
        unsafe {
            (self.api.mat_destroy)(first_mat);
            (self.api.mat_destroy)(second_mat);
            (self.api.extractor_destroy)(ex);
        }
        result
    }

    fn extract(&self, ex: Handle, name: &str) -> Result<Vec<f32>> {
        let mut mat = unsafe { (self.api.mat_create)() };
        let output_name = name;
        let c_name = CString::new(name)?;
        if unsafe { (self.api.extractor_extract)(ex, c_name.as_ptr(), &mut mat) } != 0 {
            bail!("ncnn extract failed");
        }
        let (w, h, c) = unsafe {
            (
                (self.api.mat_w)(mat) as usize,
                (self.api.mat_h)(mat) as usize,
                (self.api.mat_c)(mat) as usize,
            )
        };
        tracing::debug!(name = output_name, w, h, c, "ncnn output shape");
        let channel_len = w * h;
        let mut out = Vec::with_capacity(channel_len * c);
        for ch in 0..c {
            let p = unsafe { (self.api.mat_channel)(mat, ch as i32) as *const f32 };
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(p, channel_len) });
        }
        unsafe {
            (self.api.mat_destroy)(mat);
        }
        Ok(out)
    }
}

impl Drop for NcnnDetector {
    fn drop(&mut self) {
        unsafe { (self.api.net_destroy)(self.net) }
    }
}
