// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let ptx = CString::new(include_str!("./../kernel/kernel.ptx"))?;
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            Device::get_device(0)?,
        )?; // have to declare this in another line

        Ok(Self {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            module: Module::load_from_string(&ptx)?,
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
            _context,
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let module = &self.module;
        let stream = &self.stream;
        let mut conv_relu_input = DeviceBox::new(input)?;
        let mut conv_relu_output =
            DeviceBox::new(&[[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE])?;
        let mut ans = OutputVec([0.0; OUT_LAYER_SIZE]);
        let mut out_holder = OutputVec2([[0.0; OUT_LAYER_THREAD_NUM]; OUT_LAYER_SIZE]);
        let mut out_device_holder = DeviceBox::new(&out_holder)?;

        let conv_relu_grid_size = GridSize::x(OUT_LAYER_SIZE as u32);
        let conv_relu_block_size = BlockSize::xy(CONV_OUT_DIM as u32, CONV_OUT_DIM as u32);
        let output_grid_size = GridSize::x(OUT_LAYER_SIZE as u32);
        let output_block_size = BlockSize::x(OUT_LAYER_THREAD_NUM as u32);

        unsafe {
            let _ = launch!(module.convolution_relu_layer<<<conv_relu_grid_size, conv_relu_block_size, 0, stream>>>(
                conv_relu_input.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_relu_output.as_device_ptr()
            ));

            let _ = launch!(module.output_layer<<<output_grid_size, output_block_size, 0, stream>>>(
                conv_relu_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                out_device_holder.as_device_ptr()
            ));
        }

        stream.synchronize()?;
        out_device_holder.copy_to(&mut out_holder)?;
        for i in 0..OUT_LAYER_SIZE {
            for j in 0..OUT_LAYER_THREAD_NUM {
                ans.0[i] += out_holder.0[i][j];
            }
        }

        Ok(ans)
    }
}
