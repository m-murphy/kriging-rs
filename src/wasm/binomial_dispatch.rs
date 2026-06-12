//! Shared dispatch for calibrated binomial WASM adapters (notes, diagnostics, instance CV).

use wasm_bindgen::prelude::*;

use super::model_handle::KrigingModelInner;
use super::spacetime::WasmSpaceTimeBinomialKriging;
use super::{WasmBinomialKriging, WasmBinomialProjectedKriging, WasmBinomialTangentPlaneKriging};

pub(crate) enum BinomialAdapterRef<'a> {
    Geo(&'a WasmBinomialKriging),
    Projected(&'a WasmBinomialProjectedKriging),
    Tangent(&'a WasmBinomialTangentPlaneKriging),
    SpaceTime(&'a WasmSpaceTimeBinomialKriging),
}

impl<'a> BinomialAdapterRef<'a> {
    pub fn from_model_inner(inner: &'a KrigingModelInner) -> Option<Self> {
        match inner {
            KrigingModelInner::BinomialGeo(m) => Some(Self::Geo(m)),
            KrigingModelInner::BinomialProjected(m) => Some(Self::Projected(m)),
            KrigingModelInner::BinomialTangentPlane(m) => Some(Self::Tangent(m)),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => Some(Self::SpaceTime(m)),
            _ => None,
        }
    }

    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        match self {
            Self::Geo(m) => m.get_build_notes(),
            Self::Projected(m) => m.get_build_notes(),
            Self::Tangent(m) => m.get_build_notes(),
            Self::SpaceTime(m) => m.get_build_notes(),
        }
    }

    pub fn get_diagnostics(&self, options: JsValue) -> Result<JsValue, JsValue> {
        match self {
            Self::Geo(m) => m.get_diagnostics(options),
            Self::Projected(m) => m.get_diagnostics(options),
            Self::Tangent(m) => m.get_diagnostics(options),
            Self::SpaceTime(m) => m.get_diagnostics(options),
        }
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        match self {
            Self::Geo(m) => m.leave_one_out(),
            Self::Projected(m) => m.leave_one_out(),
            Self::Tangent(m) => m.leave_one_out(),
            Self::SpaceTime(m) => m.leave_one_out(),
        }
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        match self {
            Self::Geo(m) => m.k_fold(k),
            Self::Projected(m) => m.k_fold(k),
            Self::Tangent(m) => m.k_fold(k),
            Self::SpaceTime(m) => m.k_fold(k),
        }
    }
}
