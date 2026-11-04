pub mod mnn_generated;

pub use mnn_generated::mnn::*;

/// Parse root Net from byte buffer
pub fn parse_net<'a>(buf: &'a [u8]) -> Net<'a> {
    get_root_as_net(buf)
}
