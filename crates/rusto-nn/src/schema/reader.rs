use std::marker::PhantomData;

/// Core trait for types that can be decoded from little-endian FlatBuffers wire bytes.
pub trait Scalar: Copy {
    const WIRE_SIZE: usize = std::mem::size_of::<Self>();
    fn to_bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;
}

impl Scalar for bool {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        (bits as u8) != 0
    }
}

impl Scalar for u8 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u8
    }
}

impl Scalar for i8 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u8 as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u8 as i8
    }
}

impl Scalar for u16 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u16
    }
}

impl Scalar for i16 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u16 as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u16 as i16
    }
}

impl Scalar for u32 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u32
    }
}

impl Scalar for i32 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u32 as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as u32 as i32
    }
}

impl Scalar for u64 {
    #[inline]
    fn to_bits(self) -> u64 {
        self
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits
    }
}

impl Scalar for i64 {
    #[inline]
    fn to_bits(self) -> u64 {
        self as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        bits as i64
    }
}

impl Scalar for f32 {
    #[inline]
    fn to_bits(self) -> u64 {
        self.to_bits() as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
}

impl Scalar for f64 {
    #[inline]
    fn to_bits(self) -> u64 {
        self.to_bits()
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
}

/// A FlatBuffers table view referencing a slice and a table start position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Table<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) pos: usize,
}

pub trait FromTable<'a>: Sized {
    fn from_table(table: Table<'a>) -> Self;
}

impl<'a> FromTable<'a> for Table<'a> {
    #[inline]
    fn from_table(table: Table<'a>) -> Self {
        table
    }
}

fn read_u16_le(buf: &[u8], pos: usize) -> Option<u16> {
    let end = pos.checked_add(2)?;
    let s = buf.get(pos..end)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32_le(buf: &[u8], pos: usize) -> Option<u32> {
    let end = pos.checked_add(4)?;
    let s = buf.get(pos..end)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_le_bits(s: &[u8]) -> u64 {
    match s.len() {
        1 => s[0] as u64,
        2 => u16::from_le_bytes([s[0], s[1]]) as u64,
        4 => u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as u64,
        8 => u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
        _ => 0,
    }
}

fn read_str_at(buf: &[u8], str_pos: usize) -> Option<&str> {
    let len = read_u32_le(buf, str_pos)? as usize;
    let start = str_pos.checked_add(4)?;
    let end = start.checked_add(len)?;
    std::str::from_utf8(buf.get(start..end)?).ok()
}

fn aligned_slice_len(data: &[u8], elem_size: usize, align: usize) -> usize {
    if data.is_empty() {
        return 0;
    }
    assert!((data.as_ptr() as usize) % align == 0, "Vector memory unaligned");
    data.len() / elem_size
}

fn read_slice_elem_bits(data: &[u8], idx: usize, elem_size: usize) -> u64 {
    let len = data.len() / elem_size;
    assert!(idx < len);
    let start = idx * elem_size;
    read_le_bits(&data[start..start + elem_size])
}

impl<'a> Table<'a> {
    #[inline]
    pub fn new(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    /// Resolve field data position from vtable offset.
    #[inline(never)]
    pub fn get_field_offset(&self, vt_offset: usize) -> Option<usize> {
        let soffset = read_u32_le(self.buf, self.pos)? as i32;
        let vtable_pos = if soffset >= 0 {
            self.pos.checked_sub(soffset as usize)?
        } else {
            self.pos.checked_add(soffset.unsigned_abs() as usize)?
        };
        let vtable_len = read_u16_le(self.buf, vtable_pos)? as usize;
        let vt_slot_end = vt_offset.checked_add(2)?;
        if vt_slot_end > vtable_len {
            return None;
        }
        let field_offset = read_u16_le(self.buf, vtable_pos.checked_add(vt_offset)?)? as usize;
        if field_offset == 0 {
            return None;
        }
        let field_pos = self.pos.checked_add(field_offset)?;
        if field_pos <= self.buf.len() {
            Some(field_pos)
        } else {
            None
        }
    }

    #[inline(never)]
    fn resolve_target(&self, vt_offset: usize) -> Option<usize> {
        let field_pos = self.get_field_offset(vt_offset)?;
        let uoffset = read_u32_le(self.buf, field_pos)? as usize;
        let target_pos = field_pos.checked_add(uoffset)?;
        if target_pos <= self.buf.len() {
            Some(target_pos)
        } else {
            None
        }
    }

    #[inline(never)]
    fn read_vec_header(&self, vt_offset: usize, elem_size: usize) -> Option<(usize, usize)> {
        let vec_pos = self.resolve_target(vt_offset)?;
        let len = read_u32_le(self.buf, vec_pos)? as usize;
        let data_pos = vec_pos.checked_add(4)?;
        let byte_len = len.checked_mul(elem_size)?;
        let end = data_pos.checked_add(byte_len)?;
        if end <= self.buf.len() {
            Some((data_pos, len))
        } else {
            None
        }
    }

    #[inline(never)]
    fn read_vec_slice(&self, vt_offset: usize, elem_size: usize) -> Option<&'a [u8]> {
        let (data_pos, len) = self.read_vec_header(vt_offset, elem_size)?;
        self.buf.get(data_pos..data_pos + len * elem_size)
    }

    #[inline(never)]
    fn get_bits(&self, vt_offset: usize, size: usize, default_bits: u64) -> u64 {
        let Some(pos) = self.get_field_offset(vt_offset) else {
            return default_bits;
        };
        let Some(end) = pos.checked_add(size) else {
            return default_bits;
        };
        let Some(s) = self.buf.get(pos..end) else {
            return default_bits;
        };
        read_le_bits(s)
    }

    #[inline]
    pub fn get_scalar<T: Scalar>(&self, vt_offset: usize, default: T) -> T {
        T::from_bits(self.get_bits(vt_offset, T::WIRE_SIZE, default.to_bits()))
    }

    #[inline]
    pub fn get_table<T: FromTable<'a>>(&self, vt_offset: usize) -> Option<T> {
        let table_pos = self.resolve_target(vt_offset)?;
        Some(T::from_table(Table::new(self.buf, table_pos)))
    }

    #[inline(never)]
    pub fn get_string(&self, vt_offset: usize) -> Option<&'a str> {
        let str_pos = self.resolve_target(vt_offset)?;
        read_str_at(self.buf, str_pos)
    }

    #[inline]
    pub fn get_scalar_vec<T: Scalar>(&self, vt_offset: usize) -> Option<ScalarVec<'a, T>> {
        let data = self.read_vec_slice(vt_offset, T::WIRE_SIZE)?;
        Some(ScalarVec {
            data,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn get_byte_slice(&self, vt_offset: usize) -> Option<&'a [u8]> {
        self.read_vec_slice(vt_offset, 1)
    }

    #[inline]
    pub fn get_i8_slice(&self, vt_offset: usize) -> Option<&'a [i8]> {
        let bytes = self.get_byte_slice(vt_offset)?;
        // SAFETY: i8 and u8 have identical size (1) and alignment (1).
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) })
    }

    #[inline]
    pub fn get_table_vec<T: FromTable<'a>>(&self, vt_offset: usize) -> Option<TableVec<'a, T>> {
        let (data_pos, len) = self.read_vec_header(vt_offset, 4)?;
        Some(TableVec {
            vec: OffsetVec {
                buf: self.buf,
                data_pos,
                len,
            },
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn get_string_vec(&self, vt_offset: usize) -> Option<StringVec<'a>> {
        let (data_pos, len) = self.read_vec_header(vt_offset, 4)?;
        Some(StringVec {
            vec: OffsetVec {
                buf: self.buf,
                data_pos,
                len,
            },
        })
    }
}

/// Zero-copy vector of primitive scalar elements.
#[derive(Clone, Copy)]
pub struct ScalarVec<'a, T: Scalar> {
    data: &'a [u8],
    _marker: PhantomData<T>,
}

impl<'a, T: Scalar> ScalarVec<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() / T::WIRE_SIZE
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> T {
        T::from_bits(read_slice_elem_bits(self.data, idx, T::WIRE_SIZE))
    }

    #[inline]
    pub fn safe_slice(&self) -> &'a [T] {
        let len = aligned_slice_len(self.data, T::WIRE_SIZE, std::mem::align_of::<T>());
        if len == 0 {
            return &[];
        }
        // SAFETY: `read_vec_slice` bounds `self.data` to a multiple of `WIRE_SIZE`,
        // `aligned_slice_len` asserts `T`'s alignment, and callers only form `ScalarVec` for POD scalars.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, len) }
    }

    #[inline]
    pub fn iter(&self) -> ScalarVecIter<'a, T> {
        ScalarVecIter {
            chunks: self.data.chunks_exact(T::WIRE_SIZE),
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Scalar> IntoIterator for ScalarVec<'a, T> {
    type Item = T;
    type IntoIter = ScalarVecIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct ScalarVecIter<'a, T: Scalar> {
    chunks: std::slice::ChunksExact<'a, u8>,
    _marker: PhantomData<T>,
}

impl<'a, T: Scalar> Iterator for ScalarVecIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.chunks.next().map(|c| T::from_bits(read_le_bits(c)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl<'a, T: Scalar> ExactSizeIterator for ScalarVecIter<'a, T> {}

#[derive(Clone, Copy)]
struct OffsetVec<'a> {
    buf: &'a [u8],
    data_pos: usize,
    len: usize,
}

impl<'a> OffsetVec<'a> {
    #[inline(never)]
    fn get_target(&self, idx: usize) -> Option<usize> {
        assert!(idx < self.len);
        let elem_pos = self.data_pos + idx * 4;
        let uoffset = read_u32_le(self.buf, elem_pos)? as usize;
        let target_pos = elem_pos.checked_add(uoffset)?;
        if target_pos <= self.buf.len() {
            Some(target_pos)
        } else {
            None
        }
    }
}

/// Zero-copy vector of FlatBuffers tables.
#[derive(Clone, Copy)]
pub struct TableVec<'a, T> {
    vec: OffsetVec<'a>,
    _marker: PhantomData<T>,
}

impl<'a, T: FromTable<'a>> TableVec<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vec.len == 0
    }

    #[inline]
    pub fn get(&self, idx: usize) -> T {
        let pos = self.vec.get_target(idx).unwrap_or(self.vec.buf.len());
        T::from_table(Table::new(self.vec.buf, pos))
    }
}

/// Zero-copy vector of FlatBuffers strings.
#[derive(Clone, Copy)]
pub struct StringVec<'a> {
    vec: OffsetVec<'a>,
}

impl<'a> StringVec<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vec.len == 0
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &'a str {
        self.vec
            .get_target(idx)
            .and_then(|p| read_str_at(self.vec.buf, p))
            .unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malformed_and_truncated_buffers_do_not_panic() {
        let empty: &[u8] = &[];
        let t = Table::new(empty, 0);
        assert_eq!(t.get_scalar::<i32>(4, 42), 42);
        assert_eq!(t.get_string(4), None);
        assert!(t.get_scalar_vec::<f32>(4).is_none());
        assert!(t.get_table::<Table>(4).is_none());

        let neg_vtable = [100u8, 0, 0, 0, 0, 0, 0, 0];
        let t_neg = Table::new(&neg_vtable, 0);
        assert_eq!(t_neg.get_scalar::<i32>(4, -1), -1);
        assert!(t_neg.get_string(4).is_none());

        let fwd_oob = [0u8, 0, 0, 0x80, 0, 0, 0, 0];
        let t_oob = Table::new(&fwd_oob, 0);
        assert_eq!(t_oob.get_scalar::<u32>(4, 99), 99);
    }

    #[test]
    fn test_valid_flatbuffer_table_fields() {
        #[repr(align(4))]
        struct AlignedBuf([u8; 60]);
        let mut buf = AlignedBuf([0u8; 60]);
        let b = &mut buf.0;

        b[0..4].copy_from_slice(&16u32.to_le_bytes());
        b[4..6].copy_from_slice(&12u16.to_le_bytes());
        b[6..8].copy_from_slice(&16u16.to_le_bytes());
        b[8..10].copy_from_slice(&4u16.to_le_bytes());
        b[10..12].copy_from_slice(&8u16.to_le_bytes());
        b[12..14].copy_from_slice(&12u16.to_le_bytes());

        b[16..20].copy_from_slice(&12i32.to_le_bytes());
        b[20..24].copy_from_slice(&1234i32.to_le_bytes());
        b[24..28].copy_from_slice(&12u32.to_le_bytes());
        b[28..32].copy_from_slice(&20u32.to_le_bytes());

        b[36..40].copy_from_slice(&5u32.to_le_bytes());
        b[40..45].copy_from_slice(b"hello");

        b[48..52].copy_from_slice(&2u32.to_le_bytes());
        b[52..56].copy_from_slice(&1.5f32.to_le_bytes());
        b[56..60].copy_from_slice(&2.5f32.to_le_bytes());

        let root_off = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let tab = Table::new(b, root_off);

        assert_eq!(tab.get_scalar::<i32>(4, 0), 1234);
        assert_eq!(tab.get_scalar::<i32>(12, 999), 999);
        assert_eq!(tab.get_string(6), Some("hello"));
        let vec = tab.get_scalar_vec::<f32>(8).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), 1.5);
        assert_eq!(vec.get(1), 2.5);
        assert_eq!(vec.safe_slice(), &[1.5, 2.5]);
        assert_eq!(vec.iter().collect::<Vec<f32>>(), vec![1.5, 2.5]);
    }
}
