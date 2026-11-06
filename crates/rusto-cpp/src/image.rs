//! Image abstraction layer supporting OpenCV backend

#[cfg(feature = "opencv")]
use std::path::Path;
#[cfg(feature = "opencv")]
use rusto_core::image::{Point2f, Rect, Size};
#[cfg(feature = "opencv")]
pub use opencv::core::Mat;
#[cfg(feature = "opencv")]
pub use opencv::core::{rotate as cv_rotate, BorderTypes, RotateFlags};
#[cfg(feature = "opencv")]
pub use opencv::imgcodecs::{imread as cv_imread, imwrite as cv_imwrite, IMREAD_COLOR};
#[cfg(feature = "opencv")]
pub use opencv::imgproc::{
    get_perspective_transform, resize as cv_resize, warp_perspective as cv_warp_perspective,
    INTER_CUBIC, INTER_LINEAR,
};

#[cfg(feature = "opencv")]
pub const BORDER_REPLICATE: i32 = BorderTypes::BORDER_REPLICATE as i32;
#[cfg(feature = "opencv")]
pub const ROTATE_90_CLOCKWISE: i32 = RotateFlags::ROTATE_90_CLOCKWISE as i32;

#[cfg(feature = "opencv")]
pub type Result<T> = opencv::Result<T>;

#[cfg(feature = "opencv")]
pub fn imread<P: AsRef<Path>>(path: P) -> Result<Mat> {
    cv_imread(path.as_ref().to_str().unwrap(), IMREAD_COLOR)
}

#[cfg(feature = "opencv")]
pub fn imwrite<P: AsRef<Path>>(path: P, img: &Mat) -> Result<()> {
    cv_imwrite(
        path.as_ref().to_str().unwrap(),
        img,
        &opencv::core::Vector::new(),
    )?;
    Ok(())
}

#[cfg(feature = "opencv")]
pub fn crop(img: &Mat, rect: Rect) -> Result<Mat> {
    let roi = opencv::core::Rect::new(rect.x, rect.y, rect.width as i32, rect.height as i32);
    let cropped = Mat::roi(img, roi)?;
    let mut dst = Mat::default();
    cropped.copy_to(&mut dst)?;
    Ok(dst)
}

#[cfg(feature = "opencv")]
pub fn resize(src: &Mat, dst: &mut Mat, dsize: Size, interpolation: i32) -> Result<()> {
    cv_resize(
        src,
        dst,
        opencv::core::Size::new(dsize.width, dsize.height),
        0.0,
        0.0,
        interpolation,
    )
}

#[cfg(feature = "opencv")]
pub fn rotate(src: &Mat, dst: &mut Mat, rotation: i32) -> Result<()> {
    cv_rotate(src, dst, rotation)
}

#[cfg(feature = "opencv")]
pub fn warp_perspective(
    src: &Mat,
    dst: &mut Mat,
    matrix: &Mat,
    dsize: Size,
    flags: i32,
    border_mode: i32,
) -> Result<()> {
    cv_warp_perspective(
        src,
        dst,
        matrix,
        opencv::core::Size::new(dsize.width, dsize.height),
        flags,
        border_mode,
        opencv::core::Scalar::all(0.0),
    )
}

#[cfg(feature = "opencv")]
pub fn min_area_rect(contour: &[Point2f]) -> Result<(Point2f, Size, f32)> {
    let cv_contour: opencv::core::Vector<opencv::core::Point> = contour
        .iter()
        .map(|p| opencv::core::Point::new(p.x as i32, p.y as i32))
        .collect();

    let rect = opencv::imgproc::min_area_rect(&cv_contour)?;
    let center = Point2f::new(rect.center.x, rect.center.y);
    let size = Size::new(rect.size.width as i32, rect.size.height as i32);
    let angle = rect.angle;

    Ok((center, size, angle))
}

#[cfg(feature = "opencv")]
pub fn box_points(center: Point2f, size: Size, angle: f32) -> [Point2f; 4] {
    let rect = opencv::core::RotatedRect {
        center: opencv::core::Point2f::new(center.x, center.y),
        size: opencv::core::Size2f::new(size.width as f32, size.height as f32),
        angle,
    };

    let mut pts = [opencv::core::Point2f::default(); 4];
    rect.points(&mut pts).unwrap();

    [
        Point2f::new(pts[0].x, pts[0].y),
        Point2f::new(pts[1].x, pts[1].y),
        Point2f::new(pts[2].x, pts[2].y),
        Point2f::new(pts[3].x, pts[3].y),
    ]
}

