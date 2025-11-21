//! Image abstraction layer supporting both pure Rust and OpenCV backends

#[cfg(feature = "use-opencv")]
pub use opencv_impl::*;

#[cfg(not(feature = "use-opencv"))]
pub use rust_impl::*;

// Common types and traits
#[derive(Debug, Clone, Copy)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

impl Default for Point2f {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Point2f {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
}

// Pure Rust implementation
#[cfg(not(feature = "use-opencv"))]
mod rust_impl {
    use super::{Point2f, Size};
    use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
    use std::path::Path;

    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    #[derive(Clone)]
    pub struct Mat {
        image: DynamicImage,
    }

    impl Default for Mat {
        fn default() -> Self {
            Self {
                image: DynamicImage::new_rgb8(1, 1),
            }
        }
    }

    impl Mat {
        pub fn new(image: DynamicImage) -> Self {
            Self { image }
        }

        pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
            let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data)
                .ok_or("Failed to create image from raw data")?;
            Ok(Self {
                image: DynamicImage::ImageRgb8(img),
            })
        }

        pub fn rows(&self) -> i32 {
            self.image.height() as i32
        }

        pub fn cols(&self) -> i32 {
            self.image.width() as i32
        }

        pub fn size(&self) -> Result<Size> {
            Ok(Size::new(self.cols(), self.rows()))
        }

        pub fn empty(&self) -> bool {
            self.image.width() == 0 || self.image.height() == 0
        }

        pub fn clone(&self) -> Self {
            Self {
                image: self.image.clone(),
            }
        }

        pub fn to_rgb8(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
            self.image.to_rgb8()
        }

        pub fn as_dynamic(&self) -> &DynamicImage {
            &self.image
        }

        pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 3] {
            let pixel = self.image.get_pixel(x, y);
            [pixel[0], pixel[1], pixel[2]]
        }
    }

    pub fn imread<P: AsRef<Path>>(path: P) -> Result<Mat> {
        let img = image::open(path)?;
        Ok(Mat::new(img))
    }

    #[allow(dead_code)]
    pub fn imwrite<P: AsRef<Path>>(path: P, img: &Mat) -> Result<()> {
        img.image.save(path)?;
        Ok(())
    }

    pub fn resize(
        src: &Mat,
        dst: &mut Mat,
        dsize: Size,
        interpolation: i32,
    ) -> Result<()> {
        // Match OpenCV's interpolation modes
        let filter = match interpolation {
            1 => image::imageops::FilterType::Triangle,     // INTER_LINEAR (bilinear)
            2 => image::imageops::FilterType::CatmullRom,   // INTER_CUBIC
            _ => image::imageops::FilterType::Triangle,      // Default to bilinear
        };
        
        let resized = src
            .image
            .resize_exact(dsize.width as u32, dsize.height as u32, filter);
        *dst = Mat::new(resized);
        Ok(())
    }

    pub fn rotate(src: &Mat, dst: &mut Mat, rotation: i32) -> Result<()> {
        const ROTATE_90_CLOCKWISE: i32 = 0;
        const ROTATE_180: i32 = 1;
        const ROTATE_90_COUNTERCLOCKWISE: i32 = 2;

        let rotated = match rotation {
            ROTATE_90_CLOCKWISE => src.image.rotate90(),
            ROTATE_180 => src.image.rotate180(),
            ROTATE_90_COUNTERCLOCKWISE => src.image.rotate270(),
            _ => return Err("Invalid rotation code".into()),
        };
        *dst = Mat::new(rotated);
        Ok(())
    }

    pub fn warp_perspective(
        src: &Mat,
        dst: &mut Mat,
        matrix: &[[f64; 3]; 3],
        dsize: Size,
        _flags: i32,
        _border_mode: i32,
    ) -> Result<()> {
        // Create output image
        let mut out_img = ImageBuffer::new(dsize.width as u32, dsize.height as u32);
        let src_img = src.to_rgb8();

        // Invert the transformation matrix for reverse mapping
        let m_inv = invert_matrix_3x3(matrix)?;

        // Pre-fetch matrix elements for better performance
        let (m00, m01, m02) = (m_inv[0][0], m_inv[0][1], m_inv[0][2]);
        let (m10, m11, m12) = (m_inv[1][0], m_inv[1][1], m_inv[1][2]);
        let (m20, m21, m22) = (m_inv[2][0], m_inv[2][1], m_inv[2][2]);
        let src_cols = src.cols();
        let src_rows = src.rows();
        
        for y in 0..dsize.height as u32 {
            let y_f = y as f64;
            // Pre-compute y-dependent terms
            let m01y = m01 * y_f;
            let m11y = m11 * y_f;
            let m21y = m21 * y_f;
            
            for x in 0..dsize.width as u32 {
                // Apply inverse transform with homogeneous coordinates
                let x_f = x as f64;
                let src_x_h = m00 * x_f + m01y + m02;
                let src_y_h = m10 * x_f + m11y + m12;
                let w = m20 * x_f + m21y + m22;

                let src_x_f = src_x_h / w;
                let src_y_f = src_y_h / w;

                // Bilinear interpolation (matches OpenCV's default INTER_LINEAR)
                let x0 = src_x_f.floor() as i32;
                let y0 = src_y_f.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                // Check bounds for all 4 corners
                if x0 >= 0 && x1 < src_cols && y0 >= 0 && y1 < src_rows {
                    let fx = src_x_f - x0 as f64;
                    let fy = src_y_f - y0 as f64;

                    let p00 = src_img.get_pixel(x0 as u32, y0 as u32);
                    let p10 = src_img.get_pixel(x1 as u32, y0 as u32);
                    let p01 = src_img.get_pixel(x0 as u32, y1 as u32);
                    let p11 = src_img.get_pixel(x1 as u32, y1 as u32);

                    // Bilinear interpolation for each channel
                    let r = ((1.0 - fx) * (1.0 - fy) * p00[0] as f64
                        + fx * (1.0 - fy) * p10[0] as f64
                        + (1.0 - fx) * fy * p01[0] as f64
                        + fx * fy * p11[0] as f64) as u8;
                    let g = ((1.0 - fx) * (1.0 - fy) * p00[1] as f64
                        + fx * (1.0 - fy) * p10[1] as f64
                        + (1.0 - fx) * fy * p01[1] as f64
                        + fx * fy * p11[1] as f64) as u8;
                    let b = ((1.0 - fx) * (1.0 - fy) * p00[2] as f64
                        + fx * (1.0 - fy) * p10[2] as f64
                        + (1.0 - fx) * fy * p01[2] as f64
                        + fx * fy * p11[2] as f64) as u8;

                    out_img.put_pixel(x, y, image::Rgb([r, g, b]));
                } else if x0 >= 0 && x0 < src_cols && y0 >= 0 && y0 < src_rows {
                    // Fallback to nearest neighbor at edges
                    let pixel = src_img.get_pixel(x0 as u32, y0 as u32);
                    out_img.put_pixel(x, y, *pixel);
                }
            }
        }

        *dst = Mat::new(DynamicImage::ImageRgb8(out_img));
        Ok(())
    }

    pub fn get_perspective_transform(
        src_pts: &[[f32; 2]; 4],
        dst_pts: &[[f32; 2]; 4],
    ) -> Result<[[f64; 3]; 3]> {
        use nalgebra::DMatrix;

        // Try the simple method first: solve for 8 parameters with c22 = 1
        // This matches OpenCV's first attempt
        let mut a = DMatrix::<f64>::zeros(8, 8);
        let mut b = DMatrix::<f64>::zeros(8, 1);

        for i in 0..4 {
            let x = src_pts[i][0] as f64;
            let y = src_pts[i][1] as f64;
            let u = dst_pts[i][0] as f64;
            let v = dst_pts[i][1] as f64;

            // Row for u (x') coordinate
            a[(i, 0)] = x;
            a[(i, 1)] = y;
            a[(i, 2)] = 1.0;
            a[(i, 6)] = -u * x;
            a[(i, 7)] = -u * y;
            b[(i, 0)] = u;

            // Row for v (y') coordinate
            a[(i + 4, 3)] = x;
            a[(i + 4, 4)] = y;
            a[(i + 4, 5)] = 1.0;
            a[(i + 4, 6)] = -v * x;
            a[(i + 4, 7)] = -v * y;
            b[(i + 4, 0)] = v;
        }

        // Solve A * x = b using LU decomposition
        if let Some(lu) = a.clone().lu().solve(&b) {
            // Check if solution is good
            let residual = (&a * &lu - &b).norm();
            if residual < 1e-8 {
                return Ok([
                    [lu[(0, 0)], lu[(1, 0)], lu[(2, 0)]],
                    [lu[(3, 0)], lu[(4, 0)], lu[(5, 0)]],
                    [lu[(6, 0)], lu[(7, 0)], 1.0],
                ]);
            }
        }

        // If the simple method failed, use SVD on the full 9-parameter system
        // This is OpenCV's fallback approach
        let mut a9 = DMatrix::<f64>::zeros(8, 9);
        for i in 0..4 {
            let x = src_pts[i][0] as f64;
            let y = src_pts[i][1] as f64;
            let u = dst_pts[i][0] as f64;
            let v = dst_pts[i][1] as f64;

            a9[(i, 0)] = x;
            a9[(i, 1)] = y;
            a9[(i, 2)] = 1.0;
            a9[(i, 6)] = -u * x;
            a9[(i, 7)] = -u * y;
            a9[(i, 8)] = -u;

            a9[(i + 4, 3)] = x;
            a9[(i + 4, 4)] = y;
            a9[(i + 4, 5)] = 1.0;
            a9[(i + 4, 6)] = -v * x;
            a9[(i + 4, 7)] = -v * y;
            a9[(i + 4, 8)] = -v;
        }

        // Compute A^T * A
        let ata = a9.transpose() * &a9;
        
        // Perform SVD on A^T * A
        let svd = ata.svd(true, false);
        
        // Get the right singular vector corresponding to the smallest singular value
        let v = svd.u.ok_or("SVD failed")?;
        let h = v.column(8); // Last column corresponds to smallest singular value

        Ok([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], h[8]],
        ])
    }

    fn invert_matrix_3x3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
        use nalgebra::Matrix3;

        let mat = Matrix3::new(
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
        );

        let inv = mat.try_inverse().ok_or("Matrix is not invertible")?;

        Ok([
            [inv[(0, 0)], inv[(0, 1)], inv[(0, 2)]],
            [inv[(1, 0)], inv[(1, 1)], inv[(1, 2)]],
            [inv[(2, 0)], inv[(2, 1)], inv[(2, 2)]],
        ])
    }

    pub fn min_area_rect(contour: &[Point2f]) -> Result<(Point2f, Size, f32)> {
        // Rotating calipers algorithm to find minimum area bounding box
        if contour.is_empty() {
            return Err("Empty contour".into());
        }

        if contour.len() == 1 {
            return Ok((contour[0], Size::new(0, 0), 0.0));
        }

        if contour.len() == 2 {
            let dx = contour[1].x - contour[0].x;
            let dy = contour[1].y - contour[0].y;
            let len = (dx * dx + dy * dy).sqrt();
            let center = Point2f::new(
                (contour[0].x + contour[1].x) / 2.0,
                (contour[0].y + contour[1].y) / 2.0,
            );
            let angle = dy.atan2(dx).to_degrees();
            return Ok((center, Size::new(len as i32, 0), angle));
        }

        // Compute convex hull first
        let hull = compute_convex_hull(contour);
        
        if hull.len() < 3 {
            // Fallback to axis-aligned bbox
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for pt in contour {
                min_x = min_x.min(pt.x);
                max_x = max_x.max(pt.x);
                min_y = min_y.min(pt.y);
                max_y = max_y.max(pt.y);
            }

            let center = Point2f::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
            let width = (max_x - min_x) as i32;
            let height = (max_y - min_y) as i32;
            return Ok((center, Size::new(width, height), 0.0));
        }

        // Rotating calipers to find minimum area rectangle
        let mut min_area = f32::MAX;
        let mut best_rect = None;

        let n = hull.len();
        for i in 0..n {
            let p1 = hull[i];
            let p2 = hull[(i + 1) % n];
            
            // Edge vector
            let edge_x = p2.x - p1.x;
            let edge_y = p2.y - p1.y;
            let edge_len = (edge_x * edge_x + edge_y * edge_y).sqrt();
            
            if edge_len < 1e-6 {
                continue;
            }

            // Normalized edge direction
            let ux = edge_x / edge_len;
            let uy = edge_y / edge_len;
            
            // Perpendicular direction
            let vx = -uy;
            let vy = ux;

            // Project all hull points onto edge direction and perpendicular
            let mut min_u = f32::MAX;
            let mut max_u = f32::MIN;
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;

            for pt in &hull {
                let u = pt.x * ux + pt.y * uy;
                let v = pt.x * vx + pt.y * vy;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let width = max_u - min_u;
            let height = max_v - min_v;
            let area = width * height;

            if area < min_area {
                min_area = area;
                
                // Center in the rotated coordinate system
                let center_u = (min_u + max_u) / 2.0;
                let center_v = (min_v + max_v) / 2.0;
                
                // Transform back to original coordinates
                let center_x = center_u * ux + center_v * vx;
                let center_y = center_u * uy + center_v * vy;
                
                let angle = uy.atan2(ux).to_degrees();
                
                best_rect = Some((
                    Point2f::new(center_x, center_y),
                    Size::new(width as i32, height as i32),
                    angle,
                ));
            }
        }

        best_rect.ok_or_else(|| "Failed to compute minimum area rectangle".into())
    }

    /// Compute convex hull using Graham scan
    fn compute_convex_hull(points: &[Point2f]) -> Vec<Point2f> {
        if points.len() <= 3 {
            return points.to_vec();
        }

        // Find the point with lowest y-coordinate (and leftmost if tie)
        let mut start_idx = 0;
        for (i, pt) in points.iter().enumerate().skip(1) {
            if pt.y < points[start_idx].y || (pt.y == points[start_idx].y && pt.x < points[start_idx].x) {
                start_idx = i;
            }
        }

        let start = points[start_idx];

        // Sort points by polar angle with respect to start point
        let mut sorted: Vec<Point2f> = points.to_vec();
        sorted.swap(0, start_idx);
        
        sorted[1..].sort_by(|a, b| {
            let angle_a = (a.y - start.y).atan2(a.x - start.x);
            let angle_b = (b.y - start.y).atan2(b.x - start.x);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Graham scan
        let mut hull = Vec::new();
        hull.push(sorted[0]);
        hull.push(sorted[1]);

        for pt in sorted.iter().skip(2) {
            while hull.len() >= 2 {
                let len = hull.len();
                let p1 = hull[len - 2];
                let p2 = hull[len - 1];
                
                // Cross product to check if we make a left turn
                let cross = (p2.x - p1.x) * (pt.y - p1.y) - (p2.y - p1.y) * (pt.x - p1.x);
                
                if cross <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(*pt);
        }

        hull
    }

    pub fn box_points(center: Point2f, size: Size, angle: f32) -> [Point2f; 4] {
        let angle_rad = angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let w = size.width as f32 / 2.0;
        let h = size.height as f32 / 2.0;

        let corners = [
            (-w, -h),
            (w, -h),
            (w, h),
            (-w, h),
        ];

        corners.map(|(dx, dy)| {
            Point2f::new(
                center.x + dx * cos_a - dy * sin_a,
                center.y + dx * sin_a + dy * cos_a,
            )
        })
    }

    // Constants for compatibility
    pub const INTER_LINEAR: i32 = 1;
    #[allow(dead_code)]
    pub const INTER_CUBIC: i32 = 2;
    pub const BORDER_REPLICATE: i32 = 1;
    pub const ROTATE_90_CLOCKWISE: i32 = 0;
}

// OpenCV implementation
#[cfg(feature = "use-opencv")]
mod opencv_impl {
    use super::{Point2f, Size};
    pub use opencv::core::Mat;
    pub use opencv::imgcodecs::{imread as cv_imread, imwrite as cv_imwrite, IMREAD_COLOR};
    pub use opencv::imgproc::{
        get_perspective_transform, resize as cv_resize, warp_perspective as cv_warp_perspective,
        INTER_CUBIC, INTER_LINEAR,
    };
    pub use opencv::core::{rotate as cv_rotate, BorderTypes, RotateFlags};
    
    pub const BORDER_REPLICATE: i32 = BorderTypes::BORDER_REPLICATE as i32;
    pub const ROTATE_90_CLOCKWISE: i32 = RotateFlags::ROTATE_90_CLOCKWISE as i32;
    use std::path::Path;

    pub type Result<T> = opencv::Result<T>;

    pub fn imread<P: AsRef<Path>>(path: P) -> Result<Mat> {
        cv_imread(path.as_ref().to_str().unwrap(), IMREAD_COLOR)
    }

    pub fn imwrite<P: AsRef<Path>>(path: P, img: &Mat) -> Result<()> {
        cv_imwrite(
            path.as_ref().to_str().unwrap(),
            img,
            &opencv::core::Vector::new(),
        )?;
        Ok(())
    }

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

    pub fn rotate(src: &Mat, dst: &mut Mat, rotation: i32) -> Result<()> {
        cv_rotate(src, dst, rotation)
    }

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

    // Convert Point2f to opencv::core::Point2f
    impl From<Point2f> for opencv::core::Point2f {
        fn from(p: Point2f) -> Self {
            opencv::core::Point2f::new(p.x, p.y)
        }
    }

    impl From<opencv::core::Point2f> for Point2f {
        fn from(p: opencv::core::Point2f) -> Self {
            Point2f::new(p.x, p.y)
        }
    }

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

    pub fn box_points(center: Point2f, size: Size, angle: f32) -> [Point2f; 4] {
        let rect = opencv::core::RotatedRect {
            center: opencv::core::Point2f::new(center.x, center.y),
            size: opencv::core::Size2f::new(size.width as f32, size.height as f32),
            angle,
        };

        let mut pts = [opencv::core::Point2f::default(); 4];
        rect.points(&mut pts).unwrap();

        [
            Point2f::from(pts[0]),
            Point2f::from(pts[1]),
            Point2f::from(pts[2]),
            Point2f::from(pts[3]),
        ]
    }
}
