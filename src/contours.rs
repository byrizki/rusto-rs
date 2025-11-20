// Pure Rust contour detection implementation
// This module provides OpenCV-like contour detection for the pure Rust backend

#![allow(dead_code)] // Keep alternative implementations for reference

use image::GrayImage;

#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<(i32, i32)>,
}

impl Contour {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }
    
    pub fn len(&self) -> usize {
        self.points.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Find contours in a binary image
/// This matches OpenCV's findContours with RETR_LIST mode
pub fn find_contours(binary_img: &GrayImage) -> Vec<Contour> {
    let (width, height) = binary_img.dimensions();
    let mut label_map = vec![vec![0u32; width as usize]; height as usize];
    let mut contours = Vec::new();
    let mut label = 1u32;
    
    // First pass: label connected components using flood fill
    for y in 0..height {
        for x in 0..width {
            let ux = x as usize;
            let uy = y as usize;
            
            if binary_img.get_pixel(x, y)[0] > 127 && label_map[uy][ux] == 0 {
                // Label this component
                flood_fill_label(binary_img, &mut label_map, x as i32, y as i32, label, width as i32, height as i32);
                label += 1;
            }
        }
    }
    
    // Second pass: extract contours for each label
    for current_label in 1..label {
        // Find the topmost-leftmost pixel of this label
        let mut _start_x = 0;
        let mut _start_y = 0;
        let mut found = false;
        
        'outer: for y in 0..height {
            for x in 0..width {
                if label_map[y as usize][x as usize] == current_label {
                    _start_x = x as i32;
                    _start_y = y as i32;
                    found = true;
                    break 'outer;
                }
            }
        }
        
        if !found {
            continue;
        }
        
        // Note: In a future implementation, we could use start_x/start_y for contour tracing
        // For now, we extract all boundary pixels directly
        
        // Extract all boundary pixels for this contour
        let boundary_pixels = extract_boundary(&label_map, current_label, width as i32, height as i32);
        
        if boundary_pixels.len() >= 3 {
            contours.push(Contour { points: boundary_pixels });
        }
    }
    
    contours
}

/// Flood fill to label a connected component
fn flood_fill_label(
    img: &GrayImage,
    labels: &mut Vec<Vec<u32>>,
    start_x: i32,
    start_y: i32,
    label: u32,
    width: i32,
    height: i32,
) {
    let mut stack = vec![(start_x, start_y)];
    
    while let Some((x, y)) = stack.pop() {
        if x < 0 || x >= width || y < 0 || y >= height {
            continue;
        }
        
        let ux = x as usize;
        let uy = y as usize;
        
        if labels[uy][ux] != 0 {
            continue;
        }
        
        let pixel = img.get_pixel(x as u32, y as u32)[0];
        if pixel <= 127 {
            continue;
        }
        
        labels[uy][ux] = label;
        
        // 4-connected neighbors
        stack.push((x + 1, y));
        stack.push((x - 1, y));
        stack.push((x, y + 1));
        stack.push((x, y - 1));
    }
}

/// Check if a pixel is on the boundary of its labeled region
fn is_boundary_pixel_label(
    label_map: &[Vec<u32>],
    x: i32,
    y: i32,
    target_label: u32,
    width: i32,
    height: i32,
) -> bool {
    // Check 4-connected neighbors
    for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)].iter() {
        let nx = x + dx;
        let ny = y + dy;
        
        // Edge or different label = boundary
        if nx < 0 || nx >= width || ny < 0 || ny >= height {
            return true;
        }
        
        if label_map[ny as usize][nx as usize] != target_label {
            return true;
        }
    }
    
    false
}

/// Extract all boundary pixels for a labeled region
fn extract_boundary(
    label_map: &[Vec<u32>],
    target_label: u32,
    width: i32,
    height: i32,
) -> Vec<(i32, i32)> {
    let mut boundary = Vec::new();
    
    for y in 0..height {
        for x in 0..width {
            if label_map[y as usize][x as usize] == target_label &&
               is_boundary_pixel_label(label_map, x, y, target_label, width, height) {
                boundary.push((x, y));
            }
        }
    }
    
    boundary
}

/// Follow a contour border using the square tracing algorithm (simplified Suzuki-Abe)
fn follow_border(
    img: &[Vec<u8>],
    start_i: i32,
    start_j: i32,
    second_i: i32,
    second_j: i32,
    _nbd: i32,
    border: &mut Vec<(i32, i32)>,
    _lnbd_img: &mut [Vec<u8>],
) {
    // 8-connectivity: clockwise from (di, dj) = (0, 1)
    const DIR: [(i32, i32); 8] = [
        (0, 1),   // 0: East
        (-1, 1),  // 1: NE
        (-1, 0),  // 2: North
        (-1, -1), // 3: NW
        (0, -1),  // 4: West
        (1, -1),  // 5: SW
        (1, 0),   // 6: South
        (1, 1),   // 7: SE
    ];
    
    border.push((start_i, start_j));
    
    let mut curr_i = start_i;
    let mut curr_j = start_j;
    let mut prev_i = second_i;
    let mut prev_j = second_j;
    
    let mut step_count = 0;
    let max_steps = (img.len() * img[0].len()) * 2;
    
    loop {
        step_count += 1;
        if step_count > max_steps {
            break; // Safety: prevent infinite loops
        }
        
        // Find the direction from current to previous
        let mut search_dir = 0;
        for (idx, &(di, dj)) in DIR.iter().enumerate() {
            if curr_i + di == prev_i && curr_j + dj == prev_j {
                // Start search from next direction (clockwise)
                search_dir = (idx + 1) % 8;
                break;
            }
        }
        
        // Search for next border pixel
        let mut found = false;
        for k in 0..8 {
            let dir_idx = (search_dir + k) % 8;
            let (di, dj) = DIR[dir_idx];
            let ni = curr_i + di;
            let nj = curr_j + dj;
            
            if ni >= 0 && ni < img.len() as i32 && nj >= 0 && nj < img[0].len() as i32 {
                if img[ni as usize][nj as usize] >= 1 {
                    // Found next border pixel
                    if ni == start_i && nj == start_j && border.len() > 2 {
                        // Closed the contour
                        return;
                    }
                    
                    border.push((ni, nj));
                    prev_i = curr_i;
                    prev_j = curr_j;
                    curr_i = ni;
                    curr_j = nj;
                    found = true;
                    break;
                }
            }
        }
        
        if !found {
            break; // Isolated pixel or end of contour
        }
    }
}

/// Check if a foreground pixel has at least one background neighbor (4-connected)
fn is_border_pixel(img: &GrayImage, x: i32, y: i32, width: i32, height: i32) -> bool {
    for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)].iter() {
        let nx = x + dx;
        let ny = y + dy;
        
        // Edge of image is considered border
        if nx < 0 || nx >= width || ny < 0 || ny >= height {
            return true;
        }
        
        let neighbor_pixel = img.get_pixel(nx as u32, ny as u32)[0];
        if neighbor_pixel <= 127 {
            return true; // Has background neighbor
        }
    }
    
    false
}

/// Trace boundary contour using square tracing algorithm
fn trace_boundary(
    img: &GrayImage,
    visited: &mut Vec<Vec<bool>>,
    start_x: i32,
    start_y: i32,
    width: i32,
    height: i32,
) -> Option<Contour> {
    let mut boundary_points = Vec::new();
    
    // Use 8-connectivity for boundary tracing (clockwise)
    const DIR: [(i32, i32); 8] = [
        (1, 0),   // East
        (1, 1),   // SE
        (0, 1),   // South
        (-1, 1),  // SW
        (-1, 0),  // West
        (-1, -1), // NW
        (0, -1),  // North
        (1, -1),  // NE
    ];
    
    let mut current_x = start_x;
    let mut current_y = start_y;
    let mut dir_idx = 0; // Start looking East
    
    loop {
        // Add current point to boundary
        boundary_points.push((current_x, current_y));
        
        // Look for next boundary pixel
        let mut found = false;
        for i in 0..8 {
            let check_dir = (dir_idx + i) % 8;
            let (dx, dy) = DIR[check_dir];
            let nx = current_x + dx;
            let ny = current_y + dy;
            
            // Check bounds
            if nx < 0 || nx >= width || ny < 0 || ny >= height {
                continue;
            }
            
            let pixel = img.get_pixel(nx as u32, ny as u32)[0];
            if pixel > 127 {
                // Found next boundary pixel
                current_x = nx;
                current_y = ny;
                // Update search direction (turn left from current direction)
                dir_idx = if check_dir >= 2 { check_dir - 2 } else { check_dir + 6 };
                found = true;
                break;
            }
        }
        
        if !found {
            break; // No more boundary pixels
        }
        
        // Check if we've returned to start
        if current_x == start_x && current_y == start_y && boundary_points.len() > 1 {
            break;
        }
        
        // Safety check
        if boundary_points.len() > (width * height) as usize {
            break;
        }
    }
    
    // Mark all pixels in the region as visited using flood fill
    flood_fill_visited(img, visited, start_x, start_y, width, height);
    
    // Simplify contour (approximate like CHAIN_APPROX_SIMPLE)
    let simplified = simplify_contour(&boundary_points);
    
    if simplified.len() >= 3 {
        Some(Contour { points: simplified })
    } else {
        None
    }
}

/// Mark all pixels in a connected region as visited
fn flood_fill_visited(
    img: &GrayImage,
    visited: &mut Vec<Vec<bool>>,
    start_x: i32,
    start_y: i32,
    width: i32,
    height: i32,
) {
    let mut stack = vec![(start_x, start_y)];
    
    while let Some((x, y)) = stack.pop() {
        if x < 0 || x >= width || y < 0 || y >= height {
            continue;
        }
        
        let ux = x as usize;
        let uy = y as usize;
        
        if visited[uy][ux] {
            continue;
        }
        
        let pixel = img.get_pixel(x as u32, y as u32)[0];
        if pixel <= 127 {
            continue;
        }
        
        visited[uy][ux] = true;
        
        // Add 8-connected neighbors
        stack.push((x + 1, y));
        stack.push((x - 1, y));
        stack.push((x, y + 1));
        stack.push((x, y - 1));
        stack.push((x + 1, y + 1));
        stack.push((x - 1, y - 1));
        stack.push((x + 1, y - 1));
        stack.push((x - 1, y + 1));
    }
}

/// Simplify contour by removing collinear points (similar to CHAIN_APPROX_SIMPLE)
fn simplify_contour(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if points.len() <= 3 {
        return points.to_vec();
    }
    
    let mut simplified = Vec::new();
    simplified.push(points[0]);
    
    for i in 1..points.len() - 1 {
        let prev = simplified.last().unwrap();
        let curr = &points[i];
        let next = &points[i + 1];
        
        // Check if current point is collinear with prev and next
        let dx1 = curr.0 - prev.0;
        let dy1 = curr.1 - prev.1;
        let dx2 = next.0 - curr.0;
        let dy2 = next.1 - curr.1;
        
        // Cross product to check collinearity
        let cross = dx1 * dy2 - dy1 * dx2;
        
        // Keep point if not collinear (with tolerance)
        if cross.abs() > 0 {
            simplified.push(*curr);
        }
    }
    
    // Always keep last point
    if let Some(last) = points.last() {
        simplified.push(*last);
    }
    
    simplified
}

/// Flood fill to label connected components
fn flood_fill(
    img: &GrayImage,
    labels: &mut Vec<Vec<i32>>,
    x: i32,
    y: i32,
    label: i32,
    width: i32,
    height: i32,
) {
    let mut stack = vec![(x, y)];
    
    while let Some((cx, cy)) = stack.pop() {
        if cx < 0 || cx >= width || cy < 0 || cy >= height {
            continue;
        }
        
        let ux = cx as usize;
        let uy = cy as usize;
        
        if labels[uy][ux] != 0 {
            continue;
        }
        
        let pixel = img.get_pixel(cx as u32, cy as u32)[0];
        if pixel <= 127 {
            continue;
        }
        
        labels[uy][ux] = label;
        
        // Add 8-connected neighbors (includes diagonals)
        stack.push((cx + 1, cy));
        stack.push((cx - 1, cy));
        stack.push((cx, cy + 1));
        stack.push((cx, cy - 1));
        stack.push((cx + 1, cy + 1));
        stack.push((cx - 1, cy - 1));
        stack.push((cx + 1, cy - 1));
        stack.push((cx - 1, cy + 1));
    }
}

/// Check if a pixel is on the boundary of a region
fn is_boundary_pixel(x: i32, y: i32, labels: &[Vec<i32>], width: i32, height: i32) -> bool {
    let label = labels[y as usize][x as usize];
    
    // Check 4-connected neighbors
    for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)].iter() {
        let nx = x + dx;
        let ny = y + dy;
        
        if nx < 0 || nx >= width || ny < 0 || ny >= height {
            return true; // Edge of image is boundary
        }
        
        if labels[ny as usize][nx as usize] != label {
            return true; // Different label or background
        }
    }
    
    false
}

/// Trace a single contour using Moore-Neighbor algorithm (UNUSED - kept for reference)
#[allow(dead_code)]
fn trace_contour(
    img: &GrayImage,
    visited: &mut Vec<Vec<bool>>,
    start_x: i32,
    start_y: i32,
) -> Option<Contour> {
    let (width, height) = img.dimensions();
    let width = width as i32;
    let height = height as i32;
    
    let mut contour = Contour::new();
    let mut current_x = start_x;
    let mut current_y = start_y;
    
    // 8-connectivity neighbors (Moore neighborhood)
    // Ordered clockwise starting from top
    const DX: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
    const DY: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
    
    let mut dir = 0; // Start direction
    let mut found_start = false;
    
    loop {
        // Add current point to contour
        if !found_start || current_x != start_x || current_y != start_y {
            contour.points.push((current_x, current_y));
            if current_y >= 0 && current_y < height && current_x >= 0 && current_x < width {
                visited[current_y as usize][current_x as usize] = true;
            }
        } else if found_start {
            // Completed the contour
            break;
        }
        
        found_start = true;
        
        // Search for next contour point in Moore neighborhood
        let mut found_next = false;
        for i in 0..8 {
            let check_dir = (dir + i) % 8;
            let nx = current_x + DX[check_dir];
            let ny = current_y + DY[check_dir];
            
            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                let pixel = img.get_pixel(nx as u32, ny as u32)[0];
                if pixel > 127 {
                    current_x = nx;
                    current_y = ny;
                    dir = (check_dir + 5) % 8; // Backtrack direction for next search
                    found_next = true;
                    break;
                }
            }
        }
        
        if !found_next {
            break; // Isolated point or end of contour
        }
        
        // Safety check to prevent infinite loops
        if contour.points.len() > (width * height) as usize {
            break;
        }
    }
    
    if contour.points.len() >= 3 {
        Some(contour)
    } else {
        None
    }
}

/// Approximate contour to reduce number of points (similar to CHAIN_APPROX_SIMPLE)
pub fn approx_simple(contour: &Contour) -> Contour {
    if contour.points.len() <= 2 {
        return contour.clone();
    }
    
    let mut result = Contour::new();
    result.points.push(contour.points[0]);
    
    let mut i = 1;
    while i < contour.points.len() - 1 {
        let prev = contour.points[i - 1];
        let curr = contour.points[i];
        let next = contour.points[i + 1];
        
        // Check if current point is on the same line as prev and next
        let dx1 = curr.0 - prev.0;
        let dy1 = curr.1 - prev.1;
        let dx2 = next.0 - curr.0;
        let dy2 = next.1 - curr.1;
        
        // Cross product to check collinearity
        if dx1 * dy2 != dy1 * dx2 {
            result.points.push(curr);
        }
        
        i += 1;
    }
    
    // Always include last point
    if let Some(&last) = contour.points.last() {
        result.points.push(last);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    
    #[test]
    fn test_find_contours_simple() {
        let mut img = GrayImage::new(10, 10);
        
        // Draw a simple rectangle
        for x in 2..8 {
            for y in 2..8 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        
        let contours = find_contours(&img);
        assert!(!contours.is_empty(), "Should find at least one contour");
    }
}
