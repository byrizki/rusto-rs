use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rusto::{RustO, RustOConfig};

fn create_ocr() -> RustO {
    let config = RustOConfig {
        det_model_path: "models/PPOCR_v5/det.onnx".to_string(),
        rec_model_path: "models/PPOCR_v5/rec.onnx".to_string(),
        dict_path: "models/PPOCR_v5/dict.txt".to_string(),
    };
    RustO::new(config).expect("Failed to create OCR")
}

fn benchmark_full_ocr(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_ocr");
    group.sample_size(10);
    
    let test_images = vec![
        "models/images/ktp-teng.jpg",
        "models/test_images/example1.png",
    ];
    
    for image_path in test_images {
        if std::path::Path::new(image_path).exists() {
            group.bench_with_input(
                BenchmarkId::from_parameter(image_path),
                &image_path,
                |b, &path| {
                    let mut ocr = create_ocr();
                    b.iter(|| {
                        ocr.ocr(black_box(path)).expect("OCR failed")
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_detection_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_only");
    group.sample_size(20);
    
    // Simplified - just benchmark full OCR for now as internal modules aren't exposed
    if std::path::Path::new("models/images/ktp-teng.jpg").exists() {
        group.bench_function("ktp-teng", |b| {
            let mut ocr = create_ocr();
            b.iter(|| {
                ocr.ocr(black_box("models/images/ktp-teng.jpg")).expect("OCR failed")
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_full_ocr, benchmark_detection_only);
criterion_main!(benches);
