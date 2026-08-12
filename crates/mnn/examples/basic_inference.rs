/// Basic MNN inference example
/// Demonstrates loading a model and running inference
use rusto_mnn::{Interpreter, ScheduleConfig, ForwardType, BackendConfig, PrecisionMode, PowerMode};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MNN Basic Inference Example ===\n");

    // Path to your MNN model
    let model_path = Path::new("../../models/PPOCR_v5/det.mnn");
    
    if !model_path.exists() {
        eprintln!("Error: Model file not found at {}", model_path.display());
        eprintln!("Please ensure the model file exists.");
        return Ok(());
    }

    println!("Loading model: {}", model_path.display());
    let interpreter = Interpreter::from_file(model_path)?;
    println!("✓ Model loaded successfully\n");

    // Configure inference session
    println!("Configuring inference session...");
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::Auto); // Auto-select best backend
    config.set_num_thread(4);

    // Optional: Configure backend
    let mut backend_config = BackendConfig::new();
    backend_config.set_precision_mode(PrecisionMode::High);
    backend_config.set_power_mode(PowerMode::High);
    config.set_backend_config(backend_config);

    println!("✓ Configuration complete\n");

    // Create session
    println!("Creating inference session...");
    let session = interpreter.create_session(config)?;
    println!("✓ Session created successfully\n");

    // Get input/output information
    let inputs = interpreter.inputs(&session);
    let outputs = interpreter.outputs(&session);

    println!("Model Information:");
    println!("  Inputs: {}", inputs.len());
    for input in &inputs {
        println!("    - {}", input.name());
    }
    
    println!("  Outputs: {}", outputs.len());
    for output in &outputs {
        println!("    - {}", output.name());
    }

    println!("\n✓ Model inspection complete");
    println!("\nNote: To run actual inference, prepare input data and use:");
    println!("  interpreter.input(&mut session, \"input_name\")");
    println!("  interpreter.run_session(&mut session)");
    println!("  interpreter.output(&session, \"output_name\")");

    Ok(())
}
