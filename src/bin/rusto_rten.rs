use rusto::cli::{run_cli, EngineKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_cli(EngineKind::Rten)
}

