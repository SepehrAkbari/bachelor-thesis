/*
Orchestrator for processing, model pretraining, and PPO training.
*/

use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, exit};
use std::time::Instant;


fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        get_usage();
        exit(1);
    }

    let targets = if args[0].to_lowercase() == "all" {
        find_dist()
    } else {
        args
    };

    if targets.is_empty() {
        eprintln!("Error: No distributions found or provided.");
        exit(1);
    }

    for (i, dist) in targets.iter().enumerate() {
        println!("\nProcessing {}/{} : {}", i + 1, targets.len(), dist);
        if let Err(e) = pipeline(dist) {
            eprintln!("Error: Pipeline failed for '{}': {}.", dist, e);
            exit(1);
        }
    }
}

fn pipeline(dist: &str) -> Result<(), String> {
    let steps = vec![
        ("dataset.py", "Processing Dataset"),
        ("pretrain.py", "Pretraining Model"),
        ("train.py", "Training PPO Agent"),
    ];

    for (script, description) in steps {
        println!("Excuting {}: {}.", script, description);
        
        let timer = Instant::now();
        
        let status = Command::new("python")
            .arg(script)
            .arg(dist)
            .status()
            .map_err(|e| format!("Error: Failed to execute process {}.", e))?;

        if !status.success() {
            return Err(format!("Error: {} exited with error code.", script));
        }

        println!("  Done (took {:.1}s)", timer.elapsed().as_secs_f64());
    }

    Ok(())
}

fn find_dist() -> Vec<String> {
    let mut dists = Vec::new();
    
    let data_path = Path::new("..").join("data").join("stats");

    if !data_path.exists() {
        eprintln!("Error: Data directory not found at {:?}", data_path);
        exit(1);
    }

    let paths = fs::read_dir(data_path).expect("Error: Could not read data directory.");

    for path in paths {
        if let Ok(entry) = path {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if !name.starts_with('.') {
                        dists.push(name.to_string());
                    }
                }
            }
        }
    }
    
    dists.sort();
    dists
}

fn get_usage() {
    println!("Usage:");
    println!("  ./orchestrator all");
    println!("  ./orchestrator <dist>");
    println!("  ./orchestrator <dist1> <dist2> ... ");
}