mod genetics;
mod model;

use genetics::genome::Genome;
use rayon::prelude::*;
use burn::backend::{Wgpu, Autodiff};

// Define the Backend (WGPU for GPU support, or NdArray for CPU)
type MyBackend = Autodiff<Wgpu>;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    println!("--- Starting Rust NAS ---");
    println!("Searching for optimal Neural Network architecture...");

    let mut population: Vec<Genome> = (0..10)
        .map(|_| Genome::random(784, 10))
        .collect();

    for gen in 0..5 {
        println!("\n=== Generation {} ===", gen);

        population.par_iter_mut().for_each(|genome| {
            genome.fitness = model::trainer::get_fitness::<MyBackend>(genome, &device);
        });

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let best = &population[0];
        println!(">> Best Candidate: Layers {:?} | Fitness: {:.4}", best.layers, best.fitness);

        let survivors = &population[0..2].to_vec();
        let mut next_gen = survivors.clone();

        while next_gen.len() < population.len() {
            let parent = &survivors[rand::random::<usize>() % survivors.len()];
            let mut child = parent.clone();
            child.mutate();
            next_gen.push(child);
        }
        population = next_gen;
    }
}