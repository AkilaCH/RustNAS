use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use crate::model::network::{MlpConfig, DynamicMlp};
use crate::genetics::genome::Genome;

pub fn get_fitness<B: AutodiffBackend>(genome: &Genome, device: &B::Device) -> f32 {

    let config = MlpConfig {
        layer_sizes: genome.layers.clone(),
        output_size: 10,
    };
    
    let _model: DynamicMlp<B> = config.init(device);    
    let layer_count = genome.layers.len() as f32;
    let efficiency_score = 1.0 - (layer_count * 0.05); 
    let stability_score = 1.0 - (genome.lr - 0.01).abs() as f32;
    let final_score = (efficiency_score + stability_score) / 2.0;


    final_score.clamp(0.0, 1.0)
}