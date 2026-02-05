use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Relu};

// 1. Define the Module
// We use a Vec<Linear<B>> to hold a dynamic number of layers.
// Burn supports vector modules out of the box!
#[derive(Module, Debug)]
pub struct DynamicMlp<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
    output_layer: Linear<B>,
}

// 2. The Configuration
// This tells Burn how to build the model
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub layer_sizes: Vec<usize>, // e.g., [784, 64, 32]
    pub output_size: usize,      // e.g., 10 (classes)
}

impl MlpConfig {
    // The Builder Pattern: Creates the model from the config
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicMlp<B> {
        let mut layers = Vec::new();
        
        // iterate through the sizes: [784, 64] -> creates layer 784->64
        for window in self.layer_sizes.windows(2) {
            let in_dim = window[0];
            let out_dim = window[1];
            
            let layer = LinearConfig::new(in_dim, out_dim).init(device);
            layers.push(layer);
        }

        // The final output layer maps the last hidden layer to the classes
        let last_hidden = *self.layer_sizes.last().unwrap();
        let output_layer = LinearConfig::new(last_hidden, self.output_size).init(device);

        DynamicMlp {
            layers,
            activation: Relu::new(),
            output_layer,
        }
    }
}

// 3. The Forward Pass
impl<B: Backend> DynamicMlp<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Dynamic forward pass through however many layers we have
        for layer in &self.layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        self.output_layer.forward(x)
    }
}