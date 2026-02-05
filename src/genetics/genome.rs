use rand::Rng;

#[derive(Clone, Debug)]
pub struct Genome {
    pub layers: Vec<usize>,
    pub lr: f64,
    pub fitness: f32,
}

impl Genome {
    pub fn random(input: usize, output: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_layers = rng.gen_range(1..3);
        let mut layers = vec![input];
        for _ in 0..hidden_layers {
            layers.push(rng.gen_range(32..256));
        }
        layers.push(output);
        Self { layers, lr: 0.01, fitness: 0.0 }
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        if rng.gen_bool(0.5) && self.layers.len() > 2 {
            let idx = rng.gen_range(1..self.layers.len() - 1);
            self.layers[idx] = rng.gen_range(32..256);
        } else {
            self.lr += rng.gen_range(-0.005..0.005);
        }
    }
}