use ndarray::Array2;
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::Value;
use ort::execution_providers::{CUDAExecutionProvider, CPUExecutionProvider};
use ort::value::Tensor;
use std::sync::Arc;
use crate::board::Board;
use crate::tile::{adjacent, Tile, GRID_SIZE, TILE_ZERO};
use crate::piece_type::{PCT_COUNT};

const NUM_FEATURES: usize = 4 + PCT_COUNT; // 1 hot encoding: 2 for occupancy, 2 for color, PCT_COUNT for piece type

pub struct GameGraph{
    pub nodes: Vec<Tile>,
    pub edges: Vec<Vec<bool>>,
    pub features: Vec<Vec<f32>>,
}

impl GameGraph {
    pub fn new() -> Self {
        GameGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            features: Vec::new(),
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl Board {
    // Converts a tile to a feature vector for graph neural networks
    fn tile_to_feature(&self, tile: Tile) -> Vec<f32> {
        let piece = self.tile(tile);
        let mut feat = vec![0.0; 4 + PCT_COUNT];
        if piece.is_none() {
            feat[0] = 1.0; // Empty tile
            return feat;
        }
        feat[1] = 1.0; // Occupied tile
        // Color one-hot
        feat[2 + piece.color().index()] = 1.0;
        // Piece type one-hot
        feat[2 + 2 + piece.ptype().index()] = 1.0;
        feat
    }
    
    // Converts the board state into a graph representation
    pub fn get_graph(&self) -> GameGraph {
        let mut ids: [i16; GRID_SIZE] = [-1; GRID_SIZE];

        let start: Tile = *self.occupied_tiles[self.color().index()].first().unwrap_or_else(
            || self.occupied_tiles[self.color().index()].first().unwrap_or(&TILE_ZERO));

        let mut nodes = Vec::new();
        let mut features = Vec::new();
        nodes.push(start);
        ids[start as usize] = 0;
        features.push(self.tile_to_feature(start));

        let mut i = 0;

        // BFS to find all occupied tiles and tiles adjacent to them
        while i < nodes.len() {
            let tile = nodes[i];

            if self.tile(tile).is_none() {
                i += 1;
                continue;
            }

            for adj in adjacent(tile) {
                if ids[adj as usize] == -1 {
                    ids[adj as usize] = nodes.len() as i16;
                    nodes.push(adj);
                    features.push(self.tile_to_feature(adj));
                }
            }
            i += 1;
        }
        
        let mut edges = vec![vec![false; nodes.len()]; nodes.len()];
        for i in 0..nodes.len() {
            for adj in adjacent(nodes[i]) {
                if ids[adj as usize] != -1 {
                    edges[i][ids[adj as usize] as usize] = true;
                }
            }
        }

        GameGraph { nodes, edges, features }
    }
}
/* 
#[derive(Debug)]
pub enum GnnError {
    ModelLoad(String),
    Inference(String),
    InvalidInput(String),
}

impl std::fmt::Display for GnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GnnError::ModelLoad(msg) => write!(f, "Model loading error: {}", msg),
            GnnError::Inference(msg) => write!(f, "Inference error: {}", msg),
            GnnError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for GnnError {}

pub struct GnnEvaluator {
    session: Session,
    node_feature_dim: usize,
    max_nodes: usize,
}

impl GnnEvaluator {
    /// Load the ONNX model from file
    fn new(model_path: &str, node_feature_dim: usize, max_nodes: usize) -> Result<Self, GnnError> {
        // Initialize environment with GPU support
        ort::init()
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .commit()
            .map_err(|e| GnnError::ModelLoad(format!("Failed to create environment: {}", e)))?;

        let session = Session::builder()
            .map_err(|e| GnnError::ModelLoad(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            session,
            node_feature_dim,
            max_nodes,
        })
    }

    /// Convert adjacency matrix to edge index format
    fn create_edge_index(&self, graph: &GameGraph) -> Vec<i64> {
        let mut edge_index = Vec::new();
        
        for i in 0..graph.num_nodes() {
            for j in 0..graph.num_nodes() {
                if graph.edges[i][j] {
                    edge_index.push(i as i64);
                    edge_index.push(j as i64);
                }
            }
        }
        
        edge_index
    }

    /// Evaluate the game position using the GNN
    pub fn evaluate(&self, graph: &GameGraph) -> Result<f32, GnnError> {
        if graph.num_nodes() == 0 {
            return Err(GnnError::InvalidInput("Graph has no nodes".to_string()));
        }
        
        let num_nodes = graph.num_nodes();
        
        // Create edge index [2, num_edges]
        let edge_list = self.create_edge_index(graph);
        let num_edges = edge_list.len() / 2;
        
        if edge_list.is_empty() {
            return Err(GnnError::InvalidInput("Graph has no edges".to_string()));
        }

        // Flatten features into a single vector
        let features_flat: Vec<f32> = graph.features.iter()
            .flat_map(|f| f.iter().cloned())
            .collect();

        // Prepare ONNX inputs
        let node_features_array = Array2::from_shape_vec(
            (num_nodes, NUM_FEATURES), 
            features_flat
        ).map_err(|e| GnnError::Inference(format!("Failed to create node features array: {}", e)))?;

        // Reshape edge_list into [2, num_edges] format
        let edge_index_array = Array2::from_shape_vec(
            (2, num_edges),
            edge_list
        ).map_err(|e| GnnError::Inference(format!("Failed to create edge index array: {}", e)))?;

        // Get the allocator from the session
        let allocator = self.session.allocator();

        // Convert to ONNX Runtime Values
        let node_features_tensor = Tensor::from_array(node_features_array.into_dyn())
            .map_err(|e| GnnError::Inference(format!("Failed to create node features tensor: {}", e)))?;

        let edge_index_tensor = Tensor::from_array(edge_index_array.into_dyn())
            .map_err(|e| GnnError::Inference(format!("Failed to create edge index tensor: {}", e)))?;

        // Run inference
        let outputs = self.session
            .run(vec![node_features_tensor, edge_index_tensor])
            .map_err(|e| GnnError::Inference(format!("Inference failed: {}", e)))?;

        // Extract result
        let output = outputs
            .get(0)
            .ok_or_else(|| GnnError::Inference("No output from model".to_string()))?;

        // Extract the scalar value
        let output_tensor = output
            .try_extract::<f32>()
            .map_err(|e| GnnError::Inference(format!("Failed to extract tensor: {}", e)))?
            .view();
        
        let evaluation = output_tensor
            .iter()
            .next()
            .ok_or_else(|| GnnError::Inference("Output tensor is empty".to_string()))?;

        Ok(evaluation * 6000.0) // Scale the output to match the game score range
    }
}

    */