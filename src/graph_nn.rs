use ndarray::Array2;
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
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
    session: Arc<Session>,
    environment: Arc<Environment>,
}

impl GnnEvaluator {
    /// Load the ONNX model from file
    pub fn new(model_path: &str) -> Result<Self, GnnError> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("hive_gnn")
                .build()
                .map_err(|e| GnnError::ModelLoad(format!("Failed to create environment: {}", e)))?
        );

        let session = SessionBuilder::new(&environment)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(1)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to set thread count: {}", e)))?
            .with_model_from_file(model_path)
            .map_err(|e| GnnError::ModelLoad(format!("Failed to load model: {}", e)))?;

        Ok(Self { 
            session: Arc::new(session),
            environment,
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

        // Convert to dynamic dimension arrays
        let node_features_dyn = node_features_array.into_dyn();
        let edge_index_dyn = edge_index_array.into_dyn();

        // Convert to CowArray for ONNX Runtime Values
        use ndarray::CowArray;
        let node_features_cow = CowArray::from(node_features_dyn);
        let edge_index_cow = CowArray::from(edge_index_dyn);

        let node_features_tensor = Value::from_array(self.session.allocator(), &node_features_cow)
            .map_err(|e| GnnError::Inference(format!("Failed to create node features tensor: {}", e)))?;

        let edge_index_tensor = Value::from_array(self.session.allocator(), &edge_index_cow)
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
        let extracted = output
            .try_extract::<f32>()
            .map_err(|e| GnnError::Inference(format!("Failed to extract tensor: {}", e)))?;
        let output_tensor = extracted.view();
        
        let evaluation = output_tensor
            .iter()
            .next()
            .copied()
            .ok_or_else(|| GnnError::Inference("Output tensor is empty".to_string()))?;

        Ok(evaluation * 6000.0) // Scale the output to match the game score range
    }
}
