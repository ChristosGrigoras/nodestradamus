//! Rust graph algorithms for Nodestradamus.
//!
//! Provides high-performance implementations of graph algorithms
//! used for codebase intelligence and dependency analysis.

mod graph;

use pyo3::prelude::*;
use std::collections::HashMap;

/// PageRank algorithm for ranking node importance.
///
/// Higher scores indicate nodes that many other nodes depend on.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `alpha` - Damping factor (default 0.85)
/// * `max_iter` - Maximum iterations for convergence
///
/// # Returns
/// HashMap mapping node ID to importance score (0-1)
#[pyfunction]
#[pyo3(signature = (edges, alpha=0.85, max_iter=100))]
fn pagerank(
    edges: Vec<(String, String)>,
    alpha: f64,
    max_iter: usize,
) -> PyResult<HashMap<String, f64>> {
    graph::pagerank_impl(edges, alpha, max_iter)
}

/// Find strongly connected components (tightly coupled modules).
///
/// Returns groups where every node can reach every other node,
/// indicating high coupling.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
///
/// # Returns
/// List of node ID lists, each representing a strongly connected component
/// with 2+ nodes (single-node components are filtered out)
#[pyfunction]
fn strongly_connected(edges: Vec<(String, String)>) -> PyResult<Vec<Vec<String>>> {
    graph::strongly_connected_impl(edges)
}

/// Find all ancestors up to a maximum depth using BFS.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `target` - Starting node ID
/// * `max_depth` - Maximum traversal depth
///
/// # Returns
/// HashMap mapping ancestor node ID to its depth from the target
#[pyfunction]
fn ancestors_at_depth(
    edges: Vec<(String, String)>,
    target: String,
    max_depth: usize,
) -> PyResult<HashMap<String, usize>> {
    graph::ancestors_at_depth_impl(edges, target, max_depth)
}

/// Find all descendants up to a maximum depth using BFS.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `source` - Starting node ID
/// * `max_depth` - Maximum traversal depth
///
/// # Returns
/// HashMap mapping descendant node ID to its depth from the source
#[pyfunction]
fn descendants_at_depth(
    edges: Vec<(String, String)>,
    source: String,
    max_depth: usize,
) -> PyResult<HashMap<String, usize>> {
    graph::descendants_at_depth_impl(edges, source, max_depth)
}

/// Betweenness centrality using Brandes algorithm.
///
/// Finds nodes that sit on many shortest paths between other nodes.
/// High betweenness indicates bottleneck/gateway nodes.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `normalized` - Whether to normalize scores to 0-1 range (default: true)
///
/// # Returns
/// HashMap mapping node ID to betweenness score
#[pyfunction]
#[pyo3(signature = (edges, normalized=true))]
fn betweenness(
    edges: Vec<(String, String)>,
    normalized: bool,
) -> PyResult<HashMap<String, f64>> {
    graph::betweenness_impl(edges, normalized)
}

/// Sampled betweenness centrality for large graphs.
///
/// Uses random sampling of source nodes to approximate betweenness.
/// Much faster for large graphs: O(k*m) instead of O(n*m).
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `sample_size` - Number of random source nodes to sample (default: 100)
/// * `seed` - Random seed for reproducibility (default: 42)
/// * `normalized` - Whether to normalize scores to 0-1 range (default: true)
///
/// # Returns
/// HashMap mapping node ID to approximate betweenness score
#[pyfunction]
#[pyo3(signature = (edges, sample_size=100, seed=42, normalized=true))]
fn sampled_betweenness(
    edges: Vec<(String, String)>,
    sample_size: usize,
    seed: u64,
    normalized: bool,
) -> PyResult<HashMap<String, f64>> {
    graph::sampled_betweenness_impl(edges, sample_size, seed, normalized)
}

/// Batch ancestors: find ancestors for multiple targets in one graph build.
///
/// More efficient than calling ancestors_at_depth multiple times.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `targets` - List of target node IDs
/// * `max_depth` - Maximum traversal depth
///
/// # Returns
/// HashMap mapping target ID to HashMap of (ancestor ID -> depth)
#[pyfunction]
fn batch_ancestors(
    edges: Vec<(String, String)>,
    targets: Vec<String>,
    max_depth: usize,
) -> PyResult<HashMap<String, HashMap<String, usize>>> {
    graph::batch_ancestors_impl(edges, targets, max_depth)
}

/// Batch descendants: find descendants for multiple sources in one graph build.
///
/// More efficient than calling descendants_at_depth multiple times.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `sources` - List of source node IDs
/// * `max_depth` - Maximum traversal depth
///
/// # Returns
/// HashMap mapping source ID to HashMap of (descendant ID -> depth)
#[pyfunction]
fn batch_descendants(
    edges: Vec<(String, String)>,
    sources: Vec<String>,
    max_depth: usize,
) -> PyResult<HashMap<String, HashMap<String, usize>>> {
    graph::batch_descendants_impl(edges, sources, max_depth)
}

/// Extract a subgraph containing only nodes within max_depth of seed nodes.
///
/// Useful for lazy loading: load only the relevant portion of a large graph.
///
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `seed_nodes` - Starting nodes for the subgraph extraction
/// * `max_depth` - Maximum depth from seed nodes to include (default: 2)
/// * `include_incoming` - Include incoming edges (ancestors) (default: true)
/// * `include_outgoing` - Include outgoing edges (descendants) (default: true)
///
/// # Returns
/// Tuple of (nodes, edges) where edges is a list of (src, tgt) tuples
#[pyfunction]
#[pyo3(signature = (edges, seed_nodes, max_depth=2, include_incoming=true, include_outgoing=true))]
fn subgraph(
    edges: Vec<(String, String)>,
    seed_nodes: Vec<String>,
    max_depth: usize,
    include_incoming: bool,
    include_outgoing: bool,
) -> PyResult<(Vec<String>, Vec<(String, String)>)> {
    graph::subgraph_impl(edges, seed_nodes, max_depth, include_incoming, include_outgoing)
}

/// Python module definition for nodestradamus_graph.
#[pymodule]
fn nodestradamus_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(strongly_connected, m)?)?;
    m.add_function(wrap_pyfunction!(ancestors_at_depth, m)?)?;
    m.add_function(wrap_pyfunction!(descendants_at_depth, m)?)?;
    m.add_function(wrap_pyfunction!(betweenness, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_betweenness, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ancestors, m)?)?;
    m.add_function(wrap_pyfunction!(batch_descendants, m)?)?;
    m.add_function(wrap_pyfunction!(subgraph, m)?)?;
    Ok(())
}
