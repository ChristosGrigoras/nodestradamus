//! Graph building utilities and algorithm implementations.

use petgraph::algo::kosaraju_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, VecDeque};

/// Build a petgraph DiGraph from a list of edge tuples.
///
/// Returns the graph and a mapping from node names to indices.
pub fn build_graph(edges: Vec<(String, String)>) -> (DiGraph<String, ()>, FxHashMap<String, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut node_map: FxHashMap<String, NodeIndex> = FxHashMap::default();

    for (src, tgt) in edges {
        let src_idx = *node_map
            .entry(src.clone())
            .or_insert_with(|| graph.add_node(src));
        let tgt_idx = *node_map
            .entry(tgt.clone())
            .or_insert_with(|| graph.add_node(tgt));
        graph.add_edge(src_idx, tgt_idx, ());
    }

    (graph, node_map)
}

/// Create a reverse mapping from NodeIndex to node name.
fn index_to_name(node_map: &FxHashMap<String, NodeIndex>) -> HashMap<NodeIndex, String> {
    node_map.iter().map(|(k, v)| (*v, k.clone())).collect()
}

/// PageRank implementation using power iteration.
pub fn pagerank_impl(
    edges: Vec<(String, String)>,
    alpha: f64,
    max_iter: usize,
) -> PyResult<HashMap<String, f64>> {
    let (graph, node_map) = build_graph(edges);
    let n = graph.node_count();

    if n == 0 {
        return Ok(HashMap::new());
    }

    let id_to_name = index_to_name(&node_map);

    // Initialize scores uniformly
    let mut scores: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_scores = vec![0.0; n];

    // Pre-compute out-degrees for efficiency
    let out_degrees: Vec<usize> = graph
        .node_indices()
        .map(|node| graph.edges(node).count())
        .collect();

    for _ in 0..max_iter {
        // Reset new scores with base probability
        for score in new_scores.iter_mut() {
            *score = (1.0 - alpha) / n as f64;
        }

        // Distribute rank from each node to its successors
        for node in graph.node_indices() {
            let out_degree = out_degrees[node.index()];
            if out_degree > 0 {
                let contribution = alpha * scores[node.index()] / out_degree as f64;
                for edge in graph.edges(node) {
                    let target = edge.target();
                    new_scores[target.index()] += contribution;
                }
            } else {
                // Dangling node: distribute evenly to all nodes
                let contribution = alpha * scores[node.index()] / n as f64;
                for score in new_scores.iter_mut() {
                    *score += contribution;
                }
            }
        }

        std::mem::swap(&mut scores, &mut new_scores);
    }

    // Map back to node names
    let result: HashMap<String, f64> = graph
        .node_indices()
        .map(|idx| (id_to_name[&idx].clone(), scores[idx.index()]))
        .collect();

    Ok(result)
}

/// Find strongly connected components using Kosaraju's algorithm.
pub fn strongly_connected_impl(edges: Vec<(String, String)>) -> PyResult<Vec<Vec<String>>> {
    let (graph, node_map) = build_graph(edges);
    let id_to_name = index_to_name(&node_map);

    let sccs = kosaraju_scc(&graph);

    // Filter to components with 2+ nodes and convert to names
    let mut result: Vec<Vec<String>> = sccs
        .into_iter()
        .filter(|scc| scc.len() > 1)
        .map(|scc| scc.iter().map(|idx| id_to_name[idx].clone()).collect())
        .collect();

    // Sort by size (largest first)
    result.sort_by(|a, b| b.len().cmp(&a.len()));

    Ok(result)
}

/// BFS to find ancestors up to max_depth.
pub fn ancestors_at_depth_impl(
    edges: Vec<(String, String)>,
    target: String,
    max_depth: usize,
) -> PyResult<HashMap<String, usize>> {
    let (graph, node_map) = build_graph(edges);

    let Some(&target_idx) = node_map.get(&target) else {
        return Ok(HashMap::new());
    };

    let id_to_name = index_to_name(&node_map);
    let mut result = HashMap::new();
    let mut visited = FxHashMap::default();
    let mut queue = VecDeque::new();

    visited.insert(target_idx, 0usize);
    queue.push_back((target_idx, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // Find predecessors (incoming edges)
        for edge in graph.edges_directed(node, Direction::Incoming) {
            let pred = edge.source();
            if !visited.contains_key(&pred) {
                let new_depth = depth + 1;
                visited.insert(pred, new_depth);
                result.insert(id_to_name[&pred].clone(), new_depth);
                queue.push_back((pred, new_depth));
            }
        }
    }

    Ok(result)
}

/// BFS to find descendants up to max_depth.
pub fn descendants_at_depth_impl(
    edges: Vec<(String, String)>,
    source: String,
    max_depth: usize,
) -> PyResult<HashMap<String, usize>> {
    let (graph, node_map) = build_graph(edges);

    let Some(&source_idx) = node_map.get(&source) else {
        return Ok(HashMap::new());
    };

    let id_to_name = index_to_name(&node_map);
    let mut result = HashMap::new();
    let mut visited = FxHashMap::default();
    let mut queue = VecDeque::new();

    visited.insert(source_idx, 0usize);
    queue.push_back((source_idx, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // Find successors (outgoing edges)
        for edge in graph.edges_directed(node, Direction::Outgoing) {
            let succ = edge.target();
            if !visited.contains_key(&succ) {
                let new_depth = depth + 1;
                visited.insert(succ, new_depth);
                result.insert(id_to_name[&succ].clone(), new_depth);
                queue.push_back((succ, new_depth));
            }
        }
    }

    Ok(result)
}

/// Betweenness centrality using Brandes algorithm.
/// 
/// Finds nodes that sit on many shortest paths between other nodes.
/// High betweenness indicates bottleneck/gateway nodes.
pub fn betweenness_impl(
    edges: Vec<(String, String)>,
    normalized: bool,
) -> PyResult<HashMap<String, f64>> {
    let (graph, node_map) = build_graph(edges);
    betweenness_on_graph(&graph, &node_map, normalized, None)
}

/// Core betweenness implementation on an existing graph.
/// 
/// If `sample_sources` is Some, only uses those nodes as sources (for sampling).
fn betweenness_on_graph(
    graph: &DiGraph<String, ()>,
    node_map: &FxHashMap<String, NodeIndex>,
    normalized: bool,
    sample_sources: Option<&[NodeIndex]>,
) -> PyResult<HashMap<String, f64>> {
    let n = graph.node_count();
    
    if n == 0 {
        return Ok(HashMap::new());
    }
    
    let id_to_name = index_to_name(node_map);
    
    // Initialize betweenness scores
    let mut centrality: Vec<f64> = vec![0.0; n];
    
    // Determine which sources to iterate over
    let sources: Vec<NodeIndex> = match sample_sources {
        Some(samples) => samples.to_vec(),
        None => graph.node_indices().collect(),
    };
    let num_sources = sources.len();
    
    // Brandes algorithm: O(nm) for unweighted graphs
    for source in sources {
        // Single-source shortest paths
        let mut stack: Vec<NodeIndex> = Vec::new();
        let mut predecessors: Vec<Vec<NodeIndex>> = vec![Vec::new(); n];
        let mut sigma: Vec<f64> = vec![0.0; n];  // Number of shortest paths
        let mut dist: Vec<i32> = vec![-1; n];    // Distance from source
        let mut delta: Vec<f64> = vec![0.0; n];  // Dependency
        
        sigma[source.index()] = 1.0;
        dist[source.index()] = 0;
        
        // BFS to find shortest paths
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();
        queue.push_back(source);
        
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            
            for edge in graph.edges_directed(v, Direction::Outgoing) {
                let w = edge.target();
                
                // First time seeing w?
                if dist[w.index()] < 0 {
                    dist[w.index()] = dist[v.index()] + 1;
                    queue.push_back(w);
                }
                
                // Is this a shortest path to w via v?
                if dist[w.index()] == dist[v.index()] + 1 {
                    sigma[w.index()] += sigma[v.index()];
                    predecessors[w.index()].push(v);
                }
            }
        }
        
        // Accumulation phase - back-propagate dependencies
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w.index()] {
                let contrib = (sigma[v.index()] / sigma[w.index()]) * (1.0 + delta[w.index()]);
                delta[v.index()] += contrib;
            }
            
            if w != source {
                centrality[w.index()] += delta[w.index()];
            }
        }
    }
    
    // Scale factor for sampling: extrapolate from sample to full graph
    let sample_scale = if sample_sources.is_some() && num_sources > 0 {
        n as f64 / num_sources as f64
    } else {
        1.0
    };
    
    // Normalize if requested (for undirected: 2/((n-1)(n-2)), for directed: 1/((n-1)(n-2)))
    let norm_scale = if normalized && n > 2 {
        1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
    } else {
        1.0
    };
    
    let total_scale = sample_scale * norm_scale;
    for c in centrality.iter_mut() {
        *c *= total_scale;
    }
    
    // Convert to HashMap
    let result: HashMap<String, f64> = graph
        .node_indices()
        .map(|idx| (id_to_name[&idx].clone(), centrality[idx.index()]))
        .collect();
    
    Ok(result)
}

/// Sampled betweenness centrality for large graphs.
/// 
/// Uses random sampling of source nodes to approximate betweenness.
/// Much faster for large graphs: O(k*m) instead of O(n*m).
/// 
/// # Arguments
/// * `edges` - List of (source, target) edge tuples
/// * `sample_size` - Number of random source nodes to sample
/// * `seed` - Random seed for reproducibility
/// * `normalized` - Whether to normalize scores to 0-1 range
pub fn sampled_betweenness_impl(
    edges: Vec<(String, String)>,
    sample_size: usize,
    seed: u64,
    normalized: bool,
) -> PyResult<HashMap<String, f64>> {
    let (graph, node_map) = build_graph(edges);
    let n = graph.node_count();
    
    if n == 0 {
        return Ok(HashMap::new());
    }
    
    // If sample_size >= n, just do full betweenness
    if sample_size >= n {
        return betweenness_on_graph(&graph, &node_map, normalized, None);
    }
    
    // Simple deterministic sampling based on seed
    // Use a linear congruential generator for reproducibility
    let mut rng_state = seed;
    let mut indices: Vec<usize> = (0..n).collect();
    
    // Fisher-Yates shuffle with LCG
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }
    
    // Take first sample_size indices
    let sample_sources: Vec<NodeIndex> = indices[..sample_size]
        .iter()
        .map(|&i| NodeIndex::new(i))
        .collect();
    
    betweenness_on_graph(&graph, &node_map, normalized, Some(&sample_sources))
}

/// Batch ancestors: find ancestors for multiple targets in one graph build.
/// 
/// More efficient than calling ancestors_at_depth multiple times.
pub fn batch_ancestors_impl(
    edges: Vec<(String, String)>,
    targets: Vec<String>,
    max_depth: usize,
) -> PyResult<HashMap<String, HashMap<String, usize>>> {
    let (graph, node_map) = build_graph(edges);
    let id_to_name = index_to_name(&node_map);
    
    let mut results: HashMap<String, HashMap<String, usize>> = HashMap::new();
    
    for target in targets {
        let Some(&target_idx) = node_map.get(&target) else {
            results.insert(target, HashMap::new());
            continue;
        };
        
        let mut ancestors = HashMap::new();
        let mut visited = FxHashMap::default();
        let mut queue = VecDeque::new();
        
        visited.insert(target_idx, 0usize);
        queue.push_back((target_idx, 0usize));
        
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            
            for edge in graph.edges_directed(node, Direction::Incoming) {
                let pred = edge.source();
                if !visited.contains_key(&pred) {
                    let new_depth = depth + 1;
                    visited.insert(pred, new_depth);
                    ancestors.insert(id_to_name[&pred].clone(), new_depth);
                    queue.push_back((pred, new_depth));
                }
            }
        }
        
        results.insert(target, ancestors);
    }
    
    Ok(results)
}

/// Batch descendants: find descendants for multiple sources in one graph build.
/// 
/// More efficient than calling descendants_at_depth multiple times.
pub fn batch_descendants_impl(
    edges: Vec<(String, String)>,
    sources: Vec<String>,
    max_depth: usize,
) -> PyResult<HashMap<String, HashMap<String, usize>>> {
    let (graph, node_map) = build_graph(edges);
    let id_to_name = index_to_name(&node_map);
    
    let mut results: HashMap<String, HashMap<String, usize>> = HashMap::new();
    
    for source in sources {
        let Some(&source_idx) = node_map.get(&source) else {
            results.insert(source, HashMap::new());
            continue;
        };
        
        let mut descendants = HashMap::new();
        let mut visited = FxHashMap::default();
        let mut queue = VecDeque::new();
        
        visited.insert(source_idx, 0usize);
        queue.push_back((source_idx, 0usize));
        
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            
            for edge in graph.edges_directed(node, Direction::Outgoing) {
                let succ = edge.target();
                if !visited.contains_key(&succ) {
                    let new_depth = depth + 1;
                    visited.insert(succ, new_depth);
                    descendants.insert(id_to_name[&succ].clone(), new_depth);
                    queue.push_back((succ, new_depth));
                }
            }
        }
        
        results.insert(source, descendants);
    }
    
    Ok(results)
}

/// Extract a subgraph containing only nodes within max_depth of seed nodes.
/// 
/// Useful for lazy loading: load only the relevant portion of a large graph.
/// Returns (nodes, edges) where edges is a list of (src, tgt) tuples.
pub fn subgraph_impl(
    edges: Vec<(String, String)>,
    seed_nodes: Vec<String>,
    max_depth: usize,
    include_incoming: bool,
    include_outgoing: bool,
) -> PyResult<(Vec<String>, Vec<(String, String)>)> {
    let (graph, node_map) = build_graph(edges);
    let id_to_name = index_to_name(&node_map);
    
    // Find all nodes within max_depth of any seed node
    let mut relevant_nodes: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
    
    // Initialize with seed nodes
    for seed in &seed_nodes {
        if let Some(&idx) = node_map.get(seed) {
            relevant_nodes.insert(idx, 0);
            queue.push_back((idx, 0));
        }
    }
    
    // BFS in both directions
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        
        // Outgoing edges (descendants)
        if include_outgoing {
            for edge in graph.edges_directed(node, Direction::Outgoing) {
                let neighbor = edge.target();
                if !relevant_nodes.contains_key(&neighbor) {
                    relevant_nodes.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        
        // Incoming edges (ancestors)
        if include_incoming {
            for edge in graph.edges_directed(node, Direction::Incoming) {
                let neighbor = edge.source();
                if !relevant_nodes.contains_key(&neighbor) {
                    relevant_nodes.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
    }
    
    // Collect nodes
    let nodes: Vec<String> = relevant_nodes
        .keys()
        .map(|idx| id_to_name[idx].clone())
        .collect();
    
    // Collect edges where both endpoints are in relevant_nodes
    let mut subgraph_edges: Vec<(String, String)> = Vec::new();
    for edge in graph.edge_references() {
        let src = edge.source();
        let tgt = edge.target();
        if relevant_nodes.contains_key(&src) && relevant_nodes.contains_key(&tgt) {
            subgraph_edges.push((id_to_name[&src].clone(), id_to_name[&tgt].clone()));
        }
    }
    
    Ok((nodes, subgraph_edges))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_graph_creates_nodes_and_edges() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let (graph, node_map) = build_graph(edges);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(node_map.contains_key("a"));
        assert!(node_map.contains_key("b"));
        assert!(node_map.contains_key("c"));
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let result = pagerank_impl(vec![], 0.85, 100).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_pagerank_scores_sum_to_one() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "a".to_string()),
        ];
        let result = pagerank_impl(edges, 0.85, 100).unwrap();
        let total: f64 = result.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_strongly_connected_finds_cycle() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "a".to_string()),
        ];
        let result = strongly_connected_impl(edges).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_ancestors_at_depth() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let result = ancestors_at_depth_impl(edges, "c".to_string(), 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("b"), Some(&1));
    }

    #[test]
    fn test_descendants_at_depth() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let result = descendants_at_depth_impl(edges, "a".to_string(), 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("b"), Some(&1));
    }

    #[test]
    fn test_betweenness_empty_graph() {
        let result = betweenness_impl(vec![], true).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_betweenness_bridge_node() {
        // a -> b -> c: b is the bridge, should have highest betweenness
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let result = betweenness_impl(edges, false).unwrap();
        
        // b sits on the path from a to c
        let a_score = result.get("a").unwrap_or(&0.0);
        let b_score = result.get("b").unwrap_or(&0.0);
        let c_score = result.get("c").unwrap_or(&0.0);
        
        // b should have highest betweenness
        assert!(*b_score >= *a_score);
        assert!(*b_score >= *c_score);
    }

    #[test]
    fn test_betweenness_normalized() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "d".to_string()),
        ];
        let result = betweenness_impl(edges, true).unwrap();
        
        // All normalized scores should be <= 1.0
        for score in result.values() {
            assert!(*score <= 1.0);
        }
    }

    #[test]
    fn test_sampled_betweenness_returns_all_nodes() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "d".to_string()),
            ("d".to_string(), "e".to_string()),
        ];
        let result = sampled_betweenness_impl(edges, 3, 42, true).unwrap();
        
        // Should return scores for all nodes
        assert_eq!(result.len(), 5);
        assert!(result.contains_key("a"));
        assert!(result.contains_key("c"));
        assert!(result.contains_key("e"));
    }

    #[test]
    fn test_sampled_betweenness_reproducible() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "d".to_string()),
        ];
        
        // Same seed should give same results
        let result1 = sampled_betweenness_impl(edges.clone(), 2, 123, true).unwrap();
        let result2 = sampled_betweenness_impl(edges, 2, 123, true).unwrap();
        
        for (k, v1) in &result1 {
            let v2 = result2.get(k).unwrap();
            assert!((v1 - v2).abs() < 0.0001);
        }
    }

    #[test]
    fn test_batch_ancestors() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("x".to_string(), "c".to_string()),
        ];
        let result = batch_ancestors_impl(edges, vec!["c".to_string(), "b".to_string()], 2).unwrap();
        
        assert_eq!(result.len(), 2);
        
        // Ancestors of c: b (depth 1), a (depth 2), x (depth 1)
        let c_ancestors = result.get("c").unwrap();
        assert_eq!(c_ancestors.get("b"), Some(&1));
        assert_eq!(c_ancestors.get("x"), Some(&1));
        assert_eq!(c_ancestors.get("a"), Some(&2));
        
        // Ancestors of b: a (depth 1)
        let b_ancestors = result.get("b").unwrap();
        assert_eq!(b_ancestors.get("a"), Some(&1));
    }

    #[test]
    fn test_batch_ancestors_missing_target() {
        let edges = vec![("a".to_string(), "b".to_string())];
        let result = batch_ancestors_impl(edges, vec!["missing".to_string()], 2).unwrap();
        
        assert_eq!(result.len(), 1);
        assert!(result.get("missing").unwrap().is_empty());
    }

    #[test]
    fn test_batch_descendants() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("a".to_string(), "x".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let result = batch_descendants_impl(edges, vec!["a".to_string(), "b".to_string()], 2).unwrap();
        
        assert_eq!(result.len(), 2);
        
        // Descendants of a: b (depth 1), x (depth 1), c (depth 2)
        let a_descendants = result.get("a").unwrap();
        assert_eq!(a_descendants.get("b"), Some(&1));
        assert_eq!(a_descendants.get("x"), Some(&1));
        assert_eq!(a_descendants.get("c"), Some(&2));
        
        // Descendants of b: c (depth 1)
        let b_descendants = result.get("b").unwrap();
        assert_eq!(b_descendants.get("c"), Some(&1));
    }

    #[test]
    fn test_subgraph_both_directions() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "d".to_string()),
            ("x".to_string(), "y".to_string()),  // Disconnected
        ];
        let (nodes, subgraph_edges) = subgraph_impl(
            edges, 
            vec!["b".to_string()], 
            1, 
            true, 
            true
        ).unwrap();
        
        // Should include b, a (ancestor), c (descendant)
        assert!(nodes.contains(&"b".to_string()));
        assert!(nodes.contains(&"a".to_string()));
        assert!(nodes.contains(&"c".to_string()));
        assert!(!nodes.contains(&"d".to_string()));  // Too far
        assert!(!nodes.contains(&"x".to_string()));  // Disconnected
        
        // Should include edges between included nodes
        assert!(subgraph_edges.contains(&("a".to_string(), "b".to_string())));
        assert!(subgraph_edges.contains(&("b".to_string(), "c".to_string())));
    }

    #[test]
    fn test_subgraph_outgoing_only() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "d".to_string()),
        ];
        let (nodes, _) = subgraph_impl(
            edges, 
            vec!["b".to_string()], 
            2, 
            false,  // No incoming
            true    // Outgoing only
        ).unwrap();
        
        // Should include b, c, d (descendants only)
        assert!(nodes.contains(&"b".to_string()));
        assert!(nodes.contains(&"c".to_string()));
        assert!(nodes.contains(&"d".to_string()));
        assert!(!nodes.contains(&"a".to_string()));  // Ancestor excluded
    }

    #[test]
    fn test_subgraph_multiple_seeds() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("x".to_string(), "y".to_string()),
        ];
        let (nodes, subgraph_edges) = subgraph_impl(
            edges, 
            vec!["a".to_string(), "x".to_string()], 
            1, 
            true, 
            true
        ).unwrap();
        
        // Should include nodes from both seeds
        assert!(nodes.contains(&"a".to_string()));
        assert!(nodes.contains(&"b".to_string()));
        assert!(nodes.contains(&"x".to_string()));
        assert!(nodes.contains(&"y".to_string()));
        
        assert_eq!(subgraph_edges.len(), 2);
    }
}
