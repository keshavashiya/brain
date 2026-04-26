//! DAG dependency graph with parallel execution support.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::step::TaskStep;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Cycle detected in task graph")]
    CycleDetected,
    #[error("Missing dependency: step {step} depends on {dependency} which does not exist")]
    MissingDependency { step: String, dependency: String },
    #[error("Step not found: {0}")]
    StepNotFound(String),
}

/// Rollback action for a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackAction {
    pub step_id: String,
    pub description: String,
    pub command: Option<String>,
}

/// Directed acyclic graph of task steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraph {
    pub steps: HashMap<String, TaskStep>,
    pub edges: Vec<(String, String)>, // (from, to) = from must complete before to
}

impl TaskGraph {
    /// Create a new graph from a list of steps.
    /// Edges are derived from each step's `depends_on` field.
    pub fn from_steps(steps: Vec<TaskStep>) -> Result<Self, GraphError> {
        let step_map: HashMap<String, TaskStep> =
            steps.into_iter().map(|s| (s.id.clone(), s)).collect();

        let mut edges = Vec::new();
        for step in step_map.values() {
            for dep in &step.depends_on {
                if !step_map.contains_key(dep) {
                    return Err(GraphError::MissingDependency {
                        step: step.id.clone(),
                        dependency: dep.clone(),
                    });
                }
                edges.push((dep.clone(), step.id.clone()));
            }
        }

        let graph = Self {
            steps: step_map,
            edges,
        };
        graph.validate()?;
        Ok(graph)
    }

    /// Validate that the graph has no cycles (topological sort).
    pub fn validate(&self) -> Result<(), GraphError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in self.steps.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
            adjacency.entry(id.as_str()).or_default();
        }

        for (from, to) in &self.edges {
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            for &next in adjacency.get(node).unwrap_or(&vec![]) {
                let deg = in_degree
                    .get_mut(next)
                    .expect("invariant: every step id seeded into in_degree map at start");
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(next);
                }
            }
        }

        if visited != self.steps.len() {
            return Err(GraphError::CycleDetected);
        }
        Ok(())
    }

    /// Return step IDs that are ready to execute (all dependencies satisfied).
    pub fn ready_steps(&self, completed: &HashSet<String>) -> Vec<String> {
        self.steps
            .values()
            .filter(|step| {
                !completed.contains(&step.id)
                    && step.depends_on.iter().all(|dep| completed.contains(dep))
            })
            .map(|s| s.id.clone())
            .collect()
    }

    /// Topological sort — returns step IDs in execution order.
    pub fn topological_order(&self) -> Vec<String> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in self.steps.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
            adjacency.entry(id.as_str()).or_default();
        }

        for (from, to) in &self.edges {
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for &next in adjacency.get(node).unwrap_or(&vec![]) {
                let deg = in_degree
                    .get_mut(next)
                    .expect("invariant: every step id seeded into in_degree map at start");
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(next);
                }
            }
        }

        order
    }

    /// Reverse topological order — for rollback.
    pub fn rollback_order(&self, from_step: &str) -> Vec<RollbackAction> {
        let order = self.topological_order();
        let mut result = Vec::new();

        // Find all steps that were completed before (and including) the failed step
        let mut include = false;
        for id in order.iter().rev() {
            if id == from_step {
                include = true;
            }
            if include {
                if let Some(step) = self.steps.get(id) {
                    result.push(RollbackAction {
                        step_id: id.clone(),
                        description: format!("Rollback: {}", step.description),
                        command: None,
                    });
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{StepAction, TaskStep};
    use audit::ActionTier;

    fn make_step(id: &str, deps: Vec<&str>) -> TaskStep {
        TaskStep {
            id: id.to_string(),
            description: format!("Step {id}"),
            action: StepAction::Plan {
                output: "plan".to_string(),
            },
            depends_on: deps.into_iter().map(String::from).collect(),
            tier: ActionTier::Execute,
            estimated_tokens: 0,
        }
    }

    #[test]
    fn test_valid_graph() {
        let steps = vec![
            make_step("a", vec![]),
            make_step("b", vec!["a"]),
            make_step("c", vec!["a"]),
            make_step("d", vec!["b", "c"]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();
        assert_eq!(graph.steps.len(), 4);
        assert_eq!(graph.edges.len(), 4); // a→b, a→c, b→d, c→d
    }

    #[test]
    fn test_cycle_detected() {
        let steps = vec![
            make_step("a", vec!["c"]),
            make_step("b", vec!["a"]),
            make_step("c", vec!["b"]),
        ];
        let result = TaskGraph::from_steps(steps);
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_missing_dependency() {
        let steps = vec![make_step("a", vec!["nonexistent"])];
        let result = TaskGraph::from_steps(steps);
        assert!(matches!(result, Err(GraphError::MissingDependency { .. })));
    }

    #[test]
    fn test_ready_steps() {
        let steps = vec![
            make_step("a", vec![]),
            make_step("b", vec!["a"]),
            make_step("c", vec![]),
            make_step("d", vec!["b", "c"]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();

        let completed = HashSet::new();
        let mut ready = graph.ready_steps(&completed);
        ready.sort();
        assert_eq!(ready, vec!["a", "c"]);

        let completed: HashSet<String> = ["a".to_string()].into();
        let mut ready = graph.ready_steps(&completed);
        ready.sort();
        assert_eq!(ready, vec!["b", "c"]);

        let completed: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let ready = graph.ready_steps(&completed);
        assert_eq!(ready, vec!["d"]);
    }

    #[test]
    fn test_topological_order() {
        let steps = vec![
            make_step("a", vec![]),
            make_step("b", vec!["a"]),
            make_step("c", vec!["b"]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();
        let order = graph.topological_order();
        assert_eq!(order, vec!["a", "b", "c"]);
    }
}
