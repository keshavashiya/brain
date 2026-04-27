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
    ///
    /// `succeeded` must contain only steps that completed *successfully* —
    /// passing in the broader "terminal" set would let a failed step's
    /// dependents fire against missing artifacts. The orchestrator marks
    /// dependents of a failed step as `Skipped` separately.
    ///
    /// Result is sorted by topological order index so plans run in the
    /// order the decomposer intended, not in `HashMap` iteration order.
    pub fn ready_steps(&self, succeeded: &HashSet<String>) -> Vec<String> {
        let order = self.topological_order();
        let rank: HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let mut ready: Vec<String> = self
            .steps
            .values()
            .filter(|step| {
                !succeeded.contains(&step.id)
                    && step.depends_on.iter().all(|dep| succeeded.contains(dep))
            })
            .map(|s| s.id.clone())
            .collect();
        ready.sort_by_key(|id| rank.get(id.as_str()).copied().unwrap_or(usize::MAX));
        ready
    }

    /// Return all steps that transitively depend on `step_id` (excluding
    /// `step_id` itself). Used by the orchestrator to mark dependents
    /// `Skipped` when an upstream step fails.
    pub fn transitive_dependents(&self, step_id: &str) -> Vec<String> {
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, to) in &self.edges {
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
        }

        let mut out = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        if let Some(starts) = adjacency.get(step_id) {
            for &s in starts {
                queue.push_back(s);
            }
        }
        while let Some(node) = queue.pop_front() {
            if !seen.insert(node.to_string()) {
                continue;
            }
            out.push(node.to_string());
            if let Some(nexts) = adjacency.get(node) {
                for &n in nexts {
                    queue.push_back(n);
                }
            }
        }
        out
    }

    /// Topological sort — returns step IDs in execution order.
    ///
    /// Order is deterministic: ties (independent steps with the same
    /// in-degree) break by step id. Without this, two unrelated no-dep
    /// steps would execute in `HashMap` iteration order, which meant a
    /// "Step 1 / Step 2" plan could run in reverse on a given run.
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

        // BinaryHeap with Reverse → pop smallest id first → stable order.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut queue: BinaryHeap<Reverse<&str>> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| Reverse(id))
            .collect();

        let mut order = Vec::new();
        while let Some(Reverse(node)) = queue.pop() {
            order.push(node.to_string());
            for &next in adjacency.get(node).unwrap_or(&vec![]) {
                let deg = in_degree
                    .get_mut(next)
                    .expect("invariant: every step id seeded into in_degree map at start");
                *deg -= 1;
                if *deg == 0 {
                    queue.push(Reverse(next));
                }
            }
        }

        order
    }

    /// Splice additional steps into an existing graph. Used by the
    /// replan-on-failure loop — after an initial step fails, the
    /// orchestrator asks the decomposer for a corrective sub-plan and
    /// inserts the resulting steps so the execution loop can pick them
    /// up on the next iteration. Each new step's `depends_on` may
    /// reference either an existing step id or another new step.
    pub fn add_steps(&mut self, new_steps: Vec<TaskStep>) -> Result<(), GraphError> {
        // Pre-validate against the union of existing + incoming step ids
        // so a new step that depends on a sibling new step is allowed.
        let mut universe: HashSet<String> = self.steps.keys().cloned().collect();
        for s in &new_steps {
            universe.insert(s.id.clone());
        }
        for s in &new_steps {
            for dep in &s.depends_on {
                if !universe.contains(dep) {
                    return Err(GraphError::MissingDependency {
                        step: s.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }
        for s in new_steps {
            for dep in &s.depends_on {
                self.edges.push((dep.clone(), s.id.clone()));
            }
            self.steps.insert(s.id.clone(), s);
        }
        // Re-validate for cycles introduced by the splice.
        self.validate()
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

    #[test]
    fn test_transitive_dependents() {
        let steps = vec![
            make_step("a", vec![]),
            make_step("b", vec!["a"]),
            make_step("c", vec!["a"]),
            make_step("d", vec!["b", "c"]),
            make_step("e", vec!["d"]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();

        let mut deps = graph.transitive_dependents("a");
        deps.sort();
        assert_eq!(deps, vec!["b", "c", "d", "e"]);

        let mut deps = graph.transitive_dependents("b");
        deps.sort();
        assert_eq!(deps, vec!["d", "e"]);

        assert!(graph.transitive_dependents("e").is_empty());
    }

    #[test]
    fn test_topological_order_is_deterministic() {
        // Three independent steps — without tie-breaking these would
        // come back in HashMap iteration order. We need a stable order
        // so plans like "step 1 / step 2 / step 3" run as written.
        let steps = vec![
            make_step("c", vec![]),
            make_step("a", vec![]),
            make_step("b", vec![]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();
        let first = graph.topological_order();
        for _ in 0..20 {
            assert_eq!(graph.topological_order(), first);
        }
        assert_eq!(first, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_ready_steps_returns_deterministic_order() {
        let steps = vec![
            make_step("c", vec![]),
            make_step("a", vec![]),
            make_step("b", vec!["a"]),
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();
        let ready = graph.ready_steps(&HashSet::new());
        // `a` must come before `c` because that's its topological order.
        // Without the sort, `c` could come first due to HashMap order.
        let pos_a = ready.iter().position(|s| s == "a").unwrap();
        let pos_c = ready.iter().position(|s| s == "c").unwrap();
        assert!(pos_a < pos_c);
    }
}
