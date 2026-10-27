use hashbrown::HashMap;
// Float maths through libm without std. With std the inherent methods take precedence.
#[allow(unused_imports)]
use nalgebra::ComplexField;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeavyPhase {
    Preprocessing,
    PointDeskew,
    PlaneCorrespondence,
    MapUpdate,
}

impl HeavyPhase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Preprocessing => "preprocessing",
            Self::PointDeskew => "point deskew",
            Self::PlaneCorrespondence => "plane correspondence",
            Self::MapUpdate => "map update",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Rayon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionPolicy {
    #[default]
    Auto,
    SingleCore,
    Rayon,
}

impl ExecutionPolicy {
    fn forced_mode(self) -> Option<ExecutionMode> {
        match self {
            Self::Auto => None,
            Self::SingleCore => Some(ExecutionMode::Sequential),
            Self::Rayon => Some(ExecutionMode::Rayon),
        }
    }
}

impl ExecutionMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Sequential => "single-core",
            Self::Rayon => "Rayon",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionDecision {
    pub mode: ExecutionMode,
    pub speedup: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    squared_deviation_sum: f64,
}

impl RunningStats {
    fn record(&mut self, sample: f64) {
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta_after_mean_update = sample - self.mean;
        self.squared_deviation_sum += delta * delta_after_mean_update;
    }

    fn variance(self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.squared_deviation_sum / (self.count - 1) as f64
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct UndecidedState {
    sequential: RunningStats,
    rayon: RunningStats,
}

impl UndecidedState {
    fn stats_mut(&mut self, mode: ExecutionMode) -> &mut RunningStats {
        match mode {
            ExecutionMode::Sequential => &mut self.sequential,
            ExecutionMode::Rayon => &mut self.rayon,
        }
    }

    fn next_mode(self) -> ExecutionMode {
        if self.sequential.count <= self.rayon.count {
            ExecutionMode::Sequential
        } else {
            ExecutionMode::Rayon
        }
    }

    fn significant_winner(
        self,
        threshold_stddevs: f64,
        samples_per_mode: usize,
    ) -> Option<ExecutionDecision> {
        if self.sequential.count < samples_per_mode || self.rayon.count < samples_per_mode {
            return None;
        }

        let standard_error = (self.sequential.variance() / self.sequential.count as f64
            + self.rayon.variance() / self.rayon.count as f64)
            .sqrt();
        let mean_difference = (self.sequential.mean - self.rayon.mean).abs();

        if mean_difference <= threshold_stddevs * standard_error {
            return None;
        }

        let (mode, faster_mean, slower_mean) = if self.sequential.mean < self.rayon.mean {
            (
                ExecutionMode::Sequential,
                self.sequential.mean,
                self.rayon.mean,
            )
        } else {
            (ExecutionMode::Rayon, self.rayon.mean, self.sequential.mean)
        };

        Some(ExecutionDecision {
            mode,
            speedup: slower_mean / faster_mean,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum PhaseState {
    Undecided(UndecidedState),
    Decided(ExecutionMode),
}

impl Default for PhaseState {
    fn default() -> Self {
        Self::Undecided(UndecidedState::default())
    }
}

#[derive(Debug)]
pub struct AdaptiveExecutionPlanner {
    threshold_stddevs: f64,
    samples_per_mode: usize,
    policy: ExecutionPolicy,
    phases: HashMap<HeavyPhase, PhaseState>,
}

impl AdaptiveExecutionPlanner {
    pub fn new(threshold_stddevs: f64, samples_per_mode: usize, policy: ExecutionPolicy) -> Self {
        Self {
            threshold_stddevs,
            samples_per_mode,
            policy,
            phases: HashMap::new(),
        }
    }

    pub fn mode_for(&mut self, phase: HeavyPhase) -> ExecutionMode {
        if let Some(mode) = self.policy.forced_mode() {
            return mode;
        }
        match *self.phases.entry(phase).or_default() {
            PhaseState::Undecided(state) => state.next_mode(),
            PhaseState::Decided(mode) => mode,
        }
    }

    pub fn record(
        &mut self,
        phase: HeavyPhase,
        mode: ExecutionMode,
        normalized_duration: f64,
    ) -> Option<ExecutionDecision> {
        if self.policy != ExecutionPolicy::Auto {
            return None;
        }
        if !normalized_duration.is_finite() || normalized_duration < 0.0 {
            return None;
        }

        let state = self.phases.entry(phase).or_default();
        let PhaseState::Undecided(undecided) = state else {
            return None;
        };

        undecided.stats_mut(mode).record(normalized_duration);
        let decision =
            undecided.significant_winner(self.threshold_stddevs, self.samples_per_mode)?;
        *state = PhaseState::Decided(decision.mode);
        Some(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_trials(
        planner: &mut AdaptiveExecutionPlanner,
        sequential: &[f64],
        rayon: &[f64],
    ) -> Option<ExecutionDecision> {
        let mut decision = None;
        for (&sequential, &rayon) in sequential.iter().zip(rayon) {
            decision = planner
                .record(
                    HeavyPhase::Preprocessing,
                    ExecutionMode::Sequential,
                    sequential,
                )
                .or(decision);
            decision = planner
                .record(HeavyPhase::Preprocessing, ExecutionMode::Rayon, rayon)
                .or(decision);
        }
        decision
    }

    #[test]
    fn decides_sequential_when_it_is_significantly_faster() {
        let mut planner = AdaptiveExecutionPlanner::new(2.0, 3, ExecutionPolicy::Auto);
        let decision = record_trials(&mut planner, &[10.0, 10.2, 9.8], &[20.0, 20.2, 19.8]);

        let decision = decision.unwrap();
        assert_eq!(decision.mode, ExecutionMode::Sequential);
        assert!((decision.speedup - 2.0).abs() < 1e-9);
        assert_eq!(
            planner.mode_for(HeavyPhase::Preprocessing),
            ExecutionMode::Sequential
        );
    }

    #[test]
    fn decides_rayon_when_it_is_significantly_faster() {
        let mut planner = AdaptiveExecutionPlanner::new(2.0, 3, ExecutionPolicy::Auto);
        let decision = record_trials(&mut planner, &[20.0, 20.2, 19.8], &[10.0, 10.2, 9.8]);

        let decision = decision.unwrap();
        assert_eq!(decision.mode, ExecutionMode::Rayon);
        assert!((decision.speedup - 2.0).abs() < 1e-9);
    }

    #[test]
    fn remains_undecided_without_a_significant_difference() {
        let mut planner = AdaptiveExecutionPlanner::new(2.0, 3, ExecutionPolicy::Auto);
        let decision = record_trials(&mut planner, &[10.0, 12.0, 8.0], &[10.5, 8.5, 11.5]);

        assert_eq!(decision, None);
    }

    #[test]
    fn alternates_modes_while_undecided() {
        let mut planner = AdaptiveExecutionPlanner::new(2.0, 3, ExecutionPolicy::Auto);

        assert_eq!(
            planner.mode_for(HeavyPhase::Preprocessing),
            ExecutionMode::Sequential
        );
        planner.record(HeavyPhase::Preprocessing, ExecutionMode::Sequential, 10.0);
        assert_eq!(
            planner.mode_for(HeavyPhase::Preprocessing),
            ExecutionMode::Rayon
        );
    }

    #[test]
    fn waits_for_configured_samples_per_mode() {
        let mut planner = AdaptiveExecutionPlanner::new(0.0, 4, ExecutionPolicy::Auto);

        assert_eq!(
            record_trials(&mut planner, &[10.0, 10.0, 10.0], &[20.0, 20.0, 20.0],),
            None
        );
        assert_eq!(
            record_trials(&mut planner, &[10.0], &[20.0]),
            Some(ExecutionDecision {
                mode: ExecutionMode::Sequential,
                speedup: 2.0,
            })
        );
    }

    #[test]
    fn forced_policy_skips_measurement() {
        let mut planner = AdaptiveExecutionPlanner::new(2.0, 3, ExecutionPolicy::SingleCore);

        assert_eq!(
            planner.mode_for(HeavyPhase::Preprocessing),
            ExecutionMode::Sequential
        );
        assert_eq!(
            planner.record(HeavyPhase::Preprocessing, ExecutionMode::Sequential, 10.0,),
            None
        );
    }
}
