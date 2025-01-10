use std::{
    collections::BTreeMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use burn::{
    prelude::Backend,
    tensor::{Float, Int, Tensor, TensorData},
};

use crate::{traits::ToTensor, util::summary_from_keys};

pub trait EnvState: Into<TensorData> {}
impl<T: Into<TensorData>> EnvState for T {}

pub trait EnvAction: Into<TensorData> + From<isize> {}
impl<T: Into<TensorData> + From<isize>> EnvAction for T {}

impl<B, T> ToTensor<B, 2, Float> for Vec<T>
where
    B: Backend,
    T: EnvState,
{
    fn to_tensor(self) -> Tensor<B, 2, Float> {
        let inner_data = self.into_iter().map(|x| x.into()).collect::<Vec<_>>();

        let outer_dim = inner_data.len();
        let inner_dim = inner_data
            .get(0)
            .map(TensorData::num_elements)
            .expect("cannot convert data with shape 0");

        let dtype = inner_data.get(0).map(|x| x.dtype).unwrap();
        let len = inner_data.get(0).map(|x| x.bytes.len()).unwrap();

        let mut bytes = Vec::with_capacity(outer_dim * len);
        for mut elem in inner_data {
            bytes.append(&mut elem.bytes);
        }

        Tensor::from(TensorData {
            bytes,
            shape: vec![outer_dim, inner_dim],
            dtype,
        })
    }
}

impl<B, T> ToTensor<B, 2, Int> for Vec<T>
where
    B: Backend,
    T: EnvAction,
{
    fn to_tensor(self) -> Tensor<B, 2, Int> {
        let inner_data = self.into_iter().map(|x| x.into()).collect::<Vec<_>>();

        let outer_dim = inner_data.len();
        let inner_dim = inner_data
            .get(0)
            .map(TensorData::num_elements)
            .expect("cannot convert data with shape 0");

        let dtype = inner_data.get(0).map(|x| x.dtype).unwrap();
        let len = inner_data.get(0).map(|x| x.bytes.len()).unwrap();

        let mut bytes = Vec::with_capacity(outer_dim * len);
        for mut elem in inner_data {
            bytes.append(&mut elem.bytes);
        }

        let tensor: Tensor<B, 1, Int> = Tensor::from(TensorData {
            bytes,
            shape: vec![outer_dim * inner_dim],
            dtype,
        });
        tensor.unsqueeze_dim(1)
    }
}

/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the common case of a discrete-time MDP with one agent.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone` - When sampling batches of experiences, cloning is necessary
    type State: Clone + Debug + EnvState;

    /// A representation of an action that an agent can take to affect the environment
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone` - When sampling batches of experiences, cloning is necessary
    type Action: Clone + Debug + EnvAction;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;

    /// Select a random action from the action space
    fn random_action(&self) -> Self::Action;

    /// Determine if the environment is in an active or terminal state
    fn is_active(&self) -> bool {
        true
    }
}

/// An [Environment] with a discrete action space
pub trait DiscreteActionSpace: Environment {
    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> Vec<Self::Action>;
}

/// An [Environment] with a discrete state space
pub trait DiscreteStateSpace: Environment {
    /// Get all possible states in the environment
    fn states(&self) -> Vec<Self::State>;
}

/// An [Environment] with a deterministic model
pub trait DeterministicModel: Environment {
    /// Get the next state and reward given the provided state and action
    fn model(&self, state: Self::State, action: Self::Action) -> (Option<Self::State>, f32);
}

/// An [Environment] with known dynamics
pub trait KnownDynamics: Environment {
    /// The dynamics of the environment
    ///
    /// p(s', r | s, a)
    ///
    /// This function returns the probability of transitioning to `next_state` and receiving `reward`
    /// after taking `action` in `state`.
    fn dynamics(
        &self,
        state: Self::State,
        action: Self::Action,
        next_state: Self::State,
        reward: f32,
    ) -> f32;
}

/// A format for reporting training results to [viz](crate::viz)
///
/// Functionally a wrapper around a [BTreeMap] such that values are always returned in the same order.
/// Meant to be initialized once and used for the lifetime of an [Environment].
///
/// See examples for implementation
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    keys: Vec<&'static str>,
    map: BTreeMap<&'static str, f64>,
}

impl Report {
    /// Create a new report format
    pub fn new(mut keys: Vec<&'static str>) -> Self {
        keys.sort_unstable();
        let map = summary_from_keys(&keys);
        Self { keys, map }
    }

    /// Get keys as a slice
    pub fn keys(&self) -> &[&'static str] {
        &self.keys
    }

    /// Take the report by extracting the inner map and leaving a default
    pub fn take(&mut self) -> BTreeMap<&'static str, f64> {
        std::mem::replace(&mut self.map, summary_from_keys(&self.keys))
    }
}

impl Deref for Report {
    type Target = BTreeMap<&'static str, f64>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for Report {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) struct MockEnv;

    impl Environment for MockEnv {
        type State = i32;
        type Action = i32;

        fn step(&mut self, _action: Self::Action) -> (Option<Self::State>, f32) {
            (None, 0.0)
        }

        fn reset(&mut self) -> Self::State {
            0
        }

        fn random_action(&self) -> Self::Action {
            0
        }
    }

    #[test]
    fn report_functional() {
        let mut report = Report::new(vec!["c", "a", "b"]);
        assert_eq!(
            *report.keys(),
            ["a", "b", "c"],
            "Keys were sorted on initialization"
        );

        report.entry("a").and_modify(|x| *x += 1.0);
        assert_eq!(
            *report.get("a").unwrap(),
            1.0,
            "Mutations on entries work and report derefs into inner map"
        );

        let inner_map = report.take();
        assert!(
            inner_map.values().eq([1.0, 0.0, 0.0].iter()),
            "Inner map can be taken with correct values"
        );
        assert!(
            report.values().eq([0.0, 0.0, 0.0].iter()),
            "Taking inner map leaves default values in report"
        );
    }
}
