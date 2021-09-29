use casper_types::Key;

use super::op::Op;
use crate::shared::{
    additive_map::AdditiveMap, execution_journal::ExecutionJournal, transform::Transform,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionEffect {
    pub ops: AdditiveMap<Key, Op>,
    pub transforms: AdditiveMap<Key, Transform>,
}

impl ExecutionEffect {
    pub fn new(ops: AdditiveMap<Key, Op>, transforms: AdditiveMap<Key, Transform>) -> Self {
        ExecutionEffect { ops, transforms }
    }
}

impl From<ExecutionJournal> for ExecutionEffect {
    fn from(journal: ExecutionJournal) -> Self {
        let mut ops = AdditiveMap::new();
        let mut transforms = AdditiveMap::new();
        let journal: Vec<(Key, Transform)> = journal.into();
        for (key, transform) in journal.into_iter() {
            let op = match transform {
                Transform::Failure(_) => continue,
                Transform::Identity => Op::Read,
                Transform::Write(_) => Op::Write,
                Transform::AddInt32(_)
                | Transform::AddUInt64(_)
                | Transform::AddUInt128(_)
                | Transform::AddUInt256(_)
                | Transform::AddUInt512(_)
                | Transform::AddKeys(_) => Op::Add,
            };
            ops.insert_add(key, op);
            transforms.insert_add(key, transform);
        }

        Self { ops, transforms }
    }
}
